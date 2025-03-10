'''
Author: Matthew Sun
Description: Implementing training script for a simple RB-VAE,
purpose is to figure out the details of the architecture
'''

### IMPORTS
import os
from pathlib import Path
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from contrastive_RBVAE_model import Seq2SeqBinaryVAE
###

### LOSS FUNCTIONS ###

def l1_loss(q_logits, lamb):
    # This should only be used if there is only 1 logit per latent variable
    return lamb * torch.norm(q_logits, p=1)

def recon_loss(x_recon, x):
    return F.mse_loss(x_recon, x)

def js_loss(p_log, q_log, log_target=True, reduction='batchmean'):
    """
    A helper that does:
    0.5 * KL(p||m) + 0.5 * KL(q||m)
    with p, q in log space if log_target=True.
    """
    # p_log, q_log shape: (..., K) where K is #categories
    # m_log shape: same
    # We'll use F.kl_div with log_target=True to interpret second arg as log-prob.
    return 0.5 * (F.kl_div(m_log, p_log, log_target=True, reduction=reduction)
        + F.kl_div(m_log, q_log, log_target=True, reduction=reduction))

def js_distance_for_bernoulli(p, q, eps=1e-8, reduction='none'):
    """
    p, q have shape (batch_size, latent_dim).
    Each entry is the probability of "on" for a Bernoulli variable.
    Returns a scalar that is the mean Jensen–Shannon *distance* across all dims in the batch.
    """
    # Clamps for safety :)
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)

    # Build 2-category distributions since we have variable that implicitly represents 1-p
    # shape => (batch_size, latent_dim, 2)
    p_2d = torch.stack([p, 1 - p], dim=-1)
    q_2d = torch.stack([q, 1 - q], dim=-1)

    # Convert to log
    p_2d_log = p_2d.log()
    q_2d_log = q_2d.log()

    # Compute log(m) = log(0.5 * (p_2d + q_2d))
    m_2d = 0.5 * (p_2d + q_2d)
    m_2d_log = m_2d.log()

    # Jensen–Shannon divergence, dimension by dimension, produce shape (batch_size, latent_dim)
    # We'll do 'reduction="none"' so we can see the result per (batch, dim).
    # (PyTorch usually expects the last dimension is categories)
    kl_p_m = F.kl_div(m_2d_log, p_2d_log, log_target=True, reduction=reduction)
    kl_q_m = F.kl_div(m_2d_log, q_2d_log, log_target=True, reduction=reduction)
    js_div_per_dim = 0.5 * (kl_p_m + kl_q_m)  # shape (batch_size, latent_dim)

    js_div_batch = js_div_per_dim.mean(dim=-1)  # average across latent_dim
    js_div = js_div_batch.mean(dim=0)           # average across batch

    # Turn divergence into distance
    js_distance = torch.sqrt(js_div + 1e-12)

    return js_distance

def triplet_loss(anchor, pos, neg, margin=1.0, p=2.0, eps=1e-08, swap=True, 
    size_average=None, reduce=None, reduction='mean'):

    return F.triplet_margin_loss(
        anchor=anchor, 
        positive=pos, 
        negative=neg, 
        margin=margin, 
        p=p, 
        eps=eps, 
        swap=swap, 
        size_average=size_average, 
        reduce=reduce, 
        reduction=reduction
        )


def triplet_loss_js(anchor, positive, negative, margin=1.0, eps=1e-8, swap=False):
    """
    Triplet loss that uses the JS distance as the distance measure.
    """
    # Compute the JS distance between anchor & positive, anchor & negative
    dist_ap = js_distance_for_bernoulli(anchor, positive, eps=eps, reduction='none')  # shape (batch, ...)
    dist_an = js_distance_for_bernoulli(anchor, negative, eps=eps, reduction='none')  # shape (batch, ...)

    if swap:
        # Compute distance between positive and negative
        dist_pn = js_distance_for_bernoulli(positive, negative, eps=eps, reduction='none')
        # Use the smaller negative distance (anchor-negative or positive-negative)
        dist_neg = torch.minimum(dist_an, dist_pn)
    else:
        dist_neg = dist_an

    # Standard Triplet Loss: max(0, dist_ap - dist_an + margin)
    loss = F.relu(dist_ap - dist_an + margin)

    # Average over batch (and any other dims except the distribution dim)
    return loss.mean()


def kl_binary_concrete(q_logits, p=0.5, eps=1e-8):
    '''
    Calculates the KL Divergence between the Bernoulli implied by q_logits 
    (via Binary Concrete relaxation) and a fixed Bernoulli(p). 
    Assumes q_logits has shape (..., latent_dim).
    '''
    # Convert logits to probabilities, clamp to avoid log(0).
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)

    # Precompute log(p) and log(1 - p) for a scalar p
    log_p = np.log(p)
    log_1_minus_p = np.log(1.0 - p)

    # Compute KL for each dimension:
    # KL(Bernoulli(q) || Bernoulli(p)) = q * log(q/p) + (1-q) * log((1-q)/(1-p))
    # Add eps inside the log for extra safety, because the latent variables were
    # always at the extremes (0 or 1)
    kl = q * (torch.log(q + eps) - log_p) \
       + (1.0 - q) * (torch.log((1.0 - q) + eps) - log_1_minus_p)

    # Sum over the latent dimension, then mean over the batch
    kl = kl.sum(dim=-1)
    kl_mean = kl.mean()

    return kl_mean


def contrast_loss(x1, x2, label, margin: float=1.0):
    """
    Computes Contrastive Loss. Requires an input label to determine difference
    between classes (0 for similar, 1 for dissimilar).
    """
    dist = F.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss

# Callable image transform
RESOLUTION = 256
ImageTransforms = T.Compose([
    T.Resize((RESOLUTION, RESOLUTION)),
    T.ToTensor()
])

### DATASETS ###

# Legacy, still seems potentially useful who knows...
class SampleStatePairDataset(Dataset):
    '''
    Args:
        frames_dir: Path to the directory containing the frames.
        state_segments: List of tuples (start_idx, end_idx) determining state groupings.
        transform: Transform to be applied on each frame.
        num_items: Number of items that this Dataset will yield.
    '''
    def __init__(self, frames_dir, state_segments, transform=None, num_items=1000):
        self.frames_dir = frames_dir
        self.state_segments = state_segments
        self.transform = transform
        self.num_items = num_items
        self.num_states = len(self.state_segments)
        self.state_frame_indices = []
        for (start, end) in self.state_segments:
            self.state_frame_indices.append(list(range(start, end)))
        self.max_length = max([len(indices) for indices in self.state_frame_indices])

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        pairs = []
        for state_index in range(self.num_states):
            frame_indices = self.state_frame_indices[state_index]
            if len(frame_indices) == 1:
                index1 = index2 = 0
            else:
                index1, index2 = random.sample(frame_indices, 2)
            frame1 = self._load_frame(index1)
            frame2 = self._load_frame(index2)
            pairs.append(torch.stack([frame1, frame2], dim=0))  # [2, C, H, W]
        pairs_tensor = torch.stack(pairs, dim=0)  # [T, 2, C, H, W]
        pairs_tensor = pairs_tensor.permute(1, 0, 2, 3, 4)  # [2, T, C, H, W]
        return pairs_tensor

    def _load_frame(self, frame_index):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(self.frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

# TODO: rework this to include validation and test splits
'''
How exactly to do this though...?
I'm thinking to just take a contiguous segment of X% of frames from each state
'''

class ShuffledStatePairDataset(Dataset):
    """
    Args:
        frames_dir:      Directory with frame images named like 0000000001.jpg
        state_segments:  List of (start_idx, end_idx) for each state
        test_pct:        Fraction of total frames (per state) to devote to test
        val_pct:         Fraction of total frames (per state) to devote to val
        transform:       Any image transform (e.g. torchvision transforms)
        mode:            One of ["train", "test", "val"] – determines which subset
                         of indices (train vs. test vs. val) is served by __getitem__.
    """
    def __init__(
        self, 
        frames_dir, 
        state_segments, 
        test_pct=0.1, 
        val_pct=0.1, 
        transform=None, 
        mode="train"
    ):
        super().__init__()
        self.frames_dir = frames_dir
        self.state_segments = state_segments
        self.transform = transform
        self.mode = mode.lower().strip()
        self.num_states = len(self.state_segments)

        # Keep track of train/test/val indices for each state
        self.train_indices_per_state = []
        self.test_indices_per_state = []
        self.val_indices_per_state = []

        # For building your shuffled pairs, you may want to keep them
        # separate so you only sample pairs from the relevant subset:
        self.pairs_per_state = []

        # Compute the contiguous splits for each state
        for (start, end) in self.state_segments:
            full_indices = list(range(start, end)) 
            n = len(full_indices)
            # Size of combined test+val chunk
            test_val_count = int(n * (test_pct + val_pct))
            # The offset from the start to that "middle" chunk
            margin = (n - test_val_count) // 2

            # Middle chunk that will be test+val
            test_val_indices = full_indices[margin : margin + test_val_count]
            
            # The remaining frames (front chunk + back chunk) => train set
            train_indices = (full_indices[:margin] 
                             + full_indices[margin + test_val_count:])

            # Now split test+val portion. For example, if you want them
            # split proportionally to test_pct vs. val_pct:
            if test_val_count > 0:
                # fraction that is test out of that middle chunk
                test_fraction_of_middle = test_pct / (test_pct + val_pct)
                test_count = int(round(test_fraction_of_middle * test_val_count))
                test_indices  = test_val_indices[:test_count]
                val_indices   = test_val_indices[test_count:]
            else:
                # if there's no middle chunk at all, they are empty
                test_indices = []
                val_indices = []

            self.train_indices_per_state.append(train_indices)
            self.test_indices_per_state.append(test_indices)
            self.val_indices_per_state.append(val_indices)

        # At this point, we have the train/test/val indices for each state.
        # If you want to shuffle/pad/draw pairs from only one subset (e.g. train),
        # do that below by focusing on self.train_indices_per_state, etc.
        self._build_pairs()

    def _build_pairs(self):
        """
        Builds the pairs for each state, depending on which subset we're using
        (train, test, or val). By default, we'll just build pairs for the
        specified `self.mode`. If you want separate pair-lists for each split,
        you can do that too.
        """
        if self.mode == "train":
            all_state_indices = self.train_indices_per_state
        elif self.mode == "test":
            all_state_indices = self.test_indices_per_state
        elif self.mode == "val":
            all_state_indices = self.val_indices_per_state
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        self.pairs_per_state = []
        self.num_items = 0

        max_frames = 0
        for indices in all_state_indices:
            max_frames = max(max_frames, len(indices))

        for indices in all_state_indices:
            # If there are fewer frames than max_frames, pad them
            if len(indices) < max_frames and len(indices) > 0:
                padded = indices.copy() + random.choices(indices, k=max_frames - len(indices))
            else:
                padded = indices.copy()

            random.shuffle(padded)
            # Now form (frame1, frame2) pairs
            pairs = []
            for i in range(len(padded) // 2):
                pairs.append((padded[2*i], padded[2*i+1]))

            if len(padded) % 2 == 1:
                leftover = padded[-1]
                # pick any other index from 'indices' if possible
                if len(indices) > 1:
                    candidate = random.choice([x for x in indices if x != leftover])
                else:
                    candidate = leftover
                pairs.append((leftover, candidate))

            self.pairs_per_state.append(pairs)

        # The number of items is basically the number of pairs each state can supply
        # They should all have the same number of pairs
        max_pairs = max(len(pairs_list) for pairs_list in self.pairs_per_state)
        self.num_items = max_pairs

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        """
        Collects the (idx)-th pair from each state's pair list, wraps them
        into a single tensor. If a state's pair list is shorter than idx,
        we wrap around mod the length (or handle however you prefer).
        """

        pairs = []
        for pairs_list in self.pairs_per_state:
            if len(pairs_list) == 0:
                raise ValueError(f"State {idx} has no pairs")

            real_pair_idx = idx % len(pairs_list)
            pair = pairs_list[real_pair_idx]
            frame1 = self._load_frame(pair[0])
            frame2 = self._load_frame(pair[1])
            pairs.append(torch.stack([frame1, frame2], dim=0))  # [2, C, H, W]

        # Combine [state_1, state_2, ...] => [2, T, C, H, W]
        pairs_tensor = torch.stack(pairs, dim=0).permute(1, 0, 2, 3, 4)
        return pairs_tensor

    def _load_frame(self, frame_index):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(self.frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class ContrastiveRBVAETrainer:
    def __init__(
        self,
        model,
        device,
        train_dataloader,
        val_dataloader,
        optimizer,
        init_temperature=1.0,
        final_temperature=0.5,
        anneal_rate=1e-4,
        num_steps_to_update=100,
        bernoulli_p=0.5,
        margin=1.0,
        alpha_contrast=0.1,
        beta_kl=0.1,
        log_dir=None
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        
        # Training parameters
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.anneal_rate = anneal_rate
        self.num_steps_to_update = num_steps_to_update
        self.bernoulli_p = bernoulli_p
        self.margin = margin
        self.alpha_contrast = alpha_contrast
        self.beta_kl = beta_kl
        
        # Initialize tensorboard writer if log_dir is provided
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_one_epoch(self, epoch):
        """Trains the model for one epoch and returns average losses."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_contrast_loss = 0.0
        
        num_batches = len(self.train_dataloader)
        temperature = self.init_temperature
        
        for batch_idx, item in enumerate(self.train_dataloader):
            global_step = epoch * num_batches + batch_idx + 1
            
            # Update temperature
            if global_step % self.num_steps_to_update == 0:
                temperature = max(
                    self.final_temperature,
                    self.init_temperature * np.exp(-self.anneal_rate * global_step)
                )
                if self.writer:
                    self.writer.add_scalar('Batch/Temperature', temperature, global_step)
            
            num_batches_item, _, num_states, _, _, _ = item.size()
            item = item.to(self.device)
            frames = [item[:, i] for i in range(2)]
            recon_losses = []
            kl_losses = []
            bc_seqs = []
            h_seqs = []
            
            for frame in frames:
                x_recon, h_seq, bc_seq = self.model(frame, temperature=temperature, hard=False)
                recon_losses.append(recon_loss(x_recon, frame))
                kl_losses.append(kl_binary_concrete(bc_seq, p=self.bernoulli_p))
                h_seqs.append(h_seq)
            
            recon_loss_val = sum(recon_losses) / len(recon_losses)
            kl_loss_val = sum(kl_losses) / len(kl_losses)
            
            contrast_loss_similar = contrast_loss(h_seqs[0], h_seqs[1], label=0)
        
            contrast_loss_dissim = 0
            for state_index in range(num_states - 1):
                dissim_z_a = h_seqs[0][:, state_index]
                dissim_z_b = h_seqs[0][:, state_index + 1]
                contrast_loss_dissim += contrast_loss(dissim_z_a, dissim_z_b, label=1)
            contrast_loss_dissim /= float(num_states - 1)
            
            contrast_loss_val = contrast_loss_similar + contrast_loss_dissim
            
            total_loss_val = (
                recon_loss_val +
                self.beta_kl * kl_loss_val +
                self.alpha_contrast * contrast_loss_val
            )
            
            self.optimizer.zero_grad()
            total_loss_val.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_val.item()
            total_recon_loss += recon_loss_val.item()
            total_kl_loss += kl_loss_val.item()
            total_contrast_loss += contrast_loss_val.item()
            
            # Log batch metrics
            if self.writer:
                self.writer.add_scalar('Batch/Total_Loss', total_loss_val.item(), global_step)
                self.writer.add_scalar('Batch/Reconstruction_Loss', recon_loss_val.item(), global_step)
                self.writer.add_scalar('Batch/KL_Divergence', kl_loss_val.item(), global_step)
                self.writer.add_scalar('Batch/Contrast_Loss', contrast_loss_val.item(), global_step)
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'contrast_loss': total_contrast_loss / num_batches
        }
        
        return avg_losses

    def validate(self):
        """Evaluates the model on validation data and returns average losses."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_contrast_loss = 0.0
        
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for item in self.val_dataloader:
                num_batches_item, _, num_states, _, _, _ = item.size()
                item = item.to(self.device)
                frames = [item[:, i] for i in range(2)]
                recon_losses = []
                kl_losses = []
                bc_seqs = []
                h_seqs = []
                
                for frame in frames:
                    x_recon, h_seq, bc_seq = self.model(
                        frame,
                        temperature=self.final_temperature,
                        hard=False
                    )
                    recon_losses.append(recon_loss(x_recon, frame))
                    kl_losses.append(kl_binary_concrete(bc_seq, p=self.bernoulli_p))
                    h_seqs.append(h_seq)
                
                recon_loss_val = sum(recon_losses) / len(recon_losses)
                kl_loss_val = sum(kl_losses) / len(kl_losses)
                
                contrast_loss_similar = contrast_loss(h_seqs[0], h_seqs[1], label=0)
        
                contrast_loss_dissim = 0
                for state_index in range(num_states - 1):
                    dissim_z_a = h_seqs[0][:, state_index]
                    dissim_z_b = h_seqs[0][:, state_index + 1]
                    contrast_loss_dissim += contrast_loss(dissim_z_a, dissim_z_b, label=1)
                contrast_loss_dissim /= float(num_states - 1)
                
                contrast_loss_val = contrast_loss_similar + contrast_loss_dissim
                
                total_loss_val = (
                    recon_loss_val +
                    self.beta_kl * kl_loss_val +
                    self.alpha_contrast * contrast_loss_val
                )
                
                total_loss += total_loss_val.item()
                total_recon_loss += recon_loss_val.item()
                total_kl_loss += kl_loss_val.item()
                total_contrast_loss += contrast_loss_val.item()
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'contrast_loss': total_contrast_loss / num_batches
        }
        
        return avg_losses

    def train(self, num_epochs, save_path=None):
        """
        Trains the model for the specified number of epochs.
        Args:
            num_epochs: Number of epochs to train
            save_path: Optional path to save the best model
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_losses = self.train_one_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Log epoch metrics
            if self.writer:
                for key, value in train_losses.items():
                    self.writer.add_scalar(f'Epoch/Train_{key}', value, epoch)
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'Epoch/Val_{key}', value, epoch)
            
            # Save best model
            if val_losses['total_loss'] < history['best_val_loss']:
                history['best_val_loss'] = val_losses['total_loss']
                history['best_epoch'] = epoch
                self.best_model_state = self.model.state_dict()
                
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_losses['total_loss'],
                    }, save_path)
            
            # Store losses
            history['train_losses'].append(train_losses)
            history['val_losses'].append(val_losses)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train - Total: {train_losses['total_loss']:.4f}, Recon: {train_losses['recon_loss']:.4f}, "
                  f"KL: {train_losses['kl_loss']:.4f}, Contrast: {train_losses['contrast_loss']:.4f}")
            print(f"Val - Total: {val_losses['total_loss']:.4f}, Recon: {val_losses['recon_loss']:.4f}, "
                  f"KL: {val_losses['kl_loss']:.4f}, Contrast: {val_losses['contrast_loss']:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return history

if __name__ == "__main__":
    # Set up paths and state segmentation
    frames_dir = Path(__file__).parent.parent.parent.joinpath("videos/frames/kid_playing_with_blocks_1.mp4")
    last_frame = 1425
    flags = [152, 315, 486, 607, 734, 871, 1153, 1343]
    grey_out = 10
    
    # Create state segments
    state_segments = []
    for i in range(len(flags)):
        if i > 0:
            state_segments.append((flags[i-1] + grey_out, flags[i] - grey_out + 1))
        elif i == len(flags)-1:
            state_segments.append((flags[i] + grey_out, last_frame + 1))
        else:
            state_segments.append((0, flags[0] - grey_out + 1))
    
    # Setup datasets and dataloaders
    batch_size = 32
    train_dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms, mode="train")
    val_dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms, mode="val")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup model and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    # Create trainer
    trainer = ContrastiveRBVAETrainer(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        init_temperature=1.0,
        final_temperature=0.5,
        anneal_rate=1e-3,
        num_steps_to_update=int((num_epochs * len(train_dataset)) / 750),
        bernoulli_p=0.1,
        margin=0.2,
        alpha_contrast=1.0,
        beta_kl=1.0,
        log_dir="./runs/rb_vae_experiment"
    )
    
    # Train the model
    save_path = Path(__file__).parent.joinpath("saved_RBVAE")
    history = trainer.train(num_epochs=num_epochs, save_path=save_path)
    
    print(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
    print("Run 'tensorboard --logdir=runs' to visualize the training progress.")