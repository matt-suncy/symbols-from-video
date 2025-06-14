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
# Sometimes I need to import stuff from this file and I just don't care if the model's path lines up or not
try:
    from contrastive_RBVAE_model import Seq2SeqBinaryVAE
except:
    print("Warning: Seq2SeqBinaryVAE was NOT successfully imported.")
###

### LOSS FUNCTIONS ###

def l1_loss(q_logits, lamb):
    # This should only be used if there is only 1 logit per latent variable
    return lamb * torch.norm(q_logits, p=1)

def recon_loss(x_recon, x):
    return F.mse_loss(x_recon, x)

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


def contrast_loss(x1, x2, label, margin: float=1.0, dist='euclidean'):
    """
    Computes Contrastive Loss using cosine or euclidean distance. Requires an input label to determine difference
    between classes (0 for similar, 1 for dissimilar).
    
    Args:
        x1: First input tensor
        x2: Second input tensor 
        label: Binary label (0 for similar pairs, 1 for dissimilar pairs)
        margin: Margin for dissimilar pairs (default: 1.0)
        dist: Distance metric to use ('cosine' or 'euclidean')
    Returns:
        Contrastive loss value
    """
    # Cosine similarity is between -1 and 1, so cosine distance is between 0 and 2
    if dist == 'cosine':
        cos_sim = F.cosine_similarity(x1, x2)
        dist = 1 - cos_sim  # Convert similarity to distance (0 to 2 range)

    elif dist == 'euclidean':
        dist = F.pairwise_distance(x1, x2)

    # For similar pairs (label=0), we want to minimize the distance
    # For dissimilar pairs (label=1), we want distance > margin
    similar_loss = (1 - label) * torch.pow(dist, 2)
    dissimilar_loss = label * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    
    loss = similar_loss + dissimilar_loss
    return loss.mean()

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


def assign_label(frame_index, flags):
    """
    Assigns a class label based on the provided flag indices.
    Frames before the first flag are label 0, between the first and second flag are label 1, etc.
    """
    label = 0
    for f in flags:
        if frame_index >= f:
            label += 1
        else:
            break
    return label

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
        noise_ratio=0.1,
        margin=1.0,
        alpha_contrast=0.1,
        beta_kl=0.1,
        log_dir=None,
        flags=None
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
        self.noise_ratio = noise_ratio
        self.margin = margin
        self.alpha_contrast = alpha_contrast
        self.beta_kl = beta_kl
        self.flags = flags
        
        # Initialize tensorboard writer if log_dir is provided
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        
        # Track best validation metric
        self.best_val_metric = float('-inf')  # Changed to -inf since we want to maximize consistency
        self.best_model_state = None

        # Initialize temperature state
        self.current_temperature = init_temperature
        self.global_step = 0

    def get_current_temperature(self):
        """
        Calculate the current temperature based on the global step.
        Uses both anneal_rate and num_steps_to_update to control temperature decay.
        """
        # Only update temperature at specified intervals
        if self.global_step % self.num_steps_to_update == 0:
            # Calculate temperature using exponential decay
            self.current_temperature = max(
                self.final_temperature,
                self.init_temperature * np.exp(-self.anneal_rate * self.global_step)
            )
            
        return self.current_temperature

    def calculate_state_consistency(self, temperature):
        """
        Calculates the state consistency metric using validation data.
        Returns the weighted average consistency across all states.
        """
        self.model.eval()
        val_dataset = self.val_dataloader.dataset
        
        latent_vectors = []
        labels = []
        
        # Get all validation indices
        val_indices = []
        for indices in val_dataset.val_indices_per_state:
            val_indices.extend(indices)
        
        with torch.no_grad():
            for idx in val_indices:
                # Load and transform the frame
                filename = f"{idx:010d}.jpg"
                path = os.path.join(val_dataset.frames_dir, filename)
                image = Image.open(path).convert("RGB")
                if val_dataset.transform is not None:
                    frame = val_dataset.transform(image)
                
                # Add batch and time dimensions
                input_tensor = frame[None, None, :, :, :].to(self.device)
                # Encode to get latent vector using current temperature
                latent = self.model.encode(input_tensor, temperature=temperature, hard=True, noise_ratio=self.noise_ratio)
                latent = latent.cpu().numpy().squeeze()
                latent_vectors.append(latent)
                # Assign label
                label = assign_label(idx, self.flags)
                labels.append(label)

        latent_vectors = np.array(latent_vectors)
        labels = np.array(labels)

        # Calculate consistency for each state
        percentages = []
        for label in range(len(self.flags) + 1):
            label_mask = labels == label
            label_vectors = latent_vectors[label_mask]
            
            if len(label_vectors) == 0:
                percentages.append(0.0)
                continue
            
            # Find most common embedding
            unique_vectors, counts = np.unique(label_vectors, axis=0, return_counts=True)
            most_common_vector = unique_vectors[np.argmax(counts)]
            
            # Calculate match percentage
            matches = np.all(label_vectors == most_common_vector, axis=1)
            percentage = np.mean(matches)
            percentages.append(percentage)

        # Calculate weighted average
        counts = [np.sum(labels == label) for label in range(len(self.flags) + 1)]
        total = sum(counts)
        weighted_avg = np.dot(percentages, counts) / total if total > 0 else 0

        return weighted_avg, percentages

    def train_one_epoch(self, epoch):
        """Trains the model for one epoch and returns average losses."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_contrast_loss = 0.0
        
        num_batches = len(self.train_dataloader)
        
        for batch_idx, item in enumerate(self.train_dataloader):
            self.global_step += 1
            temperature = self.get_current_temperature()
            
            # Log temperature if it was updated
            if self.global_step % self.num_steps_to_update == 0 and self.writer:
                self.writer.add_scalar('Batch/Temperature', temperature, self.global_step)
            
            num_batches_item, _, num_states, _, _, _ = item.size()
            item = item.to(self.device)
            frames = [item[:, i] for i in range(2)]
            recon_losses = []
            kl_losses = []
            bc_seqs = []
            h_seqs = []
            
            for frame in frames:
                x_recon, h_seq, bc_seq = self.model(frame, temperature=temperature, hard=False, noise_ratio=self.noise_ratio)
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
                self.writer.add_scalar('Batch/Total_Loss', total_loss_val.item(), self.global_step)
                self.writer.add_scalar('Batch/Reconstruction_Loss', recon_loss_val.item(), self.global_step)
                self.writer.add_scalar('Batch/KL_Divergence', kl_loss_val.item(), self.global_step)
                self.writer.add_scalar('Batch/Contrast_Loss', contrast_loss_val.item(), self.global_step)
        
        # Calculate average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'contrast_loss': total_contrast_loss / num_batches,
            'temperature': self.current_temperature  # Add current temperature to metrics
        }
        
        return avg_losses

    def validate(self):
        """Evaluates the model on validation data and returns metrics including state consistency."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_contrast_loss = 0.0
        
        num_batches = len(self.val_dataloader)
        
        # Calculate normalization coefficients
        coeff_sum = 1.0 + self.beta_kl + self.alpha_contrast
        recon_coeff = 1.0 / coeff_sum
        kl_coeff = self.beta_kl / coeff_sum  
        contrast_coeff = self.alpha_contrast / coeff_sum

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
                        hard=True,
                        noise_ratio=self.noise_ratio
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
                
                # Use normalized coefficients for total loss
                total_loss_val = (
                    recon_coeff * recon_loss_val +
                    kl_coeff * kl_loss_val +
                    contrast_coeff * contrast_loss_val
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
        
        # Calculate state consistency metric
        avg_losses['consistency_score'] = consistency_score
        
        # Add individual state consistencies to metrics
        for i, pct in enumerate(state_percentages):
            avg_losses[f'state_{i}_consistency'] = pct
        
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
            'best_consistency': float('-inf')  # Changed to track best consistency
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
            
            # Save best model based on consistency score
            if val_losses['consistency_score'] > history['best_consistency']:
                history['best_consistency'] = val_losses['consistency_score']
                history['best_epoch'] = epoch
                self.best_model_state = self.model.state_dict()
                
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'consistency_score': val_losses['consistency_score'],
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
            print(f"Consistency Score: {val_losses['consistency_score']:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return history

if __name__ == "__main__":
    # Set up paths and state segmentation
    # TODO: For IKEA ASM video, maybe just do no or very very little grey out?
    frames_dir = Path(__file__).parent.parent.parent.joinpath("videos/frames/ikea_asm_table")
    last_frame = 2469
    flags = [157, 205, 441, 494, 557, 887, 909, 1010, 1048, 1315, 1388, 1438, 1702, 1847, 2096, 2174]
    grey_out = 10
    
    # Create state segments
    state_segments = []
    for i in range(len(flags)):
        if i > 0:
            state_segments.append((flags[i-1] + grey_out + 1, flags[i] - grey_out))
        else:
            state_segments.append((0, flags[0] - grey_out))
    state_segments.append((flags[-1] + grey_out + 1, last_frame + 1))
    
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
        noise_ratio=0.1,
        margin=0.2,
        alpha_contrast=1.0,
        beta_kl=1.0,
        log_dir="./runs/rb_vae_experiment",
        flags=flags
    )
    
    # Train the model
    save_path = Path(__file__).parent.joinpath("saved_RBVAE")
    history = trainer.train(num_epochs=num_epochs, save_path=save_path)
    
    print(f"Best validation consistency score: {history['best_consistency']:.4f} at epoch {history['best_epoch']}")
    print("Run 'tensorboard --logdir=runs' to visualize the training progress.")