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

# TODO: Try L1 reg

def l1_loss(q_logits, lamb):
    # This should only be used if there is only 1 logit per latent variable
    # Also I have no idea if q_logits needs to be reshaped or something
    return lamb * torch.norm(q_logits, p=1)


def recon_loss(x_recon, x):
    return F.mse_loss(x_recon, x)


def kl_binary_concrete(q_logits, p=0.5, eps=1e-10):
    '''
    Calculates the KL Divergence between input logits and a 
    Bernoulli distribution
    '''
    # Squish it
    q = torch.sigmoid(q_logits)

    # KL( Bernoulli(q) || Bernoulli(p) ) = q * log(q/p) + (1-q) * log((1-q)/(1-p))
    # Not adding eps for p because it should never be that small
    # BTW q: tensor, p: float hence why we use np.log() for p
    kl = q * (torch.log(q + eps) - np.log(p)) \
        + (1.0 - q) * (torch.log(1.0 - q + eps) - np.log(1.0 - p))

    '''Suggestions from GPT-o1
    - Optionally: sum over latent dimensions, then average over the batch
    - If q_logits has shape [B, D], kl has shape [B, D].
    - We can sum over D and then take .mean() (average over batch B).
    '''

    kl = kl.sum(dim=-1)   # Sum over last dimension (latent_dim)
    kl_mean = kl.mean()   # Average over batch

    return kl_mean


def contrast_loss(x1, x2, label, margin: float=1.0):
    """
    Computes Contrastive Loss. Requires an input label to determine difference
    between classes (0 for similar, 1 for dissimilar).
    """
    dist = F.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss

# This is a CALLABLE
ImageTransforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

### DATASETS ###

# TODO: Change to shuffle --> pair up contiguous frames
class SampleStatePairDataset(Dataset):
    '''
    --- Args ---
    frames_dir: str
        Path to the directory containing the frames.
    state_segments: list of tuples
        List of tuples (start_idx, end_idx) determining state groupings.
    transform: callable, optional
        Transform to be applied each frame.
    num_items: int
        Number of items that this Dataset will yield.
        Each item has shape [2, T, C, H, W]
    '''
    def __init__(self, frames_dir, state_segments, transform=None, num_items=1000):
        self.frames_dir = frames_dir
        self.state_segments = state_segments
        self.transform = transform
        self.num_items = num_items
        self.num_states = len(self.state_segments)

        # Creates list of indices explicitly written out so we can sample
        self.state_frame_indices = []
        for (start, end) in self.state_segments:
            frame_indices = list(range(start, end))
            self.state_frame_indices.append(frame_indices)

        # Get number of frames in the longest state
        self.max_length = max([len(indices) for indices in self.state_segments])

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        '''
        Outputs a tensor of shape [2, T, C, H, W],
        where T is the number of states and 2 is the pair dimension.
        '''
        # Store tensors with shape [T, 2, C, H, W] FOR NOW
        pairs = []
        for state_index in range(self.num_states):
            frame_indices = self.state_frame_indices[state_index]

            # If the state has only 1 frame, pair it with itself (this should never happen though)
            if len(frame_indices) == 1:
                index1 = index2 = 0
            else:
                # Randomly choose 2 distinct frames from the current state
                index1, index2 = random.sample(frame_indices, 2)

            # Load and apply transform
            frame1 = self._load_frame(index1)
            frame2 = self._load_frame(index2)

            pairs.append(torch.stack([frame1, frame2], dim=0)) # [2, C, H, W]

        # Stack along dimension 0 to get a shape of [T, 2, C, H, W]
        pairs_tensor = torch.stack(pairs, dim=0)

        pairs_tensor = pairs_tensor.permute(1, 0, 2, 3, 4) # [2, T, C, H, W]
        '''
        NOTE: Honestly, there's no particular reason to have the pair dimension first.
        A shape of [2, T, C, H, W] just feels right to me.
        '''

    def _load_frame(self, frame_index):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(self.frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

class ShuffledStatePairDataset(Dataset):
    '''
    --- Args ---
    frames_dir: str
        Path to the directory containing the frames.
    state_segments: list of tuples
        List of tuples (start_idx, end_idx) determining state groupings.
    transform: callable, optional
        Transform to be applied each frame.
    num_items: int
        Number of items that this Dataset will yield.
        Each item has shape [2, T, C, H, W]
    '''
    def __init__(self, frames_dir, state_segments, transform=None):
        self.frames_dir = frames_dir
        self.state_segments = state_segments
        self.transform = transform
        self.num_states = len(self.state_segments)

        # Creates list of indices explicitly written out so we can sample
        self.state_frame_indices = []
        for (start, end) in self.state_segments:
            frame_indices = list(range(start, end))
            self.state_frame_indices.append(frame_indices)

        # Build a list of frame indices for each state and compute max length.
        self.state_frame_indices = []
        self.max_frames = 0
        for (start, end) in self.state_segments:
            frame_indices = list(range(start, end))
            self.state_frame_indices.append(frame_indices)
            self.max_frames = max(self.max_frames, len(frame_indices))

        # Pre-compute cause I don't want to shuffle every time getitem() gets called
        self.state_pairs = []
        for indices in self.state_frame_indices:
            # Pad if necessary
            if len(indices) < self.max_frames:
                padded = indices.copy() + random.choices(indices, k=self.max_frames - len(indices))
            else:
                padded = indices.copy()

            # Shuffle the padded indices once
            random.shuffle(padded)

            # Form contiguous pairs.
            pairs = []
            for i in range(len(padded) // 2):
                pairs.append((padded[2 * i], padded[2 * i + 1]))

            # Handle odd element.
            if len(padded) % 2 == 1:
                leftover = padded[-1]
                candidate = random.choice([x for x in indices if x != leftover]) if len(indices) > 1 else leftover
                pairs.append((leftover, candidate))

            self.state_pairs.append(pairs)

        self.num_items = self.max_frames // 2 if (self.max_frames % 2 == 0) \
            else (self.max_frames + 1) // 2

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        '''
        Outputs a tensor of shape [2, T, C, H, W],
        where T is the number of states and 2 is the pair dimension.
        '''
        pairs = []
        # For each state, select a pair based on the pre-computed pairs.
        for state_idx, pairs_list in enumerate(self.state_pairs):
            pair_idx = idx % len(pairs_list)
            pair = pairs_list[pair_idx]

            frame1 = self._load_frame(pair[0])
            frame2 = self._load_frame(pair[1])
            pairs.append(torch.stack([frame1, frame2], dim=0))  # [2, C, H, W]

        # Arrange to output shape [2, T, C, H, W]
        pairs_tensor = torch.stack(pairs, dim=0).permute(1, 0, 2, 3, 4)
        return pairs_tensor
        '''
        NOTE: Honestly, there's no particular reason to have the pair dimension first.
        A shape of [2, T, C, H, W] just feels right to me.
        '''

    def _load_frame(self, frame_index):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(self.frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

### TRAINING LOOP ###

def train_one_epoch(model, device, dataloader, optimizer, epoch, writer=None, 
                    temperature=0.5, bernoulli_p=0.5, margin=1.0, 
                    alpha_contrast=0.1, beta_kl=0.1):
    """
    Trains the model for one epoch and logs losses to TensorBoard.
    Args:
        model: The RB-VAE model.
        device: PyTorch device.
        dataloader: DataLoader yielding batches of shape [B, 2, T, C, H, W].
        optimizer: Optimizer for updating model parameters.
        writer: Optional TensorBoard SummaryWriter instance.
        epoch: Current epoch number (for logging).
        temperature: Temperature parameter for sampling.
        bernoulli_p: Target probability for the Bernoulli prior.
        margin: Margin for the contrastive loss.
        alpha_contrast: Weight for contrastive loss.
        beta_kl: Weight for KL divergence loss.
    Returns:
        A tuple of average losses: (total_loss, recon_loss, kl_loss, contrast_loss).
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_contrast_loss = 0.0

    num_batches = len(dataloader)
    for batch_idx, item in enumerate(dataloader):
        # item shape: [B, 2, T, C, H, W]
        global_step = epoch * num_batches + batch_idx
        num_batches_item, _, num_states, _, _, _ = item.size()
        item = item.to(device)
        frames = [item[:, i] for i in range(2)]
        recon_losses = []
        kl_losses = []
        z_seqs = []

        for frame in frames:
            x_recon, z_seq, logits = model(frame, temperature=temperature, hard=False)
            recon_losses.append(recon_loss(x_recon, frame))
            kl_losses.append(kl_binary_concrete(logits, p=bernoulli_p))
            z_seqs.append(z_seq)

        # Average losses over the two frame inputs.
        recon_loss_val = sum(recon_losses) / len(recon_losses)
        kl_loss_val = sum(kl_losses) / len(kl_losses)
        
        # Compute contrastive loss for the similar pair.
        contrast_loss_similar = contrast_loss(z_seqs[0], z_seqs[1], label=0)
        
        # Compute contrastive loss for dissimilar consecutive states.
        contrast_loss_dissim = 0
        for state_index in range(num_states - 1):
            dissim_z_a = z_seqs[0][:, state_index]
            dissim_z_b = z_seqs[0][:, state_index + 1]
            contrast_loss_dissim += contrast_loss(dissim_z_a, dissim_z_b, label=1)
        contrast_loss_dissim /= float(num_states - 1)
        
        contrast_loss_val = contrast_loss_similar + contrast_loss_dissim

        # Combine all loss components.
        total_loss_val = recon_loss_val + beta_kl * kl_loss_val + alpha_contrast * contrast_loss_val
        
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        total_loss += total_loss_val.item()
        total_recon_loss += recon_loss_val.item()
        total_kl_loss += kl_loss_val.item()
        total_contrast_loss += contrast_loss_val.item()

        # Log each batch loss to TensorBoard.
        if writer is not None:
            writer.add_scalar('Batch/Total_Loss', total_loss_val.item(), global_step)
            writer.add_scalar('Batch/Reconstruction_Loss', recon_loss_val.item(), global_step)
            writer.add_scalar('Batch/KL_Divergence', kl_loss_val.item(), global_step)
            writer.add_scalar('Batch/Contrastive_Loss', contrast_loss_val.item(), global_step)

    avg_total_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_contrast_loss = total_contrast_loss / num_batches

    # Log epoch-level average losses.
    if writer is not None:
        writer.add_scalar('Epoch/Total_Loss', avg_total_loss, epoch)
        writer.add_scalar('Epoch/Reconstruction_Loss', avg_recon_loss, epoch)
        writer.add_scalar('Epoch/KL_Divergence', avg_kl_loss, epoch)
        writer.add_scalar('Epoch/Contrastive_Loss', avg_contrast_loss, epoch)

    return avg_total_loss, avg_recon_loss, avg_kl_loss, avg_contrast_loss


if __name__ == "__main__":

    # Set up paths and state segmentation.
    frames_dir = Path(__file__).parent.parent.parent.joinpath("videos/frames/kid_playing_with_blocks_1.mp4")
    print(str(frames_dir))
    last_frame = 1425
    flags = [152, 315, 486, 607, 734, 871, 1153, 1343]
    grey_out = 25
    state_segments = []
    for i in range(len(flags)):
        if i > 0:
            state_segments.append((flags[i-1] + grey_out, flags[i] - grey_out + 1))
        elif i == len(flags)-1:
            state_segments.append((flags[i] + grey_out, last_frame + 1))
        else:
            state_segments.append((0, flags[0] - grey_out + 1))
        
    dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    temperature = 0.5 # Temperature for Softmax
    beta = 0.05 # Coefficient for KL divergence
    alpha = 0.1 # Coefficient for contrastive loss
    p = 0.1 # Bernoulli success probability

    # Initialize TensorBoard SummaryWriter.
    writer = SummaryWriter(log_dir="./runs/rb_vae_experiment")

    # Main training loop.
    for epoch in range(num_epochs):
        avg_total_loss, avg_recon_loss, avg_kl_loss, avg_contrast_loss = train_one_epoch(
            model, device, dataloader, optimizer, epoch, writer=writer,
            temperature=temperature, alpha_contrast=alpha, beta_kl=beta, bernoulli_p=p
        )
        print(f"Epoch {epoch+1} --- Total Loss: {avg_total_loss:.4f} | Recon: {avg_recon_loss:.4f} | "
              f"KL: {avg_kl_loss:.4f} | Contrastive: {avg_contrast_loss:.4f}")

    # Optionally, add the model graph (requires a sample input).
    sample_input = next(iter(dataloader)).to(device)
    writer.add_graph(model, sample_input[:, 0])

    # Save the model.
    save_path = Path(__file__).parent.joinpath("saved_RBVAE")
    torch.save(model.state_dict(), save_path)
    writer.close()