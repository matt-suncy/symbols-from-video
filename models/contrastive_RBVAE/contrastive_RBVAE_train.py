'''
Author: Matthew Sun
Description: Implementing training script for a simple RB-VAE, 
purpose is to figure out the details of the architecture

# NOTE: MARKED FOR REVIEW
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
    T.Resize((64, 64)),
    T.ToTensor()
])

### DATASETS ###

class StateSegmentDataset(Dataset):
    """
    --- Args ---
    frames_dir: str
        Path to directory with frame PNGs.
    state_segments: list of tuples 
        Each tuple is (start_idx, end_idx) for that state.
    transform: callable, optional 
        Optional transform to be applied on the frames.
    """
    def __init__(self, frames_dir, state_segments, transform=None):
        self.frames_dir = frames_dir
        self.state_segments = state_segments
        self.transform = transform

    def __len__(self):
        '''
        The dataset length is the number of STATES, not frames
        '''
        return len(self.state_segments)

    def __getitem__(self, idx):
        start_idx, end_idx = self.state_segments[idx]
        frame_count = end_idx - start_idx

        frames = []
        for frame_idx in range(start_idx, end_idx):
            filename = f"{frame_idx:010d}.jpg"
            path = os.path.join(self.frames_dir, filename)
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            # img: [C, H, W]
            frames.append(img)

        # Stack frames along time dimension
        # After stacking: frames_tensor.shape should be [FrameCount, C, H, W]
        frames_tensor = torch.stack(frames, dim=0)

        '''
        Suggestions from GPT-o1:
        - You might want a shape of [B, T, C, H, W] for the model input where B=1 here.
        - The model expects [B, T, C, H, W], so keep [T, C, H, W].
        - The DataLoader with batch_size > 1 will add the batch dimension.

        although, I think the data is already in the shape of [B, T, C, H, W]? 
        '''
        return frames_tensor


class StatePairDataset(Dataset):
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
        return pairs_tensor

    def _load_frame(self, frame_index):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(self.frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


### TRAINING LOOP ###

def train_one_epoch(model, device, dataloader, optimizer, temperature=0.5, bernoulli_p=0.5, margin=1.0, alpha_contrast=0.1, beta_kl=0.1):
    """
    --- Args ---
    model: NN Class
        Seq2SeqBinaryVAE (or similar) neural net class.
    device: PyTorch device
        A PyTorch object determining what device the data is on.
    dataloader: Dataloader Class
        Yields sequences from each state segment.
    optimizer: Optimizer Class
        Any PyTorch optimizer.
    temperature: float
        Determines the "smoothness" of the samples.
    margin: float
        Determines the threshold for dissimilarity.
    alpha_contrast: float
        Coefficient for weighting of contrastive loss.
    beta_kl: float
        Coefficient for weighting of KL Divergence loss.

    --- Returns ---
    total_loss: float
        Loss value divided by the number of states.
    recon_loss: float
        The average reconstruction loss value.
    kl_loss: float
        The average KL divergence loss value.
    contrast_loss: float
        The average contrastive loss value. 
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_contrast_loss = 0.0
    for item in dataloader:
        # item shape: [B, 2, T, C, H, W]
        item = item.to(device)
        # Separate the pairs of frames into tensors with shape [B, 1, T, C, H, W]
        frame_1 = item[:, 0]
        frame_2 = item[:, 1]

        # Forward pass
        x_recon_1, z_seq_1, logits_1 = model(frame_1, temperature=temperature, hard=False)
        x_recon_2, z_seq_2, logits_2 = model(frame_2, temperature=temperature, hard=False)

        # Reconstruction loss
        recon_loss_1 = recon_loss(x_recon_1, frame_1)
        recon_loss_2 = recon_loss(x_recon_2, frame_2)
        recon_loss_val = (recon_loss_1 + recon_loss_2) / 2.0

        # KL loss
        kl_loss_1 = kl_binary_concrete(logits_1, p=bernoulli_p)
        kl_loss_2 = kl_binary_concrete(logits_2, p=bernoulli_p)
        kl_loss_val = (kl_loss_1 + kl_loss_2) / 2.0

        # Contrastive loss
        contrast_loss_similar = contrast_loss(z_seq_1, z_seq_2, label=0)
        contrast_loss_dissim = 0
        # Calculate the dissimilar contrastive loss over the states,
        # just for one of the two sequences
        for state_index in range(int(item.shape[2]) - 1):
            # Get latent sequences of shape [B, latent_dim] 
            # Latent seq at state_index
            dissim_z_a = z_seq_1[:, state_index]
            # Latent seq at state_index + 1
            dissim_z_b = z_seq_1[:, state_index+1]
            contrast_loss_dissim += contrast_loss(dissim_z_a, dissim_z_b, label=1)
        contrast_loss_dissim = contrast_loss_dissim / float(int(item.shape[2]) - 1)
        contrast_loss_val = contrast_loss_similar + contrast_loss_dissim

        # Combine VAE losses and contrastive loss
        loss = recon_loss_val + beta_kl * kl_loss_val + alpha_contrast * contrast_loss_val
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss_val.item()
        total_kl_loss += kl_loss_val.item()
        total_contrast_loss += contrast_loss_val.item()

    return total_loss / len(dataloader), total_recon_loss / len(dataloader), \
        total_kl_loss / len(dataloader), total_contrast_loss / len(dataloader)


if __name__ == "__main__":

    frames_dir = Path(__file__).parent.parent.parent.joinpath("videos/frames/kid_playing_with_blocks_1.mp4")
    print(str(frames_dir))
    # NOTE: Arbitrary numbers right now
    state_segments = [
        (0, 40),   # State 0 covers frames [0..39]
        (50, 90), # State 1 covers frames [50..89]
        (100, 150) # State 2 covers frames [100..149]
        ]   

    dataset = StatePairDataset(frames_dir, state_segments, transform=ImageTransforms, num_items=500)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=16, hidden_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 1
        
    # temperature is for Softmax
    temperature = 0.5

    # beta is coefficient for KL
    beta = 0.1

    # alpha is coefficient for contrastive loss
    alpha = 0.1

    # p is the success prob parameter for the Bernoulli distribution
    p = 0.1

    # TODO: Add temperature annealing schedule for reparameterization
    # Categorical reparameterization related values
    num_global_iters = 0
    max_iters = num_epochs * len(dataset)

    # DataLoader yields video sequences x: [B, 2, T, C, H, W]
    for epoch in range(num_epochs):
        
        total_loss_val, recon_loss_val, kl_loss_val, contrast_loss_val = \
            train_one_epoch(model, device, dataloader, optimizer, temperature=temperature, 
            alpha_contrast=alpha, beta_kl=beta, bernoulli_p=p) 

        print(f"Epoch {epoch+1} --- Loss: {total_loss_val} Reconstruction: {recon_loss_val} \
            KL: {kl_loss_val} Contrastive: {contrast_loss_val}")