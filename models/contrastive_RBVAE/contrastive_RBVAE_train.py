'''
Author: Matthew Sun
Description: Implementing training script for a simple RB-VAE, 
purpose is to figure out the details of the architecture
'''

### IMPORTS
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from simple_RBVAE_model import Seq2SeqBinaryVAE
###

#### LOSS FUNCTIONS

# TODO: Try L1 reg  

def l1_loss(q_logits, lamb):
    # This should only be used if there is only 1 logit per latent variable
    # Also I have no idea if q_logits needs to be reshaped or something
    return lamb * torch.norm(q_logits, 1)

def recon_loss(x_recon, x):
    return F.mse_loss(x_recon, x)

def kl_binary_gumbel(q_logits, eps=1e-10):
    # This is checking if there's more than 1 logit per latent var
    if q_logits.ndim > 2:
        # If there's more than 1 than we need to softmax over them
        q = F.softmax(q_logits, dim=-1) # [B*T, latent_dim, 3] for example
    else:
        q = q_logits

    # KL(q||p) with p=uniform(0.5,0.5):
    kl = q * (torch.log(q + eps) - np.log(0.5))
    kl = kl.sum(-1).sum(-1) # Sum over categories and latent dimensions 
    return kl.mean()

def kl_binary_concrete(q_logits, p=0.5, eps=1e-10):
    '''
    Calculates the KL Divergence between input logits and a 
    Bernoulli distribution
    '''
    # Squish it
    q = torch.sigmoid(q_logits)

    # KL( Bernoulli(q) || Bernoulli(p) ) = q * log(q/p) + (1-q) * log((1-q)/(1-p))
    # Not adding eps for p because it should never be that small
    # q: tensor, p: float
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


def contrast_loss(x1, x2, label, margin: float = 1.0):
    """
    Computes Contrastive Loss. Requires an input label to determine difference
    between classes.
    """

    dist = F.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss

# This is a CLASS
ImageTransforms = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

class StateSegmentDataset(Dataset):
    """
    Args:
    frames_dir: str
        Path to directory with frame PNGs.
    state_segments: list of tuples 
        Each tuple is (start_idx, end_idx) for that state.
    transform: callable, optional 
        Optional transform to be applied on a frame.
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
            # TODO: Change filename accordingly
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
        TODO
        Suggestions from GPT-o1:
        - You might want a shape of [B, T, C, H, W] for the model input where B=1 here.
        - The model expects [B, T, C, H, W], so keep [T, C, H, W].
        - The DataLoader with batch_size > 1 will add the batch dimension.

        although, I think the data is already in the shape of [B, T, C, H, W]? 
        '''
        return frames_tensor

### TRAINING LOOP
if __name__ == "__main__":

    frames_dir = Path(__file__).parent.parent.joinpath("videos/frames/kid_playing_with_blocks_1.mp4")
    print(str(frames_dir))
    # NOTE: Arbitrary numbers right now
    state_segments = [
        (0, 40),   # State 0 covers frames [0..39]
        (50, 90), # State 1 covers frames [50..89]
        (100, 150) # State 2 covers frames [100..149]
        ]   

    dataset = StateSegmentDataset(frames_dir, state_segments, transform=ImageTransforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=16, hidden_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reparam related values
    num_global_iters = 0
        
    # Reminder: temperature is for Softmax
    temperature = 1.0

    # Reminder: beta is coefficient for KL
    beta = 0.1
    num_epochs = 15

    # DataLoader yields video sequences x: [B, T, C, H, W]
    for epoch in range(num_epochs):
        for x in dataloader:
            # [B, T, C, H, W]
            x = x.to(device)

            x_recon, logits = model(x, temperature=0.5, hard=False)

            recon_loss_val = recon_loss(x_recon, x)
            kl_loss_val = kl_binary_concrete(logits, p=0.1)
            
            loss = recon_loss_val + beta * kl_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} - Loss: {loss.item()} Reconstruction: {recon_loss_val.item()} KL: {kl_loss_val.item()}")