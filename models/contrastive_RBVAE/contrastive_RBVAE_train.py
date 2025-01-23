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
from simple_RBVAE_model import Seq2SeqBinaryVAE
###

#### LOSS FUNCTIONS

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
        self.state_frame_indices
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
                # Randomly choose 2 distinct states from the current state
                idx1, idx1 = random.sample(frame_indices, 2)

            # Load and apply transform
            frame1 = self._load_frame(index1)
            frame2 = self._load_frame(index2)

            pairs.append(torch.stack([frame1, frame2], dim=0)) # [2, C, H, W]

        # Stack along dimension 0 to get a shape of [T, 2, C, H, W]
        pairs_tensor = torch.stack(pairs, dim=0)

        pairs_tensor = pairs_tensor.permute(1, 0, 2, 3, 4) # [2, T, C, H, W]
        '''
        NOTE: Honestly no particular reason to have the pair dimension first.
        A shape of [2, T, C, H, W] just feels right for me.
        '''
        return pairs_tensor

    def _load_frame(self, frame_index):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(self.frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


        

### TRAINING LOOP 

def train_one_epoch(model, dataloader, optimizer, margin=1.0, alpha_contrast=0.1, beta_kl=0.1):
    """
    --- Args ---
    model: NN Class
        Seq2SeqBinaryVAE (or similar) neural net class.
    dataloader: Dataloader Class
        Yields sequences from each state segment.
    optimizer: Optimizer Class
        Any PyTorch optimizer.
    margin: float
        Determines the threshold for dissimilarity .
    alpha_contrast: float
        Coefficient for weighting of contrastive loss.
    beta_kl: float
        Coefficient for weighting of KL Divergence loss.

    --- Returns ---
    mean_loss: float
        total_loss divided by the number of states.
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_contrast_loss = 0.0
    for frames_seq in dataloader:
        # frames_seq: [B, T, C, H, W], often B=1 if each segment is an item
        frames_seq = frames_seq.cuda()

        # Forward pass
        # Suppose your model is modified to return z_seq as well:
        # x_recon: [B, T, C, H, W]
        # z_seq:   [B, T, latent_dim]
        # logits:  [B*T, latent_dim, 2] or [B, T, latent_dim] (depends on your approach)
        x_recon, z_seq, logits = model(frames_seq, return_latents=True)
        
        # Standard VAE losses
        recon_loss_val = reconstruction_loss(x_recon, frames_seq)
        kl_loss_val = kl_binary(logits)

        # ---------------------------------------------------------
        # 1) Sample pairs for contrastive loss
        #    Let's do a simple example with B=1 for clarity:
        #    frames_seq => shape = [1, T, C, H, W]
        #    z_seq => shape = [1, T, latent_dim]
        # We will pick:
        #   - two frames i,j from the same state => positive pair
        #   - two frames k,l from a different state => negative pair
        # For demonstration, let's assume we have T frames all from the SAME state
        # and T frames from the NEXT state in the next iteration or some other approach.

        # If your dataset item has multiple states, you can separate them by index.
        # Or if each dataset item is exactly one state, you can store it with the
        # next state in the same batch. Many ways to do it.

        # Let's suppose we have T frames from the SAME state. We'll do "positive pairs"
        # from this state. We'll also pretend we have T frames from the NEXT state
        # in the same batch => shape [2, T, ...]. Then we can do negative pairs
        # across the batch dimension. This is just an example.

        if frames_seq.size(0) == 2:
            # B=2 => we have two states (or two segments), each with T frames
            z_seq_0 = z_seq[0]  # shape [T, latent_dim]
            z_seq_1 = z_seq[1]  # shape [T, latent_dim]

            # Positive pair: pick random frames i, j from the same state (batch 0)
            T0 = z_seq_0.size(0)
            i, j = np.random.choice(T0, size=2, replace=False)
            z_i = z_seq_0[i].unsqueeze(0)  # [1, latent_dim]
            z_j = z_seq_0[j].unsqueeze(0)  # [1, latent_dim]

            # Negative pair: pick one frame from batch 0, one from batch 1
            k = np.random.randint(0, T0)
            T1 = z_seq_1.size(0)
            l = np.random.randint(0, T1)
            z_k = z_seq_0[k].unsqueeze(0)  # [1, latent_dim]
            z_l = z_seq_1[l].unsqueeze(0)  # [1, latent_dim]

            # Build label Tensors
            y_pos = torch.ones(1, device=z_i.device)  # same state
            y_neg = torch.zeros(1, device=z_i.device) # different state

            # Compute contrastive losses
            loss_pos = contrastive_loss(z_i, z_j, y_pos, margin=margin)
            loss_neg = contrastive_loss(z_k, z_l, y_neg, margin=margin)
            contrast_loss_val = loss_pos + loss_neg

        else:
            # If B=1 or something else, you might skip or do another strategy
            contrast_loss_val = 0.0

        # 2) Combine VAE losses + contrastive loss
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