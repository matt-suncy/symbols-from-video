
### IMPORTS
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
###

#### LOSS FUNCTIONS

def recon_loss(x_recon, x):
    return F.mse_loss(x_recon, x)

def kl_binary(q_logits):
    q = F.softmax(q_logits, dim=-1) # [B*T, latent_dim, 2]
    # KL(q||p) with p=uniform(0.5,0.5):
    # NOTE: This will be changed to a Bernoulli distribution with a small value later
    kl = q * (torch.log(q + 1e-20) - np.log(0.5))
    kl = kl.sum(-1).sum(-1) # Sum over categories and latent dimensions 
    return kl.mean()

### DATA PREP   
# NOTE: Arbitrary numbers right now
STATE_SEGMENTS = [
    (0, 40),   # State 0 covers frames [0..39]
    (50, 90), # State 1 covers frames [50..89]
    (100, 150) # State 2 covers frames [100..149]
]   

# This is a class
image_transforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

class StateSegmentDataset(Dataset):
    def __init__(self, frames_dir, state_segments, transform=None):
        """
        Args:
            frames_dir (str): Path to directory with frame PNGs.
            state_segments (list of tuples): Each tuple is (start_idx, end_idx) for that state.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
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
            filename = f"frame_{frame_idx:05d}.png"
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
        '''
        return frames_tensor

### TRAINING LOOP

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    temperature = 1.0

    # Suppose you have a DataLoader yielding video sequences x: [B, T, C, H, W]
    for epoch in range(10):
        for x in dataloader:
            x = x.to(device)
            x_recon, logits = model(x, temperature=0.5, hard=False)
            recon_loss_val = reconstruction_loss(x_recon, x)
            kl_loss_val = kl_binary(logits)
            loss = recon_loss_val + 0.1 * kl_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} - Loss: {loss.item()} Reconstruction: {recon_loss_val.item()} KL: {kl_loss_val.item()}")