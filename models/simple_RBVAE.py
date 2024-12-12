import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U, eps), eps)

def gumbel_softmax_logits(logits, temperature=1.0, hard=False):
    """
    logits: [batch, latent_dim] 
       A single logit per latent variable representing p(z=1).
    temperature: float 
       Controls the 'smoothness' of the samples.
    hard: bool 
       If True, outputs discretized samples

    Returns:
       y: [batch, latent_dim] samples ~ Bernoulli(logits)
          Differentiable approximation using Gumbel-Softmax trick.
    """

    # Create uniform logistic noise
    U = torch.rand_like(logits)
    noise = torch.log(U + 1e-20) - torch.log(1- U + 1e-20)

    # Add noise
    y = (logits + noise) / temperature
    y = torch.softmax(y, dim=-1)

    if hard:
        # Discretized outputs
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y # detach() to not mess up gradients

    return y

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(256*8*8, latent_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        logits = self.fc(h) # [B, latent_dim*2]
        logits = logits.view(x.size(0), self.latent_dim)
        return logits

class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Sigmoid() # Map back to the range that pixel values will have [0, 1]
        )

    def forward(self, z):
        # z: [B, latent_dim]
        h = self.fc(z)
        h = h.view(h.size(0), 256, 8, 8)
        x_recon = self.deconv(h)
        return x_recon
