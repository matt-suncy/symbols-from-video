'''
Author: Matthew Sun
Description: Implementing a simple version of a Recurrent Binary Variational Autoencoder, 
purpose is to figure out the details of the architecture
'''

### IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
###

def binary_concrete_logits(logits, temperature=1.0, hard=False):
    """
    logits: [batch, latent_dim] 
       A single logit per latent variable representing p(z=1).
    temperature: float 
       Controls the 'smoothness' of the samples.
    hard: bool 
       If True, use straight-through estimation to produce hard samples.

    Returns:
       y: [batch, latent_dim] samples ~ Bernoulli(logits)
          Differentiable approximation using Gumbel-Softmax trick.
    """

    # Sample uniform noise
    U = torch.rand_like(logits)
    # Convert uniform noise to logistic noise
    noise = torch.log(U + 1e-20) - torch.log(1.0 - U + 1e-20)

    # Add noise to logits and scale by temperature
    y = torch.sigmoid((logits + noise) / temperature)

    if hard:
        # Straight-through estimator
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y

    return y

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_logits(logits, temperature=1.0, hard=False):
     """
    Params:
        logits: [batch, categorical_dim]
        temperature: controls smoothness of outputs
        hard: boolean to control whether outputs are discretized or not

    Returns: 
        y: differentiable samples approximating one-hot vectors.
    """
    gumbels = sample_gumbel(logits.shape).to(logits.device)
    y = logits + gumbels
    y = F.softmax(y / temperature, dim=-1)

    if hard:
        # Discretized outputs
        y_hard = (y == y.max(dim=-1, keepdim=True)[0]).float()
        y = (y_hard - y).detach() + y # detach() trick to not mess up gradients
    return y

# NOTE: These NN classes will be written for Gumbel Softmax 
# meaning two logits for each latent dimension hence shape of (n, latent_dim, 2)

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
        self.fc = nn.Linear(256*8*8, latent_dim*2)  # for binary each latent_dim => 2 logits

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        logits = self.fc(h) # [B, latent_dim*2]
        logits = logits.view(x.size(0), self.latent_dim, 2)
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
            nn.Sigmoid() # Map back to the range that pixel values will have: [0, 1]
        )

    def forward(self, z):
        # z: [B, latent_dim]
        h = self.fc(z)
        h = h.view(h.size(0), 256, 8, 8)
        x_recon = self.deconv(h)
        return x_recon

class EncoderRNN(nn.module):
    # Should latent_dim == hidden_dim? probably right, we just want to capture temporal dependencies
    def __init__(self, latent_dim=32, hidden_dim=32, num_layers=1):
        super.__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, z_seq):
        # z_seq: [B, T, latent_dim]
        h_seq, (h_n, c_n) = self.lstm(z_seq)
        # h_seq: [B, T, hidden_dim]
        return h_seq, (h_n, c_n)