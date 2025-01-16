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

# NOTE: I think eventually, BC is what we wanna use.
def binary_concrete_logits(logits, temperature=1.0, hard=False, eps=1e-10):
    """
    Args:
    logits: [batch, latent_dim] 
        A single logit per latent variable representing p(z=1).
    temperature: float 
        Controls the 'smoothness' of the samples.
    hard: bool 
        Controls whether samples are discretized or not

    Returns:
    y: [batch, latent_dim]
            Differentiable approximation using Gumbel-Softmax trick.
    """

    # Sample uniform noise
    U = torch.rand(logits.shape).to(logits.device)
    # Convert uniform noise to logistic noise
    noise = torch.log(U + eps) - torch.log(1.0 - U + eps)

    # Add noise to logits and scale by temperature
    y = torch.sigmoid((logits + noise) / temperature)

    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y

    return y

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_logits(logits, temperature=1.0, hard=False):
    """
    Params:
        logits: [batch, categorical_dim]
        temperature: float
            Controls smoothness of outputs
        hard: Boolean
            Controls whether outputs are discretized or not

    Returns: 
        y: [batch, latent_dim, 2]
           Differentiable samples approximating one-hot vectors.
    """
    gumbels = sample_gumbel(logits.shape).to(logits.device)
    y = logits + gumbels
    # TODO: Check during example training
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
            nn.Conv2d(in_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 1 logit per latent variable since we want num_categories = 2
        self.fc = nn.Linear(256*8*8, latent_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        logits = self.fc(h) # [B, latent_dim]
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
            nn.Sigmoid() # Map back to the range that pixel values will have: [0, 1]
        )

    def forward(self, z):
        # z: [B, latent_dim]
        h = self.fc(z)
        h = h.view(h.size(0), 256, 8, 8)
        x_recon = self.deconv(h)
        return x_recon


class EncoderRNN(nn.Module):
    # Should latent_dim == hidden_dim? 
    # Probably right? We just want to capture temporal dependencies so what's the point of expanding to more dimensions
    def __init__(self, latent_dim=32, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, z_seq):
        # z_seq: [B, T, latent_dim]
        h_seq, (h_n, c_n) = self.lstm(z_seq)
        # h_seq: [B, T, hidden_dim]
        return h_seq, (h_n, c_n)    


class DecoderRNN(nn.Module):
    def __init__(self,  hidden_dim=32, latent_dim=32, num_layers=1):
        super().__init__()
        # Decoder RNN maps from encoder hidden states to decoder states
        # We'll feed the encoder h_seq as inputs directly (like teacher forcing)
        self.lstm = nn.LSTM(hidden_dim, latent_dim, num_layers=num_layers, batch_first=True)
    
    def forward(self, h_seq):
        # h_seq: [B, T, hidden_dim] from the encoder
        # Decode by feeding h_seq into the decoder LSTM
        d_seq, (d_n, c_n) = self.lstm(h_seq)
        # d_seq: [B, T, latent_dim]
        return d_seq, (d_n, c_n)


class Seq2SeqBinaryVAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_cnn = ConvEncoder(in_channels, latent_dim)
        self.decoder_cnn = ConvDecoder(out_channels, latent_dim)
        self.encoder_rnn = EncoderRNN(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder_rnn = DecoderRNN(hidden_dim=hidden_dim, latent_dim=latent_dim)

    def forward(self, x, temperature=1.0, hard=False):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()

        # Feed through conv encoder
        x_reshaped = x.view(B*T, C, H, W)
        logits = self.encoder_cnn(x_reshaped) # [B*T, latent_dim, 2]
        
        '''
        This is all Gumbel Softmax with 2 logits per latent variable:

        # Sample binary latent z (discretized form)
        z_sample = gumbel_softmax_logits(logits, temperature=temperature, hard=hard)
        # # Extract probability of class '1': [B*T, latent_dim]
        z = z_sample[..., 1]
        '''

        z = binary_concrete_logits(logits, temperature=temperature, hard=hard)

        # Reshape z => [B, T, latent_dim]
        z_seq = z.view(B, T, self.latent_dim)   

        # Feed through encoder RNN
        h_seq, _ = self.encoder_rnn(z_seq) # [B, T, hidden_dim]

        # Feed through decoder RNN
        d_seq, _ = self.decoder_rnn(h_seq) # [B, T, latent_dim]

        # Decode each d_t for reconstruction
        d_seq_flat = d_seq.view(B*T, self.latent_dim)
        x_recon = self.decoder_cnn(d_seq_flat)
        x_recon = x_recon.view(B, T, C, H, W)

        return x_recon, logits

