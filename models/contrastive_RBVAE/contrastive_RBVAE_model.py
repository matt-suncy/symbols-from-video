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

def binary_concrete_logits(logits, temperature=0.5, hard=False, eps=1e-8, noise_ratio=0.1):
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
    noise = noise_ratio * (torch.log(U + eps) - torch.log(1.0 - U + eps))

    # Add noise to logits and scale by temperature
    y = torch.sigmoid((logits + noise) / temperature)

    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y # detach() trick to not mess up gradients

    return y

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),            
            nn.ReLU(),          
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),           
            nn.Flatten()
        )
        # 1 logit per latent variable since we want num_categories = 2
        self.fc = nn.Linear(64 * 8 * 8 * (4**2), latent_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        logits = self.fc(h) # [B, latent_dim]
        logits = logits.reshape(x.size(0), self.latent_dim)
        return logits


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8 * (4**2))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, out_channels, 3, 2, 1, output_padding=1),
            nn.Sigmoid() # Map back to the range that pixel values will have: [0, 1]
        )

    def forward(self, z):
        # z: [B, latent_dim]
        h = self.fc(z)
        h = h.reshape(h.size(0), 64, 8*4, 8*4)
        x_recon = self.deconv(h)    
        return x_recon


class EncoderRNN(nn.Module):
    # Should latent_dim == hidden_dim? 
    # Probably right? We just want to capture temporal dependencies so 
    # what's the point of making the hidden dim different
    def __init__(self, latent_dim=32, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)


    def forward(self, z_seq):
        # z_seq: [B, T, latent_dim]
        h_seq, (h_n, c_n) = self.lstm(z_seq)
        # h_seq: [B, T, hidden_dim]
        return h_seq, (h_n, c_n)    


class DecoderRNN(nn.Module):
    def __init__(self,  hidden_dim=32, latent_dim=32, num_layers=2):
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
    '''
    --- Returns ---
    x_recon: tensor
        Reconstructed inputs
    z_seq: tensor
        Hidden sequence of latent variables
    logits: tensor
        Logits of convolutional encoders before Binary Concrete
    '''
    def __init__(self, in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_cnn = ConvEncoder(in_channels, latent_dim)
        self.decoder_cnn = ConvDecoder(out_channels, latent_dim)
        self.encoder_rnn = EncoderRNN(latent_dim=latent_dim, hidden_dim=latent_dim)
        self.decoder_rnn = DecoderRNN(hidden_dim=latent_dim, latent_dim=latent_dim)

    def forward(self, x, temperature=1.0, hard=False, noise_ratio=0.1):
        # The pair dimension isn't here since we're feeding one by one
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()

        # Feed through conv encoder 
        # don't want time dimension to have convolution applied to it so treat it as a batch
        x_reshaped = x.reshape(B*T, C, H, W)
        x_conv = self.encoder_cnn(x_reshaped) # [B*T, latent_dim]
        x_conv= x_conv.reshape(B, T, self.latent_dim)

        # Feed through encoder RNN
        h_seq, _ = self.encoder_rnn(x_conv) # [B, T, hidden_dim]

        h_seq_reshaped = h_seq.reshape(B*T, self.latent_dim) # [B*T, latent_dim]

        z = binary_concrete_logits(h_seq_reshaped, temperature=temperature, hard=hard, noise_ratio=noise_ratio)
        z_seq = z.reshape(B, T, self.latent_dim) # Reshape z => [B, T, latent_dim]

        # Feed through decoder RNN
        d_seq, _ = self.decoder_rnn(z_seq) # [B, T, latent_dim]

        # Decode each d_t for reconstruction
        d_seq_flat = d_seq.reshape(B*T, self.latent_dim)
        x_recon = self.decoder_cnn(d_seq_flat)
        x_recon = x_recon.reshape(B, T, C, H, W)

        return x_recon, h_seq, z_seq # All of these are needed for backprop

    def encode(self, x, temperature=0.5, hard=False, noise_ratio=0.1):
        '''
        Returns the latent variables with Binary Concrete applied.
        '''
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()

        x_reshaped = x.reshape(B*T, C, H, W)
        x_conv = self.encoder_cnn(x_reshaped) # [B*T, latent_dim]
        x_conv= x_conv.reshape(B, T, self.latent_dim)

        # Feed through encoder RNN
        h_seq, _ = self.encoder_rnn(x_conv) # [B, T, hidden_dim]

        h_seq_reshaped = h_seq.reshape(B*T, self.latent_dim) # [B*T, latent_dim]

        z = binary_concrete_logits(h_seq_reshaped, temperature=temperature, hard=hard, noise_ratio=noise_ratio)
        z_seq = z.reshape(B, T, self.latent_dim) # Reshape z => [B, T, latent_dim]

        return z_seq

