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

def l1_loss(q_logits, lamb):
    # This should only be used if there is only 1 logit per latent variable
    return lamb * torch.norm(q_logits, p=1)


def recon_loss(x_recon, x):
    return F.mse_loss(x_recon, x)


def kl_binary_concrete(q_logits, p=0.5, eps=1e-10):
    '''
    Calculates the KL Divergence between input logits and a 
    Bernoulli distribution.
    '''
    q = torch.sigmoid(q_logits)
    kl = q * (torch.log(q + eps) - np.log(p)) + (1.0 - q) * (torch.log(1.0 - q + eps) - np.log(1.0 - p))
    kl = kl.sum(dim=-1)   # Sum over latent dimensions
    kl_mean = kl.mean()   # Average over the batch
    return kl_mean


def contrast_loss(x1, x2, label, margin: float=1.0):
    """
    Computes Contrastive Loss. Requires an input label to determine difference
    between classes (0 for similar, 1 for dissimilar).
    """
    dist = F.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss

# Callable image transform
ImageTransforms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

### DATASETS ###

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


class ShuffledStatePairDataset(Dataset):
    '''
    Args:
        frames_dir: Path to the directory containing the frames.
        state_segments: List of tuples (start_idx, end_idx) determining state groupings.
        transform: Transform to be applied on each frame.
    '''
    def __init__(self, frames_dir, state_segments, transform=None):
        self.frames_dir = frames_dir
        self.state_segments = state_segments
        self.transform = transform
        self.num_states = len(self.state_segments)
        self.state_frame_indices = []
        self.max_frames = 0
        for (start, end) in self.state_segments:
            indices = list(range(start, end))
            self.state_frame_indices.append(indices)
            self.max_frames = max(self.max_frames, len(indices))
        self.state_pairs = []
        for indices in self.state_frame_indices:
            if len(indices) < self.max_frames:
                padded = indices.copy() + random.choices(indices, k=self.max_frames - len(indices))
            else:
                padded = indices.copy()
            random.shuffle(padded)
            pairs = []
            for i in range(len(padded) // 2):
                pairs.append((padded[2 * i], padded[2 * i + 1]))
            if len(padded) % 2 == 1:
                leftover = padded[-1]
                candidate = random.choice([x for x in indices if x != leftover]) if len(indices) > 1 else leftover
                pairs.append((leftover, candidate))
            self.state_pairs.append(pairs)
        self.num_items = self.max_frames // 2 if (self.max_frames % 2 == 0) else (self.max_frames + 1) // 2

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        pairs = []
        for state_idx, pairs_list in enumerate(self.state_pairs):
            pair_idx = idx % len(pairs_list)
            pair = pairs_list[pair_idx]
            frame1 = self._load_frame(pair[0])
            frame2 = self._load_frame(pair[1])
            pairs.append(torch.stack([frame1, frame2], dim=0))  # [2, C, H, W]
        pairs_tensor = torch.stack(pairs, dim=0).permute(1, 0, 2, 3, 4)  # [2, T, C, H, W]
        return pairs_tensor

    def _load_frame(self, frame_index):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(self.frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

### TRAINING LOOP ###

def train_one_epoch(model, device, dataloader, optimizer, batch_size, epoch, writer=None, 
                    init_temperature=1.0, final_temperature=0.5, anneal_rate=1e-4, num_steps_to_update=100,
                    bernoulli_p=0.5, margin=1.0, alpha_contrast=0.1, beta_kl=0.1):
    """
    Trains the model for one epoch and logs losses to TensorBoard.
    Args:
        model: The RB-VAE model.
        device: PyTorch device.
        dataloader: DataLoader yielding batches of shape [B, 2, T, C, H, W].
        optimizer: Optimizer for updating model parameters.
        writer: Optional TensorBoard SummaryWriter instance.
        epoch: Current epoch number (for logging).
        init_temperature: Initial temperature for Gumbel-Softmax.
        final_temperature: Minimum temperature to anneal to.
        anneal_rate: Exponential decay rate for temperature.
        num_steps_to_update: Number of steps between temperature updates.
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
    temperature = init_temperature  # start with the initial temperature
    for batch_idx, item in enumerate(dataloader):
        # item shape: [B, 2, T, C, H, W]
        global_step = epoch * num_batches + batch_idx + 1

        # Update the temperature every num_steps_to_update steps.
        if global_step % num_steps_to_update == 0:
            temperature = max(final_temperature, init_temperature * np.exp(-anneal_rate * global_step))
            if writer is not None:
                writer.add_scalar('Batch/Temperature', temperature, global_step)

        num_batches_item, _, num_states, _, _, _ = item.size()
        item = item.to(device)
        frames = [item[:, i] for i in range(2)]
        recon_losses = []
        kl_losses = []
        h_seqs = []

        for frame in frames:
            x_recon, h_seq, bc_seq = model(frame, temperature=temperature, hard=False)
            recon_losses.append(recon_loss(x_recon, frame))
            kl_losses.append(kl_binary_concrete(bc_seq, p=bernoulli_p))
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
        total_loss_val = recon_loss_val + beta_kl * kl_loss_val + alpha_contrast * contrast_loss_val
        
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        total_loss += total_loss_val.item()
        total_recon_loss += recon_loss_val.item()
        total_kl_loss += kl_loss_val.item()
        total_contrast_loss += contrast_loss_val.item()

        if writer is not None:
            writer.add_scalar('Batch/Total_Loss', total_loss_val.item(), global_step)
            writer.add_scalar('Batch/Reconstruction_Loss', recon_loss_val.item(), global_step)
            writer.add_scalar('Batch/KL_Divergence', kl_loss_val.item(), global_step)
            writer.add_scalar('Batch/Contrastive_Loss', contrast_loss_val.item(), global_step)

    avg_total_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_contrast_loss = total_contrast_loss / num_batches

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
    
    batch_size = 4
    dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    max_iters = num_epochs * len(dataloader)
    num_temp_updates = 25
    num_steps_to_update = int(max_iters / num_temp_updates)

    init_temperature = 1.0      # Initial temperature for Gumbel-Softmax
    final_temperature = 0.5     # Minimum temperature after annealing
    anneal_rate = 1e-4          # Annealing rate
    beta = 0.1                # Coefficient for KL divergence
    alpha = 0.1               # Coefficient for contrastive loss
    p = 0.1                   # Bernoulli success probability

    writer = SummaryWriter(log_dir="./runs/rb_vae_experiment")

    for epoch in range(num_epochs):
        avg_total_loss, avg_recon_loss, avg_kl_loss, avg_contrast_loss = train_one_epoch(
            model, device, dataloader, optimizer, batch_size, epoch, writer=writer,
            init_temperature=init_temperature, final_temperature=final_temperature, anneal_rate=anneal_rate,
            num_steps_to_update=num_steps_to_update, alpha_contrast=alpha, beta_kl=beta, bernoulli_p=p
        )
        print(f"Epoch {epoch+1} --- Total Loss: {avg_total_loss:.4f} | Recon: {avg_recon_loss:.4f} | "
              f"KL: {avg_kl_loss:.4f} | Contrastive: {avg_contrast_loss:.4f}")

    model.eval()
    sample_input = next(iter(dataloader)).to(device)
    writer.add_graph(model, sample_input[:, 0])

    save_path = Path(__file__).parent.joinpath("saved_RBVAE")
    torch.save(model.state_dict(), save_path)
    writer.close()