# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from contrastive_RBVAE_model import Seq2SeqBinaryVAE

def visualize():
    # Hyperparameters for dummy input
    batch_size = 1
    num_states = 4  # T states per sequence
    channels = 3
    height, width = 512, 512

    # Create a dummy input: shape [B, T, C, H, W]
    dummy_input = torch.randn(batch_size, num_states, channels, height, width)

    # Instantiate the model
    model = Seq2SeqBinaryVAE(in_channels=channels, out_channels=channels,
                             latent_dim=32, hidden_dim=32)
    # TODO: Incorporate loss into tensorboard visuals 
    model.eval()

    # Create a TensorBoard SummaryWriter (logs will go to ./runs)
    writer = SummaryWriter(log_dir="./runs/rbvae_visualization")

    # Add model graph. Note that add_graph requires the model to be on CPU or CUDA
    writer.add_graph(model, dummy_input)
    writer.flush()
    writer.close()

    print("Model graph has been written to './runs/rbvae_visualization'.")
    print("Run 'tensorboard --logdir=runs' to visualize the RBVAE architecture.")


if __name__ == "__main__":
    visualize()

