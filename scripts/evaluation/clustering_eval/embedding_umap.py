import sys
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
from matplotlib import pyplot as plt
import umap  # UMAP for dimensionality reduction

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE

# This is a CALLABLE
RESOLUTION = 256
ImageTransforms = T.Compose([
    T.Resize((RESOLUTION, RESOLUTION)),
    T.ToTensor()
])

def load_frames(frames_dir, frame_indices: tuple, transform=None):
    """
    Loads frames into a list given the path to the frames and indices.
    """
    images = []
    for frame_index in range(frame_indices[0], frame_indices[1]):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if transform is not None:
            image = transform(image)
        images.append(image)
    return images

def assign_label(frame_index, flags):
    """
    Assigns a class label based on the provided flag indices.
    Frames before the first flag are label 0, between the first and second flag are label 1, etc.
    """
    label = 0
    for f in flags:
        if frame_index >= f:
            label += 1
        else:
            break
    return label

if __name__ == "__main__":
    # Load model
    model_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "models/contrastive_RBVAE/saved_RBVAE"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rbvae_model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    rbvae_model.load_state_dict(checkpoint['model_state_dict'])
    rbvae_model.to(device)
    rbvae_model.eval()
    # Remember: any input tensor must be sent to device before feeding into the model.

    # Set the frames directory and parameters
    frames_dir = Path(__file__).parent.parent.parent.parent.joinpath(
        "videos/frames/kid_playing_with_blocks_1.mp4"
    )
    last_frame = 1425
    # Frame indices that separate different states (classes)
    flags = [152, 315, 486, 607, 734, 871, 1153, 1343]

    print("Loading frames...")
    frames = load_frames(frames_dir, (0, last_frame), transform=ImageTransforms)

    latent_vectors = []
    labels = []

    # Loop through each frame and compute its latent embedding
    with torch.no_grad():
        for idx, frame in enumerate(frames):
            # Add a batch and time dimension and move the tensor to the proper device
            input_tensor = frame[None, None, :, :, :].to(device)
            # Get the latent representation.
            # This call assumes that the model has an "encode" method that returns the latent vector.
            latent = rbvae_model.encode(input_tensor, temperature=0.5, hard=True)  # Expected shape: [1, latent_dim]
            latent = latent.cpu().numpy().squeeze()    # Remove batch dimension and move to CPU
            latent_vectors.append(latent)
            # Assign a label based on the frame index using the flags
            label = assign_label(idx, flags)
            labels.append(label)

    latent_vectors = np.array(latent_vectors)

    # Use UMAP to reduce the latent space to 2 dimensions
    umap_model = umap.UMAP(n_neighbors=100, min_dist=0.1, metric='hamming', random_state=42)
    embedding_2d = umap_model.fit_transform(latent_vectors)

    # Visualize the UMAP projection
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap="Spectral", s=5)
    plt.title("UMAP Projection of RBVAE Latent Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label="Frame Class")
    plt.savefig(os.path.join(os.path.dirname(__file__), "RBVAE_UMAP"))
