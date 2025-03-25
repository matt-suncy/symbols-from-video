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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE
from models.percep_RBVAE.percep_RBVAE_model import Seq2SeqBinaryVAE as PercepBinaryVAE
from models.percep_RBVAE.percep_RBVAE_train import ShuffledStatePairDataset

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

def create_umap_visualization(latent_vectors, labels, title, output_path):
    """
    Creates and saves a UMAP visualization.
    """
    # Use UMAP to reduce the latent space to 2 dimensions
    umap_model = umap.UMAP(n_neighbors=100, min_dist=0.1, metric='hamming', random_state=42)
    embedding_2d = umap_model.fit_transform(latent_vectors)

    # Visualize the UMAP projection
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   label=f'Class {int(label)}', 
                   s=50,  # Increased dot size
                   alpha=0.7)
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Frame Class")
    plt.savefig(output_path)
    plt.close()

def create_tsne_visualization(latent_vectors, labels, title, output_path):
    """
    Creates and saves a t-SNE visualization.
    """
    # Use t-SNE to reduce the latent space to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding_2d = tsne.fit_transform(latent_vectors)

    # Visualize the t-SNE projection
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   label=f'Class {int(label)}', 
                   s=50,  # Increased dot size
                   alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Frame Class")
    plt.savefig(output_path)
    plt.close()

def create_pca_visualization(latent_vectors, labels, title, output_path):
    """
    Creates and saves a PCA visualization with 2 principal components.
    """
    # Use PCA to reduce the latent space to 2 dimensions
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(latent_vectors)

    # Visualize the PCA projection
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   label=f'Class {int(label)}', 
                   s=50,  # Increased dot size
                   alpha=0.7)
    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend(title="Frame Class")
    plt.savefig(output_path)
    plt.close()

def get_test_indices(state_segments, test_pct=0.1, val_pct=0.1):
    """
    Gets test set indices for each state using the same logic as ShuffledStatePairDataset.
    """
    test_indices = []
    for start, end in state_segments:
        full_indices = list(range(start, end))
        n = len(full_indices)
        test_val_count = int(n * (test_pct + val_pct))
        margin = (n - test_val_count) // 2
        test_val_indices = full_indices[margin : margin + test_val_count]
        
        if test_val_count > 0:
            test_fraction_of_middle = test_pct / (test_pct + val_pct)
            test_count = int(round(test_fraction_of_middle * test_val_count))
            test_indices.extend(test_val_indices[:test_count])
    
    return sorted(test_indices)

if __name__ == "__main__":
    # Set up paths and parameters
    frames_dir = Path(__file__).parent.parent.parent.parent.joinpath(
        "videos/frames/kid_playing_with_blocks"
    )
    last_frame = 1425
    flags = [152, 315, 486, 607, 734, 871, 1153, 1343]
    grey_out = 10
    
    # Create state segments
    state_segments = []
    for i in range(len(flags)):
        if i > 0:
            state_segments.append((flags[i-1] + grey_out + 1, flags[i] - grey_out))
        else:
            state_segments.append((0, flags[0] - grey_out))
    state_segments.append((flags[-1] + grey_out + 1, last_frame + 1))

    # Get test set indices
    test_indices = get_test_indices(state_segments)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process for contrastive RBVAE
    contrastive_model_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "scripts/evaluation/best_models/pixels/best_model_breezy-sweep-38.pt"
    )
    contrastive_latent_dim = 25
    contrastive_model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, 
        latent_dim=contrastive_latent_dim, hidden_dim=contrastive_latent_dim)
    checkpoint = torch.load(contrastive_model_path, map_location=torch.device('cpu'))
    contrastive_model.load_state_dict(checkpoint['model_state_dict'])
    contrastive_model.to(device)
    contrastive_model.eval()

    # Load and process for perceptual RBVAE
    percep_model_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "scripts/evaluation/best_models/perceps/best_model_grateful-sweep-19.pt"
    )
    percep_latent_dim = 25
    percep_model = PercepBinaryVAE(in_channels=4, out_channels=4, 
        latent_dim=percep_latent_dim, hidden_dim=percep_latent_dim)
    checkpoint = torch.load(percep_model_path, map_location=torch.device('cpu'))
    percep_model.load_state_dict(checkpoint['model_state_dict'])
    percep_model.to(device)
    percep_model.eval()

    # Load perceptual embeddings
    percep_embeddings_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "videos/kid_playing_with_blocks_perceps.npy"
    )
    percep_embeddings = np.load(percep_embeddings_path, allow_pickle=True).item()

    # Process test frames for both models
    contrastive_latent_vectors = []
    percep_latent_vectors = []
    labels = []

    print("Processing test frames...")
    with torch.no_grad():
        for idx in test_indices:
            # For contrastive model
            frame = ImageTransforms(Image.open(os.path.join(frames_dir, f"{idx:010d}.jpg")).convert("RGB"))
            input_tensor = frame[None, None, :, :, :].to(device)
            latent = contrastive_model.encode(input_tensor, temperature=0.2, hard=False)
            contrastive_latent_vectors.append(latent.cpu().numpy().squeeze())

            # For perceptual model
            key = f"{idx:010d}.jpg"
            percep_embedding = percep_embeddings.get(key)
            if percep_embedding is None:
                key = f"{idx:010d}"
                percep_embedding = percep_embeddings.get(key)
            percep_tensor = torch.tensor(percep_embedding, dtype=torch.float32)[None, :, :, :].to(device)
            latent = percep_model.encode(percep_tensor, temperature=0.2, hard=False)
            percep_latent_vectors.append(latent.cpu().numpy().squeeze())

            # Label only needs to be added once since it's the same for both
            labels.append(assign_label(idx, flags))

    contrastive_latent_vectors = np.array(contrastive_latent_vectors)
    percep_latent_vectors = np.array(percep_latent_vectors)
    labels = np.array(labels)

    # Create visualizations for all dimensionality reduction methods
    print("Creating dimensionality reduction visualizations...")
    
    # UMAP visualizations
    create_umap_visualization(
        contrastive_latent_vectors, 
        labels, 
        "UMAP Projection of Contrastive RBVAE Latent Embeddings (Test Set)",
        os.path.join(os.path.dirname(__file__), "Contrastive_RBVAE_UMAP")
    )
    create_umap_visualization(
        percep_latent_vectors, 
        labels, 
        "UMAP Projection of Perceptual RBVAE Latent Embeddings (Test Set)",
        os.path.join(os.path.dirname(__file__), "Perceptual_RBVAE_UMAP")
    )
    
    # t-SNE visualizations
    create_tsne_visualization(
        contrastive_latent_vectors, 
        labels, 
        "t-SNE Projection of Contrastive RBVAE Latent Embeddings (Test Set)",
        os.path.join(os.path.dirname(__file__), "Contrastive_RBVAE_TSNE")
    )
    create_tsne_visualization(
        percep_latent_vectors, 
        labels, 
        "t-SNE Projection of Perceptual RBVAE Latent Embeddings (Test Set)",
        os.path.join(os.path.dirname(__file__), "Perceptual_RBVAE_TSNE")
    )
    
    # PCA visualizations
    create_pca_visualization(
        contrastive_latent_vectors, 
        labels, 
        "PCA Projection of Contrastive RBVAE Latent Embeddings (Test Set)",
        os.path.join(os.path.dirname(__file__), "Contrastive_RBVAE_PCA")
    )
    create_pca_visualization(
        percep_latent_vectors, 
        labels, 
        "PCA Projection of Perceptual RBVAE Latent Embeddings (Test Set)",
        os.path.join(os.path.dirname(__file__), "Perceptual_RBVAE_PCA")
    )
