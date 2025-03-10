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

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE
# from models.contrastive_RBVAE.contrastive_RBVAE_train import ShuffledStatePairDataset

# This is a CALLABLE
RESOLUTION = 256
ImageTransforms = T.Compose([
    T.Resize((RESOLUTION, RESOLUTION)),
    T.ToTensor()
])

# We'll need some functions for "robustness to noise and occlusion" tests

# Functon for adding gaussian noise to a tensor
def add_gaussian_noise(tensor, mean=0., std=0.1):
    """
    Adds gaussian noise to the input tensor.
    Args:
        tensor: Input image tensor of shape [C, H, W] or [B, C, H, W]
        mean: Mean of the gaussian noise
        std: Standard deviation of the gaussian noise
    Returns:
        Tensor with added gaussian noise, clipped to [0,1]
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
    noise = torch.randn_like(tensor) * std + mean
    noisy = tensor + noise
    
    # Clip values to [0,1] range since these are image tensors
    noisy = torch.clamp(noisy, 0, 1)
    
    return noisy.squeeze(0) if tensor.dim() == 3 else noisy


# Function for adding occlusion to a tensor, covers X% of the image with a grey square
def add_occlusion(tensor, coverage=0.2):
    """
    Adds a grey square occlusion to the input tensor covering X% of the image.
    Args:
        tensor: Input image tensor of shape [C, H, W] or [B, C, H, W]
        coverage: Float between 0 and 1 indicating what fraction of image to occlude
    Returns:
        Tensor with grey square occlusion
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    B, C, H, W = tensor.shape
    
    # Calculate square size based on coverage
    square_size = int(np.sqrt(coverage * H * W))
    
    # Random position for top-left corner of square
    x = random.randint(0, W - square_size)
    y = random.randint(0, H - square_size)
    
    # Create copy of tensor to modify
    occluded = tensor.clone()
    
    # Add grey square (value 0.5) to all channels
    occluded[:, :, y:y+square_size, x:x+square_size] = 0.5
    
    return occluded.squeeze(0) if tensor.dim() == 3 else occluded


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
    grey_out = 10
    state_segments = []
    for i in range(len(flags)):
        if i > 0:
            state_segments.append((flags[i-1] + grey_out, flags[i] - grey_out + 1))
        elif i == len(flags)-1:
            state_segments.append((flags[i] + grey_out, last_frame + 1))
        else:
            state_segments.append((0, flags[0] - grey_out + 1))

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

    # Find the most common latent state for each class
    for label in range(len(flags)):
        label_vectors = latent_vectors[np.array(labels) == label]
        
        # Get unique latent vectors and their counts
        unique_vectors, counts = np.unique(label_vectors, axis=0, return_counts=True)
        
        # Get the most common latent vector (the one with the highest count)
        most_common_vector = unique_vectors[np.argmax(counts)]
        print(f"Most common latent state for class {label}: {most_common_vector}")
        
        # Calculate the percentage of frames that match this latent state
        matches = np.all(label_vectors == most_common_vector, axis=1)
        percentage = np.sum(matches) / len(label_vectors)
        print(f"Percentage of frames that match the most common latent state for class {label}: {percentage}")

    # Find the percentage of frames that match the most common latent state for each class
    percentages = []
    for label in range(len(flags)):
        label_vectors = latent_vectors[np.array(labels) == label]
        percentage = np.sum(label_vectors == most_common_vector) / len(label_vectors)
        print(f"Percentage of frames that match the most common latent state for class {label}: {percentage}")
        percentages.append(percentage)

    # Calculate the weighted average of the matching percentages
    weighted_average = np.sum(percentages * np.array(flags)) / np.sum(flags)
    print(f"Weighted average of the matching percentages: {weighted_average}")

    # Plot the percentage of frames that match the most common latent state for each class
    plt.plot(range(len(flags)), percentages)
    plt.xlabel("Class")
    plt.ylabel("Percentage of frames that match the most common latent state")
    plt.show()

    # Save the plot
    plt.savefig("embedding_matching.png")

    # Find the number of unique latent states for each class
    for label in range(len(flags)):
        label_vectors = latent_vectors[np.array(labels) == label]
        unique_vectors = np.unique(label_vectors, axis=0)
        print(f"Number of unique latent states for class {label}: {len(unique_vectors)}")
    
