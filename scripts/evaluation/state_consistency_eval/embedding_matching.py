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
import pandas as pd
from omegaconf import OmegaConf
import multiprocessing as mp

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

# Add stable diffusion to path
stable_diffusion_path = os.path.join(project_root, "src/stable-diffusion")
sys.path.insert(0, stable_diffusion_path)

from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE as ContrastiveRBVAE
from models.percep_RBVAE.percep_RBVAE_model import Seq2SeqBinaryVAE as PercepRBVAE
from models.contrastive_RBVAE.contrastive_RBVAE_train import ShuffledStatePairDataset

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

def calculate_state_consistency(model, test_dataset, device, temperature=0.5, noise_ratio=0.1, perturbation=None, perturbation_params=None):
    """
    Calculates state consistency for a model with optional perturbation.
    Args:
        model: The RBVAE model to evaluate
        test_dataset: Dataset containing test frames
        device: Device to run model on
        temperature: Temperature for binary concrete
        noise_ratio: Noise ratio for binary concrete
        perturbation: Optional perturbation function (add_gaussian_noise or add_occlusion)
        perturbation_params: Parameters for the perturbation function
    Returns:
        Weighted average consistency and individual state consistencies
    """
    model.eval()
    latent_vectors = []
    labels = []
    
    # Get all test indices
    test_indices = []
    for indices in test_dataset.test_indices_per_state:
        test_indices.extend(indices)
    
    with torch.no_grad():
        for idx in test_indices:
            # Load embedding for the frame
            embedding = test_dataset._load_embedding(idx)
            
            # Apply perturbation if specified
            if perturbation is not None:
                embedding = perturbation(embedding, **perturbation_params)
            
            # Add batch and time dimensions
            input_tensor = embedding[None, None, :, :, :].to(device)
            
            # Encode to get latent vector
            latent = model.encode(input_tensor, temperature=temperature, hard=True, noise_ratio=noise_ratio)
            latent = latent.cpu().numpy().squeeze()
            latent_vectors.append(latent)
            
            # Assign label
            label = assign_label(idx, flags)
            labels.append(label)

    latent_vectors = np.array(latent_vectors)
    labels = np.array(labels)

    # Calculate consistency for each state
    percentages = []
    for label in range(len(flags) + 1):
        label_mask = labels == label
        label_vectors = latent_vectors[label_mask]
        
        if len(label_vectors) == 0:
            percentages.append(0.0)
            continue
        
        # Find most common embedding
        unique_vectors, counts = np.unique(label_vectors, axis=0, return_counts=True)
        most_common_vector = unique_vectors[np.argmax(counts)]
        
        # Calculate match percentage
        matches = np.all(label_vectors == most_common_vector, axis=1)
        percentage = np.mean(matches)
        percentages.append(percentage)

    # Calculate weighted average
    counts = [np.sum(labels == label) for label in range(len(flags) + 1)]
    total = sum(counts)
    weighted_avg = np.dot(percentages, counts) / total if total > 0 else 0

    return weighted_avg, percentages

def load_model_from_config(config, ckpt, verbose=False):
    """Load the latent diffusion model from config and checkpoint."""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0 and verbose:
        print("Missing keys:", missing)
    if len(unexpected) > 0 and verbose:
        print("Unexpected keys:", unexpected)
    model.cuda()
    model.eval()
    return model

def load_img(path):
    """Load an image and prepare it for the perceptual autoencoder."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"Loaded image of size ({w}, {h}) from {path}")
    
    # Resize to 1280x720 for consistency
    target_size = (1280, 720)
    image = image.resize(target_size, resample=PIL.Image.LANCZOS)
    
    # Ensure dimensions are multiples of 32 (should already be the case with 1280x720)
    w, h = target_size
    w, h = map(lambda x: x - x % 32, (w, h))
    if (w, h) != target_size:
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # shape: [1, C, H, W]
    image = torch.from_numpy(image)
    return 2. * image - 1.

class ShuffledStatePairDataset(Dataset):
    """
    Args:
        input_data: Either a path to frames directory or a dictionary of embeddings
        state_segments:  List of (start_idx, end_idx) for each state
        test_pct:        Fraction of total frames (per state) to devote to test
        val_pct:         Fraction of total frames (per state) to devote to val
        transform:       Any transform to be applied to images (not needed for embeddings)
        mode:            One of ["train", "test", "val"] – determines which subset
                         of indices (train vs. test vs. val) is served by __getitem__.
    """
    def __init__(
        self, 
        input_data, 
        state_segments, 
        test_pct=0.1, 
        val_pct=0.1, 
        transform=None, 
        mode="train"
    ):
        super().__init__()
        # Determine if we're working with images or embeddings
        if isinstance(input_data, (str, Path)):
            self.frames_dir = input_data
            self.input_embeddings = None
            self.is_embeddings = False
        else:
            self.frames_dir = None
            self.input_embeddings = input_data
            self.is_embeddings = True
            
        self.state_segments = state_segments
        self.transform = transform
        self.mode = mode.lower().strip()
        self.num_states = len(self.state_segments)

        # Keep track of train/test/val indices for each state
        self.train_indices_per_state = []
        self.test_indices_per_state = []
        self.val_indices_per_state = []

        # For building your shuffled pairs, you may want to keep them
        # separate so you only sample pairs from the relevant subset:
        self.pairs_per_state = []

        # Compute the contiguous splits for each state
        for (start, end) in self.state_segments:
            full_indices = list(range(start, end)) 
            n = len(full_indices)
            # Size of combined test+val chunk
            test_val_count = int(n * (test_pct + val_pct))
            # The offset from the start to that "middle" chunk
            margin = (n - test_val_count) // 2

            # Middle chunk that will be test+val
            test_val_indices = full_indices[margin : margin + test_val_count]
            
            # The remaining frames (front chunk + back chunk) => train set
            train_indices = (full_indices[:margin] 
                             + full_indices[margin + test_val_count:])

            # Now split test+val portion. For example, if you want them
            # split proportionally to test_pct vs. val_pct:
            if test_val_count > 0:
                # fraction that is test out of that middle chunk
                test_fraction_of_middle = test_pct / (test_pct + val_pct)
                test_count = int(round(test_fraction_of_middle * test_val_count))
                test_indices  = test_val_indices[:test_count]
                val_indices   = test_val_indices[test_count:]
            else:
                # if there's no middle chunk at all, they are empty
                test_indices = []
                val_indices = []

            self.train_indices_per_state.append(train_indices)
            self.test_indices_per_state.append(test_indices)
            self.val_indices_per_state.append(val_indices)

        # Load all data into memory at initialization
        print(f"Loading all data into memory for {mode} dataset...")
        self.data = {}
        indices_to_load = []
        if self.mode == "train":
            for indices in self.train_indices_per_state:
                indices_to_load.extend(indices)
        elif self.mode == "test":
            for indices in self.test_indices_per_state:
                indices_to_load.extend(indices)
        elif self.mode == "val":
            for indices in self.val_indices_per_state:
                indices_to_load.extend(indices)

        def load_data(frame_index):
            if self.is_embeddings:
                # Get embedding from the dictionary using frame index as key
                key = f"{frame_index:010d}.jpg"
                data = self.input_embeddings.get(key)
                if data is None:
                    # Try without the .jpg extension
                    key = f"{frame_index:010d}"
                    data = self.input_embeddings.get(key)
                    
                if data is None:
                    raise KeyError(f"No embedding found for frame index {frame_index}")
                    
                # Convert to tensor if it's not already
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
            else:
                # Load and transform the image
                filename = f"{frame_index:010d}.jpg"
                path = os.path.join(self.frames_dir, filename)
                image = Image.open(path).convert("RGB")
                if self.transform is not None:
                    data = self.transform(image)
                else:
                    data = torch.tensor(np.array(image), dtype=torch.float32)
            
            return frame_index, data

        # Use a pool of workers to load data
        num_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers max
        with mp.Pool(num_workers) as pool:
            results = pool.map(load_data, indices_to_load)
            
        # Store loaded data in dictionary
        for frame_index, data in results:
            self.data[frame_index] = data

        print(f"Loaded {len(self.data)} items into memory")

        # Build pairs after loading data
        self._build_pairs()

    def _build_pairs(self):
        """
        Builds the pairs for each state, depending on which subset we're using
        (train, test, or val). By default, we'll just build pairs for the
        specified `self.mode`. If you want separate pair-lists for each split,
        you can do that too.
        """
        if self.mode == "train":
            all_state_indices = self.train_indices_per_state
        elif self.mode == "test":
            all_state_indices = self.test_indices_per_state
        elif self.mode == "val":
            all_state_indices = self.val_indices_per_state
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        self.pairs_per_state = []
        self.num_items = 0

        max_frames = 0
        for indices in all_state_indices:
            max_frames = max(max_frames, len(indices))

        for indices in all_state_indices:
            # If there are fewer frames than max_frames, pad them
            if len(indices) < max_frames and len(indices) > 0:
                padded = indices.copy() + random.choices(indices, k=max_frames - len(indices))
            else:
                padded = indices.copy()

            random.shuffle(padded)
            # Now form (frame1, frame2) pairs
            pairs = []
            for i in range(len(padded) // 2):
                pairs.append((padded[2*i], padded[2*i+1]))

            if len(padded) % 2 == 1:
                leftover = padded[-1]
                # pick any other index from 'indices' if possible
                if len(indices) > 1:
                    candidate = random.choice([x for x in indices if x != leftover])
                else:
                    candidate = leftover
                pairs.append((leftover, candidate))

            self.pairs_per_state.append(pairs)

        # The number of items is basically the number of pairs each state can supply
        # They should all have the same number of pairs
        max_pairs = max(len(pairs_list) for pairs_list in self.pairs_per_state)
        self.num_items = max_pairs

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        """
        Collects the (idx)-th pair from each state's pair list, wraps them
        into a single tensor. If a state's pair list is shorter than idx,
        we wrap around mod the length (or handle however you prefer).
        """
        pairs = []
        for pairs_list in self.pairs_per_state:
            if len(pairs_list) == 0:
                raise ValueError(f"State {idx} has no pairs")

            real_pair_idx = idx % len(pairs_list)
            pair = pairs_list[real_pair_idx]
            # Get data from memory instead of loading from disk
            data1 = self.data[pair[0]]
            data2 = self.data[pair[1]]
            pairs.append(torch.stack([data1, data2], dim=0))  # [2, C, H, W]

        # Combine [state_1, state_2, ...] => [2, T, C, H, W]
        pairs_tensor = torch.stack(pairs, dim=0).permute(1, 0, 2, 3, 4)
        return pairs_tensor

    def _load_embedding(self, frame_index):
        """
        Load embedding for a specific frame index from the input_embeddings dictionary.
        """
        if not self.is_embeddings:
            raise ValueError("This dataset is configured for images, not embeddings")
            
        # Get embedding from the dictionary using frame index as key
        key = f"{frame_index:010d}.jpg"
        embedding = self.input_embeddings.get(key)
        if embedding is None:
            # Try without the .jpg extension
            key = f"{frame_index:010d}"
            embedding = self.input_embeddings.get(key)
            
        if embedding is None:
            raise KeyError(f"No embedding found for frame index {frame_index}")
            
        # Convert to tensor if it's not already
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            
        return embedding

if __name__ == "__main__":
    # Set up paths and state segmentation
    frames_dir = Path(__file__).parent.parent.parent.parent.joinpath(
        "videos/frames/kid_playing_with_blocks"
    )
    input_dir = Path(__file__).parent.parent.parent.parent.joinpath(
        "videos/kid_playing_with_blocks_perceps.npy"
    )
    input_embeddings = np.load(input_dir, allow_pickle=True).item()  # dictionary

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
    
    # Setup test dataset
    test_dataset = ShuffledStatePairDataset(input_embeddings, state_segments, transform=None, mode="test")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load RBVAE models
    contrastive_latent_dim = 25
    contrastive_model = ContrastiveRBVAE(in_channels=3, out_channels=3, 
        latent_dim=contrastive_latent_dim, hidden_dim=contrastive_latent_dim)
    percep_latent_dim = 25
    percep_model = PercepRBVAE(in_channels=4, out_channels=4,
        latent_dim=percep_latent_dim, hidden_dim=percep_latent_dim)
    
    # Load model checkpoints
    contrastive_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "scripts/evaluation/best_models/pixels/best_model_breezy-sweep-38.pt"
    )
    percep_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "scripts/evaluation/best_models/perceps/best_model_grateful-sweep-19.pt"
    )
    
    contrastive_checkpoint = torch.load(contrastive_path, map_location=device)
    percep_checkpoint = torch.load(percep_path, map_location=device)
    
    contrastive_model.load_state_dict(contrastive_checkpoint['model_state_dict'])
    percep_model.load_state_dict(percep_checkpoint['model_state_dict'])
    
    contrastive_model.to(device)
    percep_model.to(device)
    
    # Define perturbation parameters
    perturbations = {
        'clean': None,
        'gaussian_noise': {'std': 0.1},
        'occlusion': {'coverage': 0.2}
    }
    
    # Store results
    results = []
    
    # Evaluate both models
    for model_name, model in [('Contrastive RBVAE', contrastive_model), ('Percep RBVAE', percep_model)]:
        for pert_name, pert_params in perturbations.items():
            if pert_name == 'clean':
                weighted_avg, state_percentages = calculate_state_consistency(
                    model, test_dataset, device, temperature=0.2, noise_ratio=0.1
                )
            elif pert_name == 'gaussian_noise':
                weighted_avg, state_percentages = calculate_state_consistency(
                    model, test_dataset, device, temperature=0.2, noise_ratio=0.1,
                    perturbation=add_gaussian_noise, perturbation_params=pert_params
                )
            else:  # occlusion
                weighted_avg, state_percentages = calculate_state_consistency(
                    model, test_dataset, device, temperature=0.2, noise_ratio=0.1,
                    perturbation=add_occlusion, perturbation_params=pert_params
                )
            
            # Store results
            result = {
                'Model': model_name,
                'Perturbation': pert_name,
                'Weighted Average': weighted_avg
            }
            for i, pct in enumerate(state_percentages):
                result[f'State {i}'] = pct
            results.append(result)
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    print("\nResults Table:")
    print(df.to_string(index=False))
    
    # Save results to CSV
    output_path = Path(__file__).parent.joinpath("state_consistency_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    x = np.arange(len(perturbations))
    width = 0.35
    
    plt.bar(x - width/2, df[df['Model'] == 'Contrastive RBVAE']['Weighted Average'], width, label='Contrastive RBVAE')
    plt.bar(x + width/2, df[df['Model'] == 'Percep RBVAE']['Weighted Average'], width, label='Percep RBVAE')
    
    plt.xlabel('Perturbation Type')
    plt.ylabel('Weighted Average State Consistency')
    plt.title('State Consistency Comparison')
    plt.xticks(x, list(perturbations.keys()))
    plt.legend()
    
    # Save plot
    plt.savefig(Path(__file__).parent.joinpath("state_consistency_comparison.png"))
    plt.close()
    
