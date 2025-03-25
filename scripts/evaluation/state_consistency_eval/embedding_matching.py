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

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE as ContrastiveRBVAE
from models.percep_RBVAE.percep_RBVAE_model import Seq2SeqBinaryVAE as PercepRBVAE
from ldm.util import instantiate_from_config

# This is a CALLABLE for images being passed to ContrastiveRBVAE
RESOLUTION = 256
ImageTransforms = T.Compose([
    T.Resize((RESOLUTION, RESOLUTION)),
    T.ToTensor()
])

class TestDataset(Dataset):
    """
    Dataset class for state consistency evaluation that loads all images into memory.
    Args:
        frames_dir: Path to the directory containing the frames
        state_segments: List of (start_idx, end_idx) for each state
        test_pct: Fraction of total frames (per state) to devote to test
        val_pct: Fraction of total frames (per state) to devote to val
        mode: One of ["train", "test", "val"] - determines which subset of indices to use
    """
    def __init__(
        self,
        frames_dir,
        state_segments,
        test_pct=0.1,
        val_pct=0.1,
        mode="test"
    ):
        super().__init__()
        self.state_segments = state_segments
        self.mode = mode.lower().strip()
        self.num_states = len(self.state_segments)

        # Keep track of train/test/val indices for each state
        self.train_indices_per_state = []
        self.test_indices_per_state = []
        self.val_indices_per_state = []

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

            # Now split test+val portion
            if test_val_count > 0:
                # fraction that is test out of that middle chunk
                test_fraction_of_middle = test_pct / (test_pct + val_pct)
                test_count = int(round(test_fraction_of_middle * test_val_count))
                test_indices = test_val_indices[:test_count]
                val_indices = test_val_indices[test_count:]
            else:
                test_indices = []
                val_indices = []

            self.train_indices_per_state.append(train_indices)
            self.test_indices_per_state.append(test_indices)
            self.val_indices_per_state.append(val_indices)

        # Get the indices for the current mode
        if self.mode == "train":
            self.indices = []
            for indices in self.train_indices_per_state:
                self.indices.extend(indices)
        elif self.mode == "test":
            self.indices = []
            for indices in self.test_indices_per_state:
                self.indices.extend(indices)
        elif self.mode == "val":
            self.indices = []
            for indices in self.val_indices_per_state:
                self.indices.extend(indices)
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        # Load all frames into memory
        print(f"Loading all frames into memory for {mode} dataset...")
        self.frames = {}
        for frame_index in self.indices:
            filename = f"{frame_index:010d}.jpg"
            path = os.path.join(frames_dir, filename)
            image = Image.open(path).convert("RGB")
            self.frames[frame_index] = image

        print(f"Loaded {len(self.frames)} frames into memory")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_index = self.indices[idx]
        return self.frames[frame_index]

# Load Stable Diffusion model for perceptual embeddings
def load_sd_model(config_path, ckpt_path):
    """Load the Stable Diffusion model for generating perceptual embeddings."""
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    return model

def generate_perceptual_embedding(image, sd_model, device):
    """Generate perceptual embedding for an image using Stable Diffusion model."""
    with torch.no_grad():
        encoded = sd_model.encode_first_stage(image)
        latent_embedding = sd_model.get_first_stage_encoding(encoded)
    return latent_embedding.squeeze().cpu().numpy()

# Functions for robustness to noise and occlusion tests

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
    
    # return noisy.squeeze(0) if tensor.dim() == 3 else noisy
    return noisy.squeeze()


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
    
    # return occluded.squeeze(0) if tensor.dim() == 3 else occluded
    return occluded.squeeze()


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

def calculate_state_consistency(model, test_dataset, device, sd_model=None, temperature=0.5, noise_ratio=0.1, perturbation=None, perturbation_params=None, transform=None):
    """
    Calculates state consistency for a model with optional perturbation.
    Args:
        model: The RBVAE model to evaluate
        test_dataset: Dataset containing test frames
        device: Device to run model on
        sd_model: Stable Diffusion model for generating perceptual embeddings
        temperature: Temperature for binary concrete
        noise_ratio: Noise ratio for binary concrete
        perturbation: Optional perturbation function (add_gaussian_noise or add_occlusion)
        perturbation_params: Parameters for the perturbation function
        transform: Transform to be applied to images (only used for ContrastiveRBVAE)
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
            # Get raw image
            image = test_dataset.frames[idx]
            
            # Apply transform if needed (for ContrastiveRBVAE)
            if transform is not None:
                image = transform(image)
            
            # Apply perturbation if needed
            if perturbation is not None:
                image = T.ToTensor()(image) if not isinstance(image, torch.Tensor) else image
                image = perturbation(image, **perturbation_params)
                # Convert back to PIL image
                image = T.ToPILImage()(image)

            # Generate input tensor based on model type
            if sd_model is not None:  # Perceptual model
                # Convert PIL image to numpy array and prepare for SD model
                image = load_img_for_sd(image)
                # Generate perceptual embedding from perturbed image
                image = image.to(device)
                embedding = generate_perceptual_embedding(image, sd_model, device)
                input_tensor = torch.from_numpy(embedding)[None, None, :, :, :].to(device)
            else:  # Contrastive model
                # Use perturbed image directly
                image = T.ToTensor()(image) if not isinstance(image, torch.Tensor) else image
                input_tensor = image[None, None, :, :, :].to(device)
            
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

def load_img_for_sd(image):
    """Load an image and prepare it for the perceptual autoencoder."""
    # Ensure image is in RGB mode
    image = image.convert("RGB")
    w, h = image.size
    print(f"Processing image of size ({w}, {h})")
    
    # Resize to 1280x720 for consistency
    target_size = (1280, 720)
    image = image.resize(target_size, resample=Image.LANCZOS)
    
    # Ensure dimensions are multiples of 32 (should already be the case with 1280x720)
    w, h = target_size
    w, h = map(lambda x: x - x % 32, (w, h))
    if (w, h) != target_size:
        image = image.resize((w, h), resample=Image.LANCZOS)
    
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # shape: [1, C, H, W]
    image = torch.from_numpy(image)
    return 2. * image - 1.

if __name__ == "__main__":
    # Set up paths and state segmentation
    frames_dir = Path(__file__).parent.parent.parent.parent.joinpath(
        "videos/frames/C10118_rgb"
    )

    last_frame = 2469
    flags = [157, 205, 441, 494, 557, 887, 909, 1010, 1048, 1315, 1388, 1438, 1702, 1847, 2096, 2174]
    grey_out = 1
    
    # Create state segments
    state_segments = []
    for i in range(len(flags)):
        if i > 0:
            state_segments.append((flags[i-1] + grey_out + 1, flags[i] - grey_out))
        else:
            state_segments.append((0, flags[0] - grey_out))
    state_segments.append((flags[-1] + grey_out + 1, last_frame + 1))
    
    # Setup test dataset for raw images (no transforms applied yet)
    test_dataset = TestDataset(frames_dir, state_segments, test_pct=0.1, val_pct=0.1, mode="test")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load RBVAE models
    contrastive_latent_dim = 50
    contrastive_model = ContrastiveRBVAE(in_channels=3, out_channels=3, 
        latent_dim=contrastive_latent_dim, hidden_dim=contrastive_latent_dim)
    percep_latent_dim = 50
    percep_model = PercepRBVAE(in_channels=4, out_channels=4,
        latent_dim=percep_latent_dim, hidden_dim=percep_latent_dim)
    
    # Load model checkpoints
    contrastive_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "scripts/evaluation/best_models/pixels/best_model_rare-sweep-12.pt"
    )
    percep_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "scripts/evaluation/best_models/perceps/best_model_fresh-sweep-9.pt"
    )
    
    contrastive_checkpoint = torch.load(contrastive_path, map_location=device)
    percep_checkpoint = torch.load(percep_path, map_location=device)
    
    contrastive_model.load_state_dict(contrastive_checkpoint['model_state_dict'])
    percep_model.load_state_dict(percep_checkpoint['model_state_dict'])
    
    contrastive_model.to(device)
    percep_model.to(device)
    
    # Load Stable Diffusion model for perceptual embeddings
    sd_config_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "src/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    )
    sd_ckpt_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "src/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
    )
    sd_model = load_sd_model(sd_config_path, sd_ckpt_path)
    sd_model.to(device)
    
    # Define perturbation parameters
    perturbations = {
        'clean': None,
        'gaussian_noise': {'std': 0.1},
        'occlusion': {'coverage': 0.2}
    }
    
    # Number of trials to run
    num_trials = 10
    
    # Dictionary to store all results across trials
    all_trial_results = {}
    
    print(f"Running {num_trials} trials...")
    
    # Run multiple trials
    for trial in range(num_trials):
        print(f"\nRunning trial {trial+1}/{num_trials}...")
        
        # Store results for this trial
        trial_results = []
        
        # Evaluate both models with their corresponding datasets
        for model_name, model, use_sd in [
            ('Contrastive RBVAE', contrastive_model, False),
            ('Percep RBVAE', percep_model, True)
        ]:
            for pert_name, pert_params in perturbations.items():
                if pert_name == 'clean':
                    weighted_avg, state_percentages = calculate_state_consistency(
                        model, test_dataset, device, sd_model if use_sd else None,
                        temperature=0.2, noise_ratio=0.1,
                        transform=ImageTransforms if not use_sd else None
                    )
                elif pert_name == 'gaussian_noise':
                    weighted_avg, state_percentages = calculate_state_consistency(
                        model, test_dataset, device, sd_model if use_sd else None,
                        temperature=0.2, noise_ratio=0.1,
                        perturbation=add_gaussian_noise, perturbation_params=pert_params,
                        transform=ImageTransforms if not use_sd else None
                    )
                else:  # occlusion
                    weighted_avg, state_percentages = calculate_state_consistency(
                        model, test_dataset, device, sd_model if use_sd else None,
                        temperature=0.2, noise_ratio=0.1,
                        perturbation=add_occlusion, perturbation_params=pert_params,
                        transform=ImageTransforms if not use_sd else None
                    )
                
                # Store results
                result = {
                    'Model': model_name,
                    'Perturbation': pert_name,
                    'Weighted Average': weighted_avg
                }
                for i, pct in enumerate(state_percentages):
                    result[f'State {i}'] = pct
                trial_results.append(result)
        
        # Store results for this trial
        trial_df = pd.DataFrame(trial_results)
        
        # Print results for this trial
        print(f"\nTrial {trial+1} Results:")
        print(trial_df.to_string(index=False))
        
        # Save trial results to CSV
        trial_output_path = Path(__file__).parent.joinpath(f"trial_{trial+1}_results.csv")
        trial_df.to_csv(trial_output_path, index=False)
        
        # Store results in all_trial_results dictionary for averaging
        for result in trial_results:
            key = (result['Model'], result['Perturbation'])
            if key not in all_trial_results:
                all_trial_results[key] = {
                    'Weighted Average': [result['Weighted Average']],
                }
                for i in range(len(flags) + 1):
                    all_trial_results[key][f'State {i}'] = [result[f'State {i}']]
            else:
                all_trial_results[key]['Weighted Average'].append(result['Weighted Average'])
                for i in range(len(flags) + 1):
                    all_trial_results[key][f'State {i}'].append(result[f'State {i}'])
    
    # Calculate average results across all trials
    average_results = []
    for key, values in all_trial_results.items():
        model_name, pert_name = key
        result = {
            'Model': model_name,
            'Perturbation': pert_name,
            'Weighted Average': np.mean(values['Weighted Average'])
        }
        for i in range(len(flags) + 1):
            result[f'State {i}'] = np.mean(values[f'State {i}'])
        average_results.append(result)
    
    # Create DataFrame for average results
    avg_df = pd.DataFrame(average_results)
    
    # Calculate standard deviations
    std_results = []
    for key, values in all_trial_results.items():
        model_name, pert_name = key
        result = {
            'Model': model_name,
            'Perturbation': pert_name,
            'Weighted Average': np.std(values['Weighted Average'])
        }
        for i in range(len(flags) + 1):
            result[f'State {i}'] = np.std(values[f'State {i}'])
        std_results.append(result)
    
    # Create DataFrame for standard deviations
    std_df = pd.DataFrame(std_results)
    
    # Print average results
    print("\nAverage Results Across All Trials:")
    print(avg_df.to_string(index=False))
    
    # Print standard deviations
    print("\nStandard Deviations Across All Trials:")
    print(std_df.to_string(index=False))
    
    # Save average results to CSV
    avg_output_path = Path(__file__).parent.joinpath("average_results.csv")
    avg_df.to_csv(avg_output_path, index=False)
    
    # Save standard deviations to CSV
    std_output_path = Path(__file__).parent.joinpath("std_results.csv")
    std_df.to_csv(std_output_path, index=False)
    
    print(f"\nAverage results saved to {avg_output_path}")
    print(f"Standard deviations saved to {std_output_path}")
    
    # Plot average results
    plt.figure(figsize=(12, 6))
    x = np.arange(len(perturbations))
    width = 0.35
    
    contrastive_avg = avg_df[avg_df['Model'] == 'Contrastive RBVAE']['Weighted Average'].values
    percep_avg = avg_df[avg_df['Model'] == 'Percep RBVAE']['Weighted Average'].values
    
    contrastive_std = std_df[std_df['Model'] == 'Contrastive RBVAE']['Weighted Average'].values
    percep_std = std_df[std_df['Model'] == 'Percep RBVAE']['Weighted Average'].values
    
    plt.bar(x - width/2, contrastive_avg, width, yerr=contrastive_std, 
            label='Contrastive RBVAE', capsize=5)
    plt.bar(x + width/2, percep_avg, width, yerr=percep_std, 
            label='Percep RBVAE', capsize=5)
    
    plt.xlabel('Perturbation Type')
    plt.ylabel('Average Weighted State Consistency')
    plt.title(f'State Consistency Comparison (Average of {num_trials} Trials)')
    plt.xticks(x, list(perturbations.keys()))
    plt.legend()
    
    # Save plot
    plot_path = Path(__file__).parent.joinpath("average_state_consistency_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Comparison plot saved to {plot_path}")
    
