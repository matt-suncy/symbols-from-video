import sys
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE
from models.percep_RBVAE.percep_RBVAE_model import Seq2SeqBinaryVAE as PercepBinaryVAE

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

def hamming_distance(vec1, vec2):
    """
    Calculate the Hamming distance between two binary vectors.
    """
    return np.sum(vec1 != vec2)

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

def find_most_common_vector(vectors):
    """
    Find the most common binary vector in a list of vectors.
    """
    # Convert numpy arrays to tuples for hashing
    vector_tuples = [tuple(v) for v in vectors]
    # Find the most common vector
    most_common = Counter(vector_tuples).most_common(1)[0][0]
    # Convert back to numpy array
    return np.array(most_common)

def plot_hamming_distances(hamming_distances, model_name, output_path):
    """
    Plot the Hamming distances between adjacent states.
    """
    plt.figure(figsize=(12, 6))
    
    x = range(len(hamming_distances))
    plt.bar(x, hamming_distances)
    
    plt.xlabel('State Transition')
    plt.ylabel('Hamming Distance')
    plt.title(f'Hamming Distance Between Adjacent States - {model_name}')
    
    # Add labels for state transitions
    state_transition_labels = [f'States {i}-{i+1}' for i in range(len(hamming_distances))]
    plt.xticks(x, state_transition_labels, rotation=45)
    
    # Add values above bars
    for i, v in enumerate(hamming_distances):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

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

    # Load Contrastive RBVAE model
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

    # Load Perceptual RBVAE model
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

    # Collect latent vectors for each state
    contrastive_state_vectors = [[] for _ in range(len(flags) + 1)]
    percep_state_vectors = [[] for _ in range(len(flags) + 1)]

    print("Processing test frames...")
    with torch.no_grad():
        for idx in test_indices:
            # Get state label
            state_label = assign_label(idx, flags)
            
            # For contrastive model
            frame = ImageTransforms(Image.open(os.path.join(frames_dir, f"{idx:010d}.jpg")).convert("RGB"))
            input_tensor = frame[None, None, :, :, :].to(device)
            latent = contrastive_model.encode(input_tensor, temperature=0.2, hard=True)
            contrastive_state_vectors[state_label].append(latent.cpu().numpy().squeeze())

            # For perceptual model
            key = f"{idx:010d}.jpg"
            percep_embedding = percep_embeddings.get(key)
            if percep_embedding is None:
                key = f"{idx:010d}"
                percep_embedding = percep_embeddings.get(key)
            percep_tensor = torch.tensor(percep_embedding, dtype=torch.float32)[None, :, :, :].to(device)
            latent = percep_model.encode(percep_tensor, temperature=0.2, hard=True)
            percep_state_vectors[state_label].append(latent.cpu().numpy().squeeze())

    # Find most common vector for each state
    contrastive_most_common = []
    percep_most_common = []
    
    for state in range(len(flags) + 1):
        if contrastive_state_vectors[state]:
            contrastive_most_common.append(find_most_common_vector(contrastive_state_vectors[state]))
        else:
            print(f"Warning: No contrastive vectors for state {state}")
            contrastive_most_common.append(np.zeros(contrastive_latent_dim))
            
        if percep_state_vectors[state]:
            percep_most_common.append(find_most_common_vector(percep_state_vectors[state]))
        else:
            print(f"Warning: No perceptual vectors for state {state}")
            percep_most_common.append(np.zeros(percep_latent_dim))

    # Calculate Hamming distances between adjacent states
    contrastive_hamming_distances = []
    percep_hamming_distances = []
    
    for i in range(len(contrastive_most_common) - 1):
        contrastive_dist = hamming_distance(contrastive_most_common[i], contrastive_most_common[i+1])
        contrastive_hamming_distances.append(contrastive_dist)
        
        percep_dist = hamming_distance(percep_most_common[i], percep_most_common[i+1])
        percep_hamming_distances.append(percep_dist)

    # Print results
    print("\nContrastive RBVAE Hamming Distances between adjacent states:")
    for i, dist in enumerate(contrastive_hamming_distances):
        print(f"States {i}-{i+1}: {dist}")
    
    print("\nPerceptual RBVAE Hamming Distances between adjacent states:")
    for i, dist in enumerate(percep_hamming_distances):
        print(f"States {i}-{i+1}: {dist}")
    
    # Calculate average Hamming distances
    contrastive_avg = np.mean(contrastive_hamming_distances)
    percep_avg = np.mean(percep_hamming_distances)
    
    print(f"\nContrastive RBVAE Average Hamming Distance: {contrastive_avg:.2f}")
    print(f"Perceptual RBVAE Average Hamming Distance: {percep_avg:.2f}")
    
    # Plot Hamming distances
    plot_hamming_distances(
        contrastive_hamming_distances, 
        "Contrastive RBVAE",
        os.path.join(os.path.dirname(__file__), "Contrastive_RBVAE_Hamming_Distances")
    )
    
    plot_hamming_distances(
        percep_hamming_distances, 
        "Perceptual RBVAE",
        os.path.join(os.path.dirname(__file__), "Perceptual_RBVAE_Hamming_Distances")
    )
    
    # Create combined bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(contrastive_hamming_distances))
    width = 0.35
    
    plt.bar(x - width/2, contrastive_hamming_distances, width, label='Contrastive RBVAE')
    plt.bar(x + width/2, percep_hamming_distances, width, label='Perceptual RBVAE')
    
    plt.xlabel('State Transition')
    plt.ylabel('Hamming Distance')
    plt.title('Hamming Distance Between Adjacent States - Model Comparison')
    
    # Add labels for state transitions
    state_transition_labels = [f'States {i}-{i+1}' for i in range(len(contrastive_hamming_distances))]
    plt.xticks(x, state_transition_labels, rotation=45)
    
    plt.legend()
    plt.tight_layout()
    
    # Save comparison plot
    plt.savefig(os.path.join(os.path.dirname(__file__), "Hamming_Distances_Comparison"))
    plt.close()
    
    # Save data to CSV
    import pandas as pd
    
    # Create a DataFrame with the Hamming distances
    data = {
        'State Transition': [f'States {i}-{i+1}' for i in range(len(contrastive_hamming_distances))],
        'Contrastive RBVAE': contrastive_hamming_distances,
        'Perceptual RBVAE': percep_hamming_distances
    }
    
    df = pd.DataFrame(data)
    df.loc[len(df)] = ['Average', contrastive_avg, percep_avg]
    
    # Save to CSV
    df.to_csv(os.path.join(os.path.dirname(__file__), "Hamming_Distances.csv"), index=False)
    print(f"Results saved to {os.path.join(os.path.dirname(__file__), 'Hamming_Distances.csv')}") 