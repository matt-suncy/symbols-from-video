import os
import glob
import torch
import numpy as np
import PIL
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.util import instantiate_from_config

# ----------------------------
# Configurable variables
# ----------------------------
# Change working directory to this file's directory
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# Path to the folder containing input images
IMAGE_FOLDER = "/home/jovyan/Documents/latplan-temporal-segmentation/videos/frames/kid_playing_with_blocks_1.mp4"
# Path to the model configuration file (YAML)
CONFIG_PATH = os.path.join(BASE_DIR, "configs/stable-diffusion/v1-inference.yaml")
# Path to the model checkpoint file
CKPT_PATH = os.path.join(BASE_DIR, "models/ldm/stable-diffusion-v1/model.ckpt")
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
# Output file to save embeddings
OUTPUT_EMBEDDINGS_FILE = "embeddings.npy"

# ----------------------------
# Utility functions
# ----------------------------
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
    """Load an image and prepare it for the perceptual autoencoder.
    
    The image is converted to RGB, resized to an integer multiple of 32, normalized to [-1, 1],
    and converted to a torch tensor.
    """
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"Loaded image of size ({w}, {h}) from {path}")
    # Resize to integer multiple of 32
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # shape: [1, C, H, W]
    image = torch.from_numpy(image)
    return 2. * image - 1.

# ----------------------------
# Main processing
# ----------------------------
def main():
    # Use CUDA if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load configuration and model
    config = OmegaConf.load(CONFIG_PATH)
    model = load_model_from_config(config, CKPT_PATH)
    model = model.to(device)
    
    # Get list of image paths in the specified folder
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*"))
    if not image_paths:
        print("No images found in the folder:", IMAGE_FOLDER)
        return

    embeddings = {}  # dictionary to store embeddings (key: image path, value: embedding array)

    # Process each image individually
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and preprocess the image
            img = load_img(img_path).to(device)
            # Compute latent embedding via the perceptual autoencoder.
            # Here, encode_first_stage returns the encoded representation and
            # get_first_stage_encoding converts it to the latent space.
            with torch.no_grad():
                encoded = model.encode_first_stage(img)
                latent_embedding = model.get_first_stage_encoding(encoded)
            # Convert the tensor to a numpy array and store it
            embeddings[img_path] = latent_embedding.cpu().numpy()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Save all embeddings to a single NumPy file
    np.save(OUTPUT_EMBEDDINGS_FILE, embeddings)
    print(f"\nSaved embeddings for {len(embeddings)} images to {OUTPUT_EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()
