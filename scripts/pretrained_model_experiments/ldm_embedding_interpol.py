import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if len(missing_keys) > 0 and verbose:
        print("Missing keys:")
        print(missing_keys)
    if len(unexpected_keys) > 0 and verbose:
        print("Unexpected keys:")
        print(unexpected_keys)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"Loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # Resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def slerp(t, v0, v1):
    """Spherical linear interpolation"""
    v0_flat = v0.view(-1)
    v1_flat = v1.view(-1)
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if sin_omega == 0:
        return (1.0 - t) * v0 + t * v1  # LERP
    else:
        return (torch.sin((1.0 - t) * omega) / sin_omega) * v0 + (torch.sin(t * omega) / sin_omega) * v1


def interpolate_embeddings(embedding1, embedding2, steps=5, method='linear'):
    embeddings = []
    for i in range(steps):
        t = i / (steps - 1)
        if method == 'linear':
            inter_emb = (1 - t) * embedding1 + t * embedding2
        elif method == 'spherical':
            inter_emb = slerp(t, embedding1, embedding2)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        embeddings.append(inter_emb)
    return embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img1",
        type=str,
        required=True,
        help="Path to the first input image"
    )

    parser.add_argument(
        "--init-img2",
        type=str,
        required=True,
        help="Path to the second input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/interpolation",
        help="Directory to write results to"
    )

    parser.add_argument(
        "--interpolation_method",
        type=str,
        choices=["linear", "spherical"],
        default="linear",
        help="Interpolation method"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of interpolation steps"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="Path to config which constructs model"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="Path to checkpoint of model"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed (for reproducible sampling)"
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["full", "autocast"],
        default="autocast",
        help="Evaluate at this precision"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    os.makedirs(opt.outdir, exist_ok=True)
    base_count = len(os.listdir(opt.outdir))

    assert os.path.isfile(opt.init_img1), f"File {opt.init_img1} does not exist"
    assert os.path.isfile(opt.init_img2), f"File {opt.init_img2} does not exist"
    image1 = load_img(opt.init_img1).to(device)
    image2 = load_img(opt.init_img2).to(device)

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                # Encode images to latent space
                embedding1 = model.get_first_stage_encoding(model.encode_first_stage(image1))
                embedding2 = model.get_first_stage_encoding(model.encode_first_stage(image2))

                # Interpolate between embeddings
                interpolated_embeddings = interpolate_embeddings(
                    embedding1, embedding2,
                    steps=opt.steps,
                    method=opt.interpolation_method
                )

                # Decode embeddings back to images
                for i, embedding in enumerate(interpolated_embeddings):
                    decoded_image = model.decode_first_stage(embedding.unsqueeze(0))
                    x_sample = torch.clamp((decoded_image + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = x_sample.cpu().numpy()
                    x_sample = 255. * rearrange(x_sample[0], 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(opt.outdir, f"interpolated_{i+base_count:05}.png"))
                    print(f"Saved image {i+1}/{opt.steps} to {opt.outdir}")

    print(f"Interpolation completed. Images saved to {opt.outdir}")


if __name__ == "__main__":
    main()
