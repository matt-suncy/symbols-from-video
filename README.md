# Toward Learning Symbolic Representations from Video

## Abstract

This research investigates the extraction of state representations from videos, addressing a key challenge in computer vision: learning representations that align with symbolic planning. Typically, representations derived from deep learning models are incompatible with the symbolic structures utilized by classical planning methods. To bridge this gap, this study introduces a novel approach to extract propositional representations directly from video data by leveraging perceptual autoencoders—pretrained representation-learning models employed within Latent Diffusion Models (LDM). The proposed method utilizes a pretrained encoder to generate semantically compressed embeddings for individual video frames to be used as inputs. Subsequently, to achieve representations compatible with symbolic planning, we train an autoencoder specifically designed to produce binary representations from these semantic embeddings, capturing visual features in a compact, discrete form. The architecture of this autoencoder supports the extraction of high-level symbolic information. Ultimately, this framework aims to explore the capabilities and limitations of vision-based neural networks in learning representations conducive to symbolic planning directly from video data.

![RBVAE_architecture-cropped](https://github.com/user-attachments/assets/7b7375ee-4a47-4b2c-8ed7-6b02f442b7cd)

## Usage

The videos used for training and model weights can be found here: https://drive.google.com/drive/folders/1RWonm0SOELsXIRCMIL7Ou1CyitJtQZLC?usp=sharing

After downloading the folders for each video, move them to the "videos" directory.

The model architectures, training, and hyperparameter sweep code can be found in "models/contrastive_RBVAE" (trained on pixels) and "models/percep_RBVAE" (trained on precomputed embeddings). It is recommended that after downloading the models to a new directory with path "scripts/evaluation/best_models".

Transition flags and other details about the videos concerning training can be found in "videos/frames/transition_flag.txt".

Refer to "scripts/evaluation/best_models.txt" for relevant model information for running evaluations.

## Requirements

Create a new Python 3.8.5 virtual environment and download from "requirements.txt".

## Citing This Work
