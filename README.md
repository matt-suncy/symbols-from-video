# Toward Learning Symbolic Representations from Video

## Abstract

This research investigates the extraction of state representations from videos, addressing a key challenge in computer vision: learning representations that align with symbolic planning. Typically, representations derived from deep learning models are incompatible with the symbolic structures utilized by classical planning methods. To bridge this gap, this study introduces a novel approach to extract propositional representations directly from video data by leveraging perceptual autoencoders—pretrained representation-learning models employed within Latent Diffusion Models (LDM). The proposed method utilizes a pretrained encoder to generate semantically compressed embeddings for individual video frames to be used as inputs. Subsequently, to achieve representations compatible with symbolic planning, we train an autoencoder specifically designed to produce binary representations from these semantic embeddings, capturing visual features in a compact, discrete form. The architecture of this autoencoder supports the extraction of high-level symbolic information. Ultimately, this framework aims to explore the capabilities and limitations of vision-based neural networks in learning representations conducive to symbolic planning directly from video data.

![RBVAE_architecture-cropped](https://github.com/user-attachments/assets/7b7375ee-4a47-4b2c-8ed7-6b02f442b7cd)

## Usage

## Requirements


## Citing This Work
