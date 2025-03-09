import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import argparse
import os

# Import the model and training utilities
from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE
from models.contrastive_RBVAE.contrastive_RBVAE_train import (
    train_one_epoch, 
    ShuffledStatePairDataset, 
    ImageTransforms
)

def train_with_config():
    """
    Train a model with hyperparameters from wandb.config and log metrics
    """
    # Access the config values provided by W&B
    config = wandb.config
    
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up paths and state segmentation
    frames_dir = Path(config.frames_dir)
    last_frame = config.last_frame
    flags = config.flags
    grey_out = config.grey_out
    
    # Create state segments
    state_segments = []
    for i in range(len(flags)):
        if i > 0:
            state_segments.append((flags[i-1] + grey_out, flags[i] - grey_out + 1))
        elif i == len(flags)-1:
            state_segments.append((flags[i] + grey_out, last_frame + 1))
        else:
            state_segments.append((0, flags[0] - grey_out + 1))
    
    # Initialize the model with hyperparameters from config
    model = Seq2SeqBinaryVAE(
        in_channels=3, 
        out_channels=3, 
        latent_dim=config.latent_dim, 
        hidden_dim=config.hidden_dim
    ).to(device)
    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Setup dataset and dataloader
    dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        avg_total_loss, avg_recon_loss, avg_kl_loss, avg_triplet_loss = train_one_epoch(
            model=model,
            device=device,
            dataloader=dataloader,
            optimizer=optimizer,
            batch_size=config.batch_size,
            epoch=epoch,
            init_temperature=config.init_temperature,
            final_temperature=config.final_temperature,
            anneal_rate=config.anneal_rate,
            num_steps_to_update=config.num_steps_to_update,
            bernoulli_p=config.bernoulli_p,
            margin=config.margin,
            alpha_triplet=config.alpha_triplet,
            beta_kl=config.beta_kl
        )
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "total_loss": avg_total_loss,
            "reconstruction_loss": avg_recon_loss,
            "kl_divergence": avg_kl_loss,
            "triplet_loss": avg_triplet_loss
        })
        
        # Save the best model
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(wandb.run.dir, 'best_model.pt'))
            wandb.save('best_model.pt')
    
    # Return the best loss as the metric to optimize
    return best_loss

def train_with_wandb():
    """Wrapper function to initialize wandb run and call the training function"""
    # Train the model and get the best loss
    best_loss = train_with_config()
    
    # Log final metrics
    wandb.run.summary['best_loss'] = best_loss

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run W&B hyperparameter sweep for Contrastive RBVAE')
    parser.add_argument('--sweep_id', type=str, help='W&B sweep ID to join an existing sweep')
    parser.add_argument('--create_sweep', action='store_true', help='Create a new sweep instead of joining an existing one')
    parser.add_argument('--project_name', type=str, default='contrastive-rbvae', help='W&B project name')
    args = parser.parse_args()
    
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # Can be 'random', 'grid', or 'bayes'
        'metric': {
            'name': 'total_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'latent_dim': {
                'values': [16, 32, 64, 128]
            },
            'hidden_dim': {
                'values': [16, 32, 64, 128]
            },
            'init_temperature': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 2.0
            },
            'final_temperature': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.9
            },
            'anneal_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },
            'num_steps_to_update': {
                'values': [50, 100, 200, 500]
            },
            'bernoulli_p': {
                'distribution': 'uniform',
                'min': 0.3,
                'max': 0.7
            },
            'margin': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 2.0
            },
            'alpha_triplet': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 1.0
            },
            'beta_kl': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 1.0
            },
            'num_epochs': {
                'value': 30
            },
            # Fixed parameters - these aren't part of the sweep but are needed for training
            'frames_dir': {
                'value': str(Path(__file__).parent.parent.parent.joinpath("videos/frames/kid_playing_with_blocks_1.mp4"))
            },
            'last_frame': {
                'value': 1425
            },
            'flags': {
                'value': [152, 315, 486, 607, 734, 871, 1153, 1343]
            },
            'grey_out': {
                'value': 10
            }
        }
    }
    
    # Create or join a sweep
    if args.create_sweep:
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        if not sweep_id:
            print("Error: Please provide a sweep_id or use --create_sweep to create a new one")
            return
    
    # Start an agent that will run the training function with different configs
    wandb.agent(sweep_id, function=train_with_wandb, project=args.project_name)

if __name__ == "__main__":
    main() 