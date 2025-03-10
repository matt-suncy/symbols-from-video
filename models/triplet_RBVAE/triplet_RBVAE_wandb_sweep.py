import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import argparse
import os

from contrastive_RBVAE_model import Seq2SeqBinaryVAE
from contrastive_RBVAE_train import (
    ContrastiveRBVAETrainer,
    ShuffledStatePairDataset,
    ImageTransforms
)

def train_with_config():
    """
    Train a model with hyperparameters from wandb.config and log metrics
    """
    # Access the config values provided by W&B
    config = wandb.config
    
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
    
    # Setup datasets and dataloaders
    train_dataset = ShuffledStatePairDataset(
        frames_dir, 
        state_segments, 
        transform=ImageTransforms, 
        mode="train"
    )
    val_dataset = ShuffledStatePairDataset(
        frames_dir, 
        state_segments, 
        transform=ImageTransforms, 
        mode="val"
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with config hyperparameters
    model = Seq2SeqBinaryVAE(
        in_channels=3,
        out_channels=3,
        latent_dim=config.latent_dim,
        hidden_dim=config.latent_dim
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Calculate num_steps_to_update based on config
    max_iters = config.num_epochs * len(train_dataset)
    num_steps_to_update = int(max_iters / config.num_temp_updates)
    
    # Create unique log directory for this run
    log_dir = Path("./runs") / wandb.run.name
    
    # Initialize trainer with config hyperparameters
    trainer = ContrastiveRBVAETrainer(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        init_temperature=config.init_temperature,
        final_temperature=config.final_temperature,
        anneal_rate=config.anneal_rate,
        num_steps_to_update=num_steps_to_update,
        bernoulli_p=config.bernoulli_p,
        margin=config.margin,
        alpha_triplet=config.alpha_triplet,
        beta_kl=config.beta_kl,
        log_dir=log_dir
    )
    
    # Train the model and get history
    save_path = Path("./models") / f"model_{wandb.run.name}.pt"
    history = trainer.train(num_epochs=config.num_epochs, save_path=save_path)
    
    # Log final metrics to wandb
    wandb.log({
        "best_val_loss": history["best_val_loss"],
        "best_epoch": history["best_epoch"]
    })
    
    # Save the best model to wandb
    if trainer.best_model_state is not None:
        best_model_path = Path("./models") / f"best_model_{wandb.run.name}.pt"
        torch.save({
            'model_state_dict': trainer.best_model_state,
            'config': dict(config),
            'best_epoch': history["best_epoch"],
            'best_val_loss': history["best_val_loss"]
        }, best_model_path)
        wandb.save(str(best_model_path))
    
    return history["best_val_loss"]

def train_with_wandb():
    """Wrapper function to initialize wandb run and call the training function"""
    best_val_loss = train_with_config()
    wandb.run.summary['best_val_loss'] = best_val_loss

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run W&B hyperparameter sweep for Contrastive RBVAE')
    parser.add_argument('--sweep_id', type=str, help='W&B sweep ID to join an existing sweep')
    parser.add_argument('--create_sweep', action='store_true', help='Create a new sweep instead of joining an existing one')
    parser.add_argument('--project_name', type=str, default='contrastive-rbvae', help='W&B project name')
    args = parser.parse_args()
    
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'best_val_loss',
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
            'init_temperature': {
                'distribution': 'uniform',
                'min': 1.0,
                'max': 5.0
            },
            'final_temperature': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.7
            },
            'anneal_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },
            'num_temp_updates': {
                'distribution': 'int_uniform',
                'min': 550,
                'max': 1100
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
            # Fixed parameters
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