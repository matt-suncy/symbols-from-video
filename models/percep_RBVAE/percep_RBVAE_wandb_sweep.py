import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import argparse
import os

from percep_RBVAE_model import Seq2SeqBinaryVAE
from percep_RBVAE_train import (
    ContrastiveRBVAETrainer,
    ShuffledStatePairDataset
)

# NOTE: Change data paths here 
FRAMES_PATH = "/home/jovyan/Documents/latplan-temporal-segmentation/videos/frames/chin_chess"
EMBEDDINGS_PATH = "/home/jovyan/Documents/latplan-temporal-segmentation/videos/chin_chess_perceps.npy"

def train_with_config():
    """
    Train a model with hyperparameters from wandb.config and log metrics
    """
    wandb.init()
    # Access the config values provided by W&B
    config = wandb.config
    # print("WandB Config:", config)  # Debug print to see what's inside
     
    # Load embeddings
    embeddings_path = config.get("embeddings_path", EMBEDDINGS_PATH)
    input_embeddings = np.load(embeddings_path, allow_pickle=True).item()
    
    # Set up paths and state segmentation
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
        input_embeddings, 
        state_segments, 
        transform=None,  # No transforms needed for embeddings
        mode="train"
    )
    val_dataset = ShuffledStatePairDataset(
        input_embeddings, 
        state_segments, 
        transform=None,  # No transforms needed for embeddings
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
        in_channels=4,  # Update to match embedding dimensions
        out_channels=4,  # Update to match embedding dimensions
        latent_dim=config.latent_dim,
        hidden_dim=config.latent_dim
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Calculate num_steps_to_update based on total steps and desired number of updates
    total_steps = config.num_epochs * len(train_dataset)
    num_steps_to_update = max(1, total_steps // config.num_temp_updates)
    
    # Determine the base directory (i.e., the sweep program's directory)
    base_dir = Path(__file__).resolve().parent
    
    # Create unique log directory for this run inside the base directory
    log_dir = base_dir / "runs" / wandb.run.name
    log_dir.mkdir(parents=True, exist_ok=True)
    
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
        noise_ratio=config.noise_ratio,
        margin=config.margin,
        alpha_contrast=config.alpha_contrast,
        beta_kl=config.beta_kl,
        log_dir=log_dir,
        flags=flags
    )
    
    # Create the models directory inside the base directory
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model and get history, saving the model to the models directory
    save_path = models_dir / f"model_{wandb.run.name}.pt"
    history = trainer.train(num_epochs=config.num_epochs, save_path=save_path)
    
    # Log final metrics to wandb
    wandb.log({
        "best_consistency_score": history["best_consistency"],
        "best_epoch": history["best_epoch"],
        "final_temperature": trainer.current_temperature,
        "total_steps": trainer.global_step,
        "num_temp_updates": config.num_temp_updates,
        "actual_num_updates": trainer.global_step // num_steps_to_update
    })
    
    # Save the best model to wandb inside the models directory
    if trainer.best_model_state is not None:
        best_model_path = models_dir / f"best_model_{wandb.run.name}.pt"
        torch.save({
            'model_state_dict': trainer.best_model_state,
            'config': dict(config),
            'best_epoch': history["best_epoch"],
            'best_consistency_score': history["best_consistency"],
            'final_temperature': trainer.current_temperature,
            'total_steps': trainer.global_step,
            'num_temp_updates': config.num_temp_updates,
            'actual_num_updates': trainer.global_step // num_steps_to_update
        }, best_model_path)
        wandb.save(str(best_model_path))
    
    return history["best_consistency"]

def train_with_wandb():
    """Wrapper function to initialize wandb run and call the training function"""
    best_consistency = train_with_config()
    wandb.run.summary['best_consistency_score'] = best_consistency

def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run W&B hyperparameter sweep for Perceptual RBVAE')
    parser.add_argument('--sweep_id', type=str, help='W&B sweep ID to join an existing sweep')
    parser.add_argument('--create_sweep', action='store_true', help='Create a new sweep instead of joining an existing one')
    parser.add_argument('--project_name', type=str, default='percep-rbvae', help='W&B project name')
    args = parser.parse_args()
    
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'best_consistency_score',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'latent_dim': {
                'values': [25, 50, 75, 100]
            },
            'init_temperature': {
                'distribution': 'uniform',
                'min': 1.0,
                'max': 5.0
            },
            'final_temperature': {
                'distribution': 'uniform',
                'min': 0.2,
                'max': 0.5
            },
            'anneal_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'num_temp_updates': {
                'distribution': 'int_uniform',
                'min': 550,
                'max': 1100
            },
            'noise_ratio': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.2
            },
            'margin': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'alpha_contrast': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 5
            },
            'beta_kl': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 5
            },
            'num_epochs': {
                'value': 750
            },
            # Fixed parameters
            'bernoulli_p': {
                'value': 0.1
            },
            'embeddings_path': {
                'value': EMBEDDINGS_PATH
            },
            'last_frame': {
                'value': 479
            },
            'flags': {
                'value': [74, 206, 282, 389]
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

    # Command template to use:
    # python percep_RBVAE_wandb_sweep.py --create_sweep --project_name contrastive-RBVAE-perceps