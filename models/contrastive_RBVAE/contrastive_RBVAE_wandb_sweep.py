import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import argparse
import os
import torch.nn.functional as F

# Import the model and training utilities
from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE
from models.contrastive_RBVAE.contrastive_RBVAE_train import (
    train_one_epoch, 
    ShuffledStatePairDataset, 
    ImageTransforms
)

def evaluate_model(model, device, dataloader, batch_size, epoch, 
                  init_temperature=1.0, bernoulli_p=0.5, margin=1.0, 
                  alpha_triplet=0.1, beta_kl=0.1):
    """
    Evaluates the model on validation data and returns losses.
    Args are the same as train_one_epoch() but without optimizer and annealing params.
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_triplet_loss = 0.0
    
    num_batches = len(dataloader)
    temperature = init_temperature  # Use constant temperature for evaluation
    
    with torch.no_grad():
        for batch_idx, item in enumerate(dataloader):
            # item shape: [B, 2, T, C, H, W]
            num_batches_item, _, num_states, _, _, _ = item.size()
            item = item.to(device)
            frames = [item[:, i] for i in range(2)]
            recon_losses = []
            kl_losses = []
            bc_seqs = []

            for frame in frames:
                x_recon, h_seq, bc_seq = model(frame, temperature=temperature, hard=False)
                recon_losses.append(F.mse_loss(x_recon, frame))  # Use MSE directly for validation
                kl_losses.append(torch.mean(torch.sum(bc_seq * torch.log(bc_seq / bernoulli_p + 1e-8) + 
                                        (1 - bc_seq) * torch.log((1 - bc_seq) / (1 - bernoulli_p) + 1e-8), dim=-1)))
                bc_seqs.append(bc_seq)

            recon_loss_val = sum(recon_losses) / len(recon_losses)
            kl_loss_val = sum(kl_losses) / len(kl_losses)
            
            triplet_loss_val = 0
            # Loop over states, excluding the last state
            for state_index in range(num_states - 1):
                # Define the triplet:
                anchor = bc_seqs[0][:, state_index]    # From the first frame at current state
                positive = bc_seqs[1][:, state_index]    # From the second frame (same state)
                negative = bc_seqs[0][:, state_index + 1]  # From the first frame of the next state
                
                # Compute distances
                dist_ap = torch.sqrt(torch.sum((anchor - positive)**2, dim=1))
                dist_an = torch.sqrt(torch.sum((anchor - negative)**2, dim=1))
                triplet_loss_batch = torch.clamp(dist_ap - dist_an + margin, min=0.0)
                triplet_loss_val += triplet_loss_batch.mean()
                
            triplet_loss_val /= float(num_states - 1)
            
            # Calculate the total loss
            total_loss_val = recon_loss_val + beta_kl * kl_loss_val + alpha_triplet * triplet_loss_val
            
            total_loss += total_loss_val.item()
            total_recon_loss += recon_loss_val.item()
            total_kl_loss += kl_loss_val.item()
            total_triplet_loss += triplet_loss_val.item()
    
    avg_total_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_triplet_loss = total_triplet_loss / num_batches
    
    return avg_total_loss, avg_recon_loss, avg_kl_loss, avg_triplet_loss

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
        hidden_dim=config.latent_dim
    ).to(device)
    
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Setup dataset and dataloader
    dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Set up validation dataset and dataloader
    val_dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms, mode="val")
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    # Logic for determining num_steps_to_update
    max_iters = config.num_epochs * len(dataset)
    num_temp_updates = config.num_temp_updates
    num_steps_to_update = int(max_iters / num_temp_updates)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Train for one epoch
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
            num_steps_to_update=num_steps_to_update, # This calculated from the config
            bernoulli_p=config.bernoulli_p,
            margin=config.margin,
            alpha_triplet=config.alpha_triplet,
            beta_kl=config.beta_kl
        )
        
        # Evaluate on validation set
        val_total_loss, val_recon_loss, val_kl_loss, val_triplet_loss = evaluate_model(
            model=model,
            device=device,
            dataloader=val_dataloader,
            batch_size=config.batch_size,
            epoch=epoch,
            init_temperature=config.init_temperature,
            bernoulli_p=config.bernoulli_p,
            margin=config.margin,
            alpha_triplet=config.alpha_triplet,
            beta_kl=config.beta_kl
        )
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train/total_loss": avg_total_loss,
            "train/reconstruction_loss": avg_recon_loss,
            "train/kl_divergence": avg_kl_loss,
            "train/triplet_loss": avg_triplet_loss,
            "val/total_loss": val_total_loss,
            "val/reconstruction_loss": val_recon_loss,
            "val/kl_divergence": val_kl_loss,
            "val/triplet_loss": val_triplet_loss
        })
        
        # Save the best model based on validation loss
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_total_loss,
                'val_loss': best_val_loss,
            }, os.path.join(wandb.run.dir, 'best_model.pt'))
            wandb.save('best_model.pt')
            
            # Log the best epoch
            wandb.run.summary['best_epoch'] = epoch
    
    # Return the best validation loss as the metric to optimize
    return best_val_loss

def train_with_wandb():
    """Wrapper function to initialize wandb run and call the training function"""
    # Train the model and get the best validation loss
    best_val_loss = train_with_config()
    
    # Log final metrics
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
        'method': 'bayes',  # Can be 'random', 'grid', or 'bayes'
        'metric': {
            'name': 'val/total_loss',  # Updated to use validation loss
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
                'distribution': 'uniform',
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