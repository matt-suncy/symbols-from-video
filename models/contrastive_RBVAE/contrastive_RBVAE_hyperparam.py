import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

# Import your model and training utilities. Adjust the import paths as needed.
from contrastive_RBVAE_model import Seq2SeqBinaryVAE
from contrastive_RBVAE_train import train_one_epoch, ShuffledStatePairDataset, ImageTransforms

def objective(trial):
    # Sample hyperparameters with Optuna
    anneal_rate = trial.suggest_loguniform('anneal_rate', 1e-5, 1e-3)
    beta_kl = trial.suggest_float('beta_kl', 0.05, 0.5)
    alpha_contrast = trial.suggest_float('alpha_contrast', 0.05, 0.5)
    margin = trial.suggest_float('margin', 0.1, 2.0)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    
    # Set up device, model, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup your dataset and dataloader.
    # Make sure to set the correct paths and state_segments for your data.
    frames_dir = Path("path/to/frames")  # Adjust accordingly
    state_segments = [(0, 100), (100, 200)]  # Example segments; update based on your data
    dataset = ShuffledStatePairDataset(frames_dir, state_segments, transform=ImageTransforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Use a small number of epochs for quick evaluation.
    num_epochs = 3
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        avg_total_loss, _, _, _ = train_one_epoch(
            model, device, dataloader, optimizer, batch_size=4, epoch=epoch,
            init_temperature=1.0, final_temperature=0.5,
            anneal_rate=anneal_rate, num_steps_to_update=50,      # Adjust num_steps_to_update as needed
            alpha_contrast=alpha_contrast, margin=margin, beta_kl=beta_kl, bernoulli_p=0.5
        )
        # For demonstration, we use the training loss as a proxy for validation loss.
        # In a real setup, use a separate validation dataset.
        best_val_loss = min(best_val_loss, avg_total_loss)
    
    return best_val_loss

if __name__ == "__main__":
    # Create and run the study.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Loss: {}".format(trial.value))
    print("  Parameters: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))