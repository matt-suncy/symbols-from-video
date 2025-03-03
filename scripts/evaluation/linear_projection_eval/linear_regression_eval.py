"""
Performs linear regression between any number of independent and 
any number of dependent variables
"""
import sys
import os
from pathlib import Path
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from matplotlib import pyplot as plt 

from sklearn.datasets import make_regression  # For generating synthetic data
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)
# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from models.contrastive_RBVAE.contrastive_RBVAE_model import Seq2SeqBinaryVAE

# This is a CALLABLE
RESOLUTION = 256
ImageTransforms = T.Compose([
    T.Resize((RESOLUTION, RESOLUTION)),
    T.ToTensor()
])

def load_frames(frames_dir, frame_indices: tuple, transform=None):
    '''
    Loads frames into a list given the path to the frames and indices.
    '''
    images = []
    for frame_index in range(frame_indices[0], frame_indices[1]):
        filename = f"{frame_index:010d}.jpg"
        path = os.path.join(frames_dir, filename)
        image = Image.open(path).convert("RGB")
        if transform is not None:
            image = transform(image)
        images.append(image)

    return images

def save_tensor_as_image(tensor):
    tensor_reshaped = torch.squeeze(tensor).detach()
    transform = T.ToPILImage()
    image = transform(tensor_reshaped)
    save_path = Path(__file__).parent.joinpath("example_reconstruction.jpg")
    image = image.save(save_path)
    

def main():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Parameters for synthetic dataset:
    n_samples = 100       # Number of data points
    n_features = 5        # Number of independent variables (features)
    n_targets = 3         # Number of dependent variables (targets)

    # Generate a synthetic dataset for multi-output regression.
    # make_regression can generate multiple targets by setting n_targets.
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        noise=0.1,         # Adding a little noise
        random_state=42    # Ensures reproducibility
    )

    # Load model
    model_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "models/contrastive_RBVAE/saved_RBVAE_50_epochs"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rbvae_model = Seq2SeqBinaryVAE(in_channels=3, out_channels=3, latent_dim=32, hidden_dim=32)
    rbvae_model.load_state_dict(torch.load(model_path, weights_only=True))
    rbvae_model.to(device)
    rbvae_model.eval()
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model

    frames_dir = Path(__file__).parent.parent.parent.parent.joinpath(
        "videos/frames/kid_playing_with_blocks_1.mp4"
        )
    frame_indices = (0, 128)

    frames = load_frames(frames_dir, frame_indices, transform=ImageTransforms) # each element is a PIL

    # TODO get the embeddings and do the regression

    embeddings = []
    for i in range(len(frames)):
        # Give frames so that shape is [1, 1, 3, H, W]
        frame_expanded = frames[i][None, None, :, :, :].to(device)
        reconstruction, h_seq, bc_seq = rbvae_model(frame_expanded, temperature=0.5)
        save_tensor_as_image(reconstruction)
        # print(torch.count_nonzero(bc_seq))
        embeddings.append(torch.squeeze(h_seq))
    embeddings = torch.stack(embeddings, dim=0).to('cpu')

    # Flatten out frame tensors
    frames = torch.stack([torch.flatten(f) for f in frames], dim=0).to('cpu')

    # Split the dataset into training (80%) and testing (20%) sets.
    # FOR NOW, these train-test splits don't really mean anything because 
    # im not holding anything out.
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, frames, test_size=0.2, random_state=42
    )

    with torch.no_grad():
        # Initialize the Linear Regression model.
        model = LinearRegression()

        # Fit the model on the training data.
        model.fit(X_train, y_train)

        # Use the trained model to predict the targets for the test set.
        y_pred = model.predict(X_test)

    # Calculate regression metrics:

    # R-squared Score: Indicates how well the model explains the variability of the data.
    # For multi-output regression, 'uniform_average' computes the average R^2 over all targets.
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    # Mean Squared Error (MSE): The average squared difference between the actual and predicted values.
    mse = mean_squared_error(y_test, y_pred)

    # Mean Absolute Error (MAE): The average absolute difference between actual and predicted values.
    mae = mean_absolute_error(y_test, y_pred)

    # Explained Variance Score: Measures the proportion of variance explained by the model.
    evs = explained_variance_score(y_test, y_pred, multioutput='uniform_average')

    # Print the computed regression metrics.
    print("Regression Metrics:")
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")

    # Also print the model coefficients:
    # The intercept represents the constant term for each target.
    # The coefficients represent the weight of each independent variable for each target.
    print("\nModel Coefficients:")
    print(f"Intercepts: {model.intercept_}")
    print("Coefficients (each row corresponds to a target variable):")
    print(model.coef_)

if __name__ == '__main__':
    main()
