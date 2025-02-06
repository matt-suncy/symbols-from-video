"""
Performs linear regression between any number of independent and 
any number of dependent variables
"""
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

from models.contrastive_RBVAE.contr astive_RBVAE_model import Seq2SeqBinaryVAE


from sklearn.datasets import make_regression  # For generating synthetic data
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

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
    model_path = Path(__file__).parent.joinpath("saved_RBVAE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model

    # Split the dataset into training (80%) and testing (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the Linear Regression model.
    model = LinearRegression()

    # Fit the model on the training data.
    model.fit(X_train, y_train)

    # Use the trained model to predict the targets for the test set.
    y_pred = model.predict(X_test)

    # Calculate regression metrics:

    # R-squared Score: Indicates how well the model explains the variability of the data.
    # For multi-output regression, 'uniform_average' computes the average R² over all targets.
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
