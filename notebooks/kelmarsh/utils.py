"""
Utility functions for graph creation, model training, and data scaling.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_adjacency_from_coords(coords: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Builds a row-stochastic adjacency matrix from coordinates using a k-NN graph.

    Args:
        coords: A numpy array of shape (N, 2) containing coordinates.
        k: The number of nearest neighbors to consider.

    Returns:
        A row-stochastic adjacency matrix of shape (N, N).
    """
    dist_mx = pairwise_distances(coords, coords)
    N = dist_mx.shape[0]

    # Create a k-NN mask
    adj = np.zeros_like(dist_mx)
    for i in range(N):
        # Get indices of k smallest distances (excluding self at index 0)
        nearest_neighbors = np.argsort(dist_mx[i, :])[1:k+1]
        adj[i, nearest_neighbors] = 1

    # Symmetrize the adjacency matrix
    adj = np.maximum(adj, adj.T)
    return adj


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalizes an adjacency matrix and adds self-loops.
    This is a key preprocessing step for GCNs.
    A_hat = D_tilde^(-0.5) * A_tilde * D_tilde^(-0.5)

    Args:
        adj: The adjacency matrix.

    Returns:
        The normalized adjacency matrix.
    """
    adj_tilde = adj + np.eye(adj.shape[0])
    d_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(adj_tilde, axis=1)))
    return d_tilde_inv_sqrt @ adj_tilde @ d_tilde_inv_sqrt


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                A_hat: torch.Tensor, config: dict, device: str) -> nn.Module:
    """
    Trains a PyTorch neural network model with early stopping.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        A_hat: The normalized adjacency matrix tensor.
        config: Dictionary containing training parameters (epochs, lr).
        device: The device to train on ('cpu' or 'cuda').

    Returns:
        The trained model with the best validation loss weights.
    """
    model.to(device)
    A_hat_t = A_hat.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience_limit = 10  # Stop training if val loss doesn't improve for 10 epochs

    for epoch in range(config['epochs']):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, A_hat_t)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, A_hat_t)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Copy state dict to CPU to save GPU memory
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load the best model state and return
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model