"""
Utility functions for graph creation, model training, and data scaling.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def build_adjacency_from_coords(coords: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Builds a symmetric adjacency matrix from coordinates using a k-NN graph.
    """
    dist_mx = pairwise_distances(coords, coords)
    adj = np.zeros_like(dist_mx)
    for i in range(dist_mx.shape[0]):
        nearest_neighbors = np.argsort(dist_mx[i, :])[:k+1]
        adj[i, nearest_neighbors] = 1
    # Ensure symmetry
    return np.maximum(adj, adj.T)


def normalize_adj_torch(adj: np.ndarray) -> torch.Tensor:
    """
    Symmetrically normalizes an adjacency matrix and converts to a dense PyTorch tensor.
    A_hat = D_tilde^(-0.5) * A_tilde * D_tilde^(-0.5)
    """
    adj_tilde = adj + np.eye(adj.shape[0])
    d_tilde = np.sum(adj_tilde, axis=1)
    d_tilde_inv_sqrt = np.power(d_tilde, -0.5)
    d_tilde_inv_sqrt[np.isinf(d_tilde_inv_sqrt)] = 0.
    d_matrix_inv_sqrt = np.diag(d_tilde_inv_sqrt)
    adj_normalized = adj_tilde @ d_matrix_inv_sqrt.T @ d_matrix_inv_sqrt
    return torch.from_numpy(adj_normalized).float()


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                A_hat: torch.Tensor, config: dict, device: str) -> nn.Module:
    """
    Trains a PyTorch neural network model with early stopping.
    """
    model.to(device)
    A_hat_t = A_hat.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience_limit = 10

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb, A_hat_t)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = model(xb, A_hat_t)
                val_loss += criterion(pred, yb).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {epoch_loss/len(train_loader):.5f}, Val Loss: {val_loss:.5f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model.to(device)
