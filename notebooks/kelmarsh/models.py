"""
Contains all model definitions and standardized wrappers with optimizations.
"""
import warnings
import logging
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader
import os

from data_handler import TimeSeriesDataset, create_tabular_features_for_window
from utils import train_model

# ===================================================================
# 1. PYTORCH MODEL DEFINITIONS (T-GCN and STGCN)
# ===================================================================

class GCNLayer(nn.Module):
    """Standard Graph Convolutional Layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.einsum("bnf,fo->bno", x, self.weight)
        output = torch.einsum("nm,bmo->bno", adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

class STConvBlock(nn.Module):
    """Spatio-temporal convolutional block with dilated convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size, dropout, dilation):
        super().__init__()
        self.t_conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),
                                 padding=(0, (kernel_size - 1) * dilation), dilation=(1, dilation))
        self.g_conv = GCNLayer(out_channels, out_channels)
        self.t_conv2 = nn.Conv2d(out_channels, out_channels, (1, kernel_size),
                                 padding=(0, (kernel_size - 1) * dilation), dilation=(1, dilation))
        self.layer_norm = nn.LayerNorm([out_channels])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_hat):
        # x shape: (B, C, N, T) where C is features/channels
        x_res = x
        time_len = x.size(3)

        x = torch.relu(self.t_conv1(x))
        x = x[..., :time_len]
        
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1) # -> (B, T, N, C)
        
        x_gcn_input = x.reshape(B * T, N, C)
        x_gcn_output = self.g_conv(x_gcn_input, A_hat)
        x = x_gcn_output.reshape(B, T, N, C)
        
        x = x.permute(0, 3, 2, 1) # -> (B, C, N, T)
        x = torch.relu(self.t_conv2(x))
        x = x[..., :time_len]
        
        x = self.layer_norm((x + x_res).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)

class STGCN(nn.Module):
    """The full Spatio-Temporal Graph Convolutional Network."""
    def __init__(self, n_features: int, n_turbines: int, horizon: int,
                 hidden_size: int = 64, dropout: float = 0.2, kernel_size: int = 3, n_blocks: int = 2):
        super().__init__()
        self.start_conv = nn.Conv2d(n_features, hidden_size, (1, 1))
        
        self.st_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2**i
            self.st_blocks.append(STConvBlock(hidden_size, hidden_size, kernel_size, dropout, dilation=dilation))
            
        # Project from hidden_size directly ---
        self.output_projection = nn.Linear(hidden_size, horizon)

    def forward(self, x, A_hat):
        # Input x: (B, T, N, F)
        x = x.permute(0, 3, 2, 1) # -> (B, F, N, T)
        x = self.start_conv(x)
        
        for block in self.st_blocks:
            x = block(x, A_hat)
        
        x = x[..., -1] # Take last time step -> (B, C, N)
        x = x.permute(0, 2, 1) # -> (B, N, C)
        
        output = self.output_projection(x) # -> (B, N, H)
        return output.transpose(1, 2) # -> (B, H, N)

# ===================================================================
# 2. ARIMA BASELINE DEFINITION
# ===================================================================
class ARIMABaseline:
    """A wrapper for a robust ARIMA baseline model for the aggregate farm power."""
    def __init__(self, p_range=range(0, 4), q_range=range(0, 3)):
        self.p_range = p_range
        self.q_range = q_range
        self.model_fit = None

    def _find_best_order(self, data: np.ndarray) -> tuple:
        if np.std(data) < 1e-6: return (0, 1, 0)
        best_aic, best_order_found = float("inf"), None
        warnings.filterwarnings("ignore")
        try:
            d = 1 if adfuller(data)[1] > 0.05 else 0
        except Exception:
            d = 1
        for p in self.p_range:
            for q in self.q_range:
                if p == 0 and q == 0: continue
                try:
                    res = ARIMA(data, order=(p, d, q)).fit()
                    if res.aic < best_aic:
                        best_aic, best_order_found = res.aic, (p, d, q)
                except Exception: continue
        warnings.filterwarnings("default")
        return best_order_found or (1, d, 0)

    def fit(self, data: np.ndarray, order: tuple):
        """Fits the ARIMA model with a pre-determined order."""
        if order is not None:
             logging.info(f"Fitting ARIMA with order: {order}")
        self.model_fit = ARIMA(data, order=order).fit()

    def predict(self, n_periods: int) -> np.ndarray:
        return self.model_fit.forecast(steps=n_periods)

# ===================================================================
# 3. UNIFIED MODEL WRAPPERS
# ===================================================================

class NNModelWrapper:
    """Wrapper for training and predicting with PyTorch sequence models."""
    def __init__(self, **kwargs):
        self.model_class = globals()[kwargs['class']]
        self.model_params = kwargs['params']
        self.training_params = kwargs['training']
        self.data_params = kwargs['data_params']
        self.run_params = kwargs['run_params']
        self.n_turbines = kwargs['n_turbines']
        self.device = kwargs['device']
        self.model = None
        # --- CRITICAL FIX: Use MinMaxScaler for non-negative data ---
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def train(self, seq_data: np.ndarray, train_indices: list, A_hat: torch.Tensor):
        logging.info("Training NN model...")
        
        lookback = self.data_params['lookback']
        n_features = seq_data.shape[2]
        target_idx = self.data_params['target_col_idx']
        
        train_window_data = seq_data[train_indices[0] : train_indices[-1] + lookback]
        # --- BUG FIX: Corrected reshape operation ---
        self.scaler_X.fit(train_window_data.reshape(-1, n_features))
        self.scaler_y.fit(train_window_data[:, :, target_idx].reshape(-1, 1))

        full_dataset = TimeSeriesDataset(
            seq_data, train_indices, self.data_params['lookback'], self.data_params['horizon'],
            self.scaler_X, self.scaler_y, target_idx
        )
        val_size = int(len(full_dataset) * 0.15)
        if val_size == 0 and len(full_dataset) > 0:
            val_size = 1
        
        if len(full_dataset) <= val_size:
            train_subset, val_subset = full_dataset, full_dataset
        else:
             train_subset, val_subset = torch.utils.data.random_split(
                full_dataset, [len(full_dataset) - val_size, val_size]
            )
        
        train_loader = DataLoader(
            train_subset, batch_size=self.training_params['batch_size'], shuffle=True,
            num_workers=self.run_params.get('num_workers', 0), pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.training_params['batch_size'] * 2,
            num_workers=self.run_params.get('num_workers', 0), pin_memory=True
        )

        self.model = self.model_class(
            n_features=n_features, n_turbines=self.n_turbines, horizon=self.data_params['horizon'], **self.model_params
        )
        self.model = train_model(self.model, train_loader, val_loader, A_hat, self.training_params, self.device)

    def predict(self, seq_data_window: np.ndarray, **kwargs) -> np.ndarray:
        A_hat = kwargs.get("A_hat")
        if A_hat is None: raise ValueError("A_hat (adjacency matrix) is required for GCN prediction.")
        self.model.eval()
        with torch.no_grad():
            T, N, F = seq_data_window.shape
            x_scaled = self.scaler_X.transform(seq_data_window.reshape(-1, F)).reshape(T, N, F)
            x_tensor = torch.from_numpy(x_scaled).float().unsqueeze(0).to(self.device)
            A_hat_t = A_hat.to(self.device)
            pred_scaled = self.model(x_tensor, A_hat_t).squeeze(0) # (H, N)
            pred_unscaled = self.scaler_y.inverse_transform(pred_scaled.cpu().numpy().reshape(-1,1))
            return pred_unscaled.reshape(self.data_params['horizon'], N)

class ARIMAModelWrapper:
    """Wrapper for the aggregate farm power ARIMA baseline."""
    def __init__(self, **kwargs):
        self.model_class = globals()[kwargs['class']]
        self.model_params = kwargs['params']
        # --- BUG FIX: Corrected typo in key access ---
        self.horizon = kwargs['data_params']['horizon']
        self.model = None
        self.best_order = None 

    def train(self, farm_power_series: np.ndarray, **kwargs):
        logging.info("Training ARIMA model on aggregate farm power...")
        if self.model is None:
            self.model = self.model_class(**self.model_params)

        if self.best_order is None:
            logging.info("Performing initial search for best ARIMA order...")
            self.best_order = self.model._find_best_order(farm_power_series)
        
        self.model.fit(farm_power_series, order=self.best_order)


    def predict(self, **kwargs) -> np.ndarray:
        preds = self.model.predict(n_periods=self.horizon)
        return np.maximum(0, preds)

class DirectTreeWrapper:
    """
    Wrapper for a tree-based model that supports probabilistic forecasting
    via quantile regression.
    """
    def __init__(self, **kwargs):
        self.params = kwargs['params']
        self.horizon = kwargs['data_params']['horizon']
        self.probabilistic_config = kwargs.get('probabilistic')
        self.models = []

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs):
        logging.info("Training direct LightGBM models...")
        self.models = []
        quantiles = self.probabilistic_config['quantiles'] if self.probabilistic_config else [0.5] # Default to median if not probabilistic

        for h in range(1, self.horizon + 1):
            horizon_step_models = {}
            for q in quantiles:
                logging.info(f"Training model for horizon step {h}, quantile {q}...")
                
                # Set the alpha parameter for quantile regression
                model_params = self.params.copy()
                if self.probabilistic_config:
                    model_params['alpha'] = q
                
                model = lgb.LGBMRegressor(**model_params)
                target_col = f'target_h{h}'
                model.fit(X_train, y_train[target_col])
                horizon_step_models[q] = model
            self.models.append(horizon_step_models)

    def predict(self, X_test: pd.DataFrame, **kwargs) -> np.ndarray:
        # The output shape will be (horizon, num_quantiles)
        quantiles = self.probabilistic_config['quantiles'] if self.probabilistic_config else [0.5]
        predictions = np.zeros((self.horizon, len(quantiles)))
        
        for h in range(self.horizon):
            for i, q in enumerate(quantiles):
                model = self.models[h][q]
                pred_q = model.predict(X_test)[0]
                predictions[h, i] = pred_q
        
        # If not probabilistic, return a 1D array for consistency
        if not self.probabilistic_config:
            return predictions.flatten()
            
        return np.maximum(0, predictions)

