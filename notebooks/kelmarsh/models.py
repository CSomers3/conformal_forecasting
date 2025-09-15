"""
Contains all model definitions and standardized wrappers.

Each model type (NN, ARIMA, Quantile Regression) has a corresponding
wrapper class that provides a unified `train` and `predict` API.
This simplifies the main benchmarking script.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader, Subset

from data_handler import TimeSeriesDataset
from utils import train_model

# ===================================================================
# 1. PYTORCH MODEL DEFINITIONS (GCN-GRU)
# ===================================================================

class GCNLayer(nn.Module):
    """A single Graph Convolutional Network Layer."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.matmul(A_hat, x))


class GCN_GRU(nn.Module):
    """A hybrid model combining GCN and GRU with residual connections."""
    def __init__(self, n_features: int, n_turbines: int, horizon: int,
                 gcn_hidden: int = 32, gru_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_turbines = n_turbines
        self.horizon = horizon

        self.gcn = GCNLayer(n_features, gcn_hidden)
        self.layer_norm = nn.LayerNorm(gcn_hidden)
        self.residual_proj = nn.Linear(n_features, gcn_hidden)
        self.gru = nn.GRU(gcn_hidden, gru_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden, horizon)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        x_reshaped = x.view(B * T, N, F)
        gcn_out = self.gcn(x_reshaped, A_hat)
        res_in = self.residual_proj(x_reshaped)
        x_gcn_out = self.layer_norm(torch.relu(gcn_out) + res_in)
        x_for_gru_view = x_gcn_out.view(B, T, N, -1)
        x_for_gru = x_for_gru_view.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        gru_out, _ = self.gru(x_for_gru)
        last_h = self.dropout(gru_out[:, -1, :])
        out = self.fc(last_h).view(B, N, self.horizon)
        return out.permute(0, 2, 1)


# ===================================================================
# 2. ARIMA BASELINE DEFINITION
# ===================================================================

class ARIMABaseline:
    """A wrapper for a robust ARIMA baseline model."""
    def __init__(self, p_range=range(0, 4), q_range=range(0, 3)):
        self.p_range = p_range
        self.q_range = q_range
        self.best_order = None
        self.model_fit = None

    def _find_best_order(self, data: np.ndarray) -> tuple:
        if np.std(data) < 1e-6:
            return (0, 1, 0)

        best_aic, best_order_found = float("inf"), None
        warnings.filterwarnings("ignore")
        try:
            d = 1 if adfuller(data)[1] > 0.05 else 0
        except ValueError:
            d = 1

        for p in self.p_range:
            for q in self.q_range:
                if p == 0 and q == 0: continue
                try:
                    res = ARIMA(data, order=(p, d, q)).fit()
                    if res.aic < best_aic:
                        best_aic, best_order_found = res.aic, (p, d, q)
                except Exception:
                    continue
        warnings.filterwarnings("default")
        return best_order_found or (1, d, 0)

    def fit(self, data: np.ndarray):
        self.best_order = self._find_best_order(data)
        self.model_fit = ARIMA(data, order=self.best_order).fit()

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
        self.n_turbines = kwargs['n_turbines']
        self.device = kwargs['device']
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def train(self, seq_data: np.ndarray, train_indices: list, A_hat: np.ndarray):
        print("Training NN model...")

        num_samples = min(len(train_indices), 5000)
        lookback = self.data_params['lookback']
        n_turbines, n_features = seq_data.shape[1], seq_data.shape[2]
        sample_indices = np.random.choice(train_indices, num_samples, replace=False)

        sample_data_x = np.empty((num_samples * lookback * n_turbines, n_features), dtype=np.float32)
        sample_data_y = np.empty((num_samples * lookback * n_turbines, 1), dtype=np.float32)

        for i, idx in enumerate(sample_indices):
            start_row = i * lookback * n_turbines
            end_row = start_row + lookback * n_turbines
            window = seq_data[idx : idx + lookback]
            sample_data_x[start_row:end_row] = window.reshape(-1, n_features)
            sample_data_y[start_row:end_row] = window[:, :, 0].reshape(-1, 1)

        self.scaler_X.fit(sample_data_x)
        self.scaler_y.fit(sample_data_y)

        full_dataset = TimeSeriesDataset(
            seq_data, train_indices, self.data_params['lookback'], self.data_params['horizon'],
            self.scaler_X, self.scaler_y
        )
        val_size = int(len(full_dataset) * 0.15)
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - val_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=self.training_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.training_params['batch_size'] * 2)

        self.model = self.model_class(
            n_features=n_features, n_turbines=self.n_turbines, horizon=self.data_params['horizon'], **self.model_params
        )
        A_hat_t = torch.from_numpy(A_hat).float()
        self.model = train_model(self.model, train_loader, val_loader, A_hat_t, self.training_params, self.device)

    def predict(self, seq_data_window: np.ndarray, **kwargs) -> np.ndarray:
        # **FIX**: This now requires A_hat to be passed via kwargs to use the graph structure.
        A_hat = kwargs.get("A_hat")
        if A_hat is None:
            raise ValueError("A_hat (adjacency matrix) is required for GCN model prediction.")

        self.model.eval()
        with torch.no_grad():
            T, N, F = seq_data_window.shape
            x_scaled = self.scaler_X.transform(seq_data_window.reshape(-1, F)).reshape(T, N, F)
            x_tensor = torch.from_numpy(x_scaled).float().unsqueeze(0).to(self.device)
            A_hat_t = torch.from_numpy(A_hat).float().to(self.device)

            # **FIX**: Use the actual A_hat_t, not an identity matrix.
            pred_scaled = self.model(x_tensor, A_hat_t)
            pred_scaled_flat = pred_scaled.squeeze(0).permute(1,0).cpu().numpy().reshape(-1, 1)
            pred_unscaled = self.scaler_y.inverse_transform(pred_scaled_flat)
            return pred_unscaled.reshape(N, self.data_params['horizon']).T


class ARIMAModelWrapper:
    """Wrapper for the turbine-wise ARIMA baseline."""
    def __init__(self, **kwargs):
        self.model_class = globals()[kwargs['class']]
        self.model_params = kwargs['params']
        self.horizon = kwargs['data_params']['horizon']
        self.data_params = kwargs['data_params']
        self.models = []

    def train(self, seq_data: np.ndarray, train_indices: list, **kwargs):
        print("Training ARIMA models for each turbine...")
        self.models = []
        n_turbines = seq_data.shape[1]
        start_idx = train_indices[0]
        end_idx = train_indices[-1] + self.data_params['lookback']
        train_data = seq_data[start_idx:end_idx]

        for i in range(n_turbines):
            print(f"  Fitting ARIMA for turbine {i+1}/{n_turbines}...")
            turbine_series = train_data[:, i, 0]
            model = self.model_class(**self.model_params)
            model.fit(turbine_series)
            self.models.append(model)

    def predict(self, seq_data_window: np.ndarray, **kwargs) -> np.ndarray:
        n_turbines = seq_data_window.shape[1]
        preds = np.zeros((self.horizon, n_turbines))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(n_periods=self.horizon)
        return preds


class QuantileRegressionWrapper:
    """Wrapper for LightGBM Quantile Regression."""
    def __init__(self, **kwargs):
        self.params = kwargs['params']
        self.quantiles = kwargs['quantiles']
        self.horizon = kwargs['data_params']['horizon']
        self.models = {}
        self.feature_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs):
        print("Training LightGBM Quantile Regression models...")
        self.feature_names = list(X_train.columns)
        for h in range(self.horizon):
            print(f"  Fitting models for horizon h={h+1}...")
            y_h = y_train.iloc[:, h]
            for q in self.quantiles:
                lgb_params = self.params.copy()
                lgb_params['alpha'] = q
                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(X_train, y_h)
                self.models[(q, h)] = model

    def predict(self, X_test: pd.DataFrame, **kwargs) -> dict:
        predictions = {}
        X_test_ordered = X_test[self.feature_names]
        for q in self.quantiles:
            q_preds_horizon = np.zeros((len(X_test_ordered), self.horizon))
            for h in range(self.horizon):
                model = self.models[(q, h)]
                q_preds_horizon[:, h] = model.predict(X_test_ordered)
            predictions[q] = np.maximum(0, q_preds_horizon) # Power cannot be negative
        return predictions