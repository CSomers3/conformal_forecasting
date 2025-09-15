"""
Data loading, caching, and preprocessing module.

Handles efficient loading of time series data using a Zarr cache,
creation of tabular features for tree-based models, and a custom
PyTorch Dataset for sequence models.
"""
import os
import numpy as np
import pandas as pd
import torch
import zarr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def load_or_create_zarr_cache(source_path: str, zarr_path: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Loads raw sequence data from a Zarr cache or creates it from a pickle file.
    """
    if os.path.exists(zarr_path):
        try:
            data_store = zarr.open(zarr_path, mode='r')
            print(f"Loaded sequence data from Zarr cache: {zarr_path}")
            all_data = data_store['data'][:]
            timestamps = pd.to_datetime(data_store['timestamps'][:], utc=True)
            return all_data, timestamps
        except KeyError:
            print("Zarr cache is incomplete. Recreating from source pickle...")

    print("Zarr cache not found or invalid. Creating from source pickle...")
    data_dict = pd.read_pickle(source_path)
    turbines = sorted(list(data_dict.keys()))
    all_data = np.stack([data_dict[t].values for t in turbines], axis=1).astype(np.float32)

    if all_data.ndim == 2:
        all_data = np.expand_dims(all_data, axis=2)

    timestamps = data_dict[turbines[0]].index
    if timestamps.tzinfo is None:
        timestamps = timestamps.tz_localize('UTC')

    root = zarr.open(zarr_path, mode='w')
    
    # **FIX**: Replace `None` in chunks with the actual dimension sizes from the data's shape.
    # This is a robust fix for the zarr TypeError.
    chunks = (1000, all_data.shape[1], all_data.shape[2])
    root.create_dataset('data', shape=all_data.shape, chunks=chunks, dtype=all_data.dtype)
    root['data'][:] = all_data
    
    ts_data = timestamps.values.astype(str)
    root.create_dataset('timestamps', shape=ts_data.shape, dtype=ts_data.dtype)
    root['timestamps'][:] = ts_data

    print(f"Saved sequence data to Zarr cache: {zarr_path}")
    return all_data, timestamps


def create_tabular_features(all_data: np.ndarray, timestamps: pd.DatetimeIndex, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a full tabular feature set using efficient, vectorized pandas operations.
    """
    print("Creating tabular features for tree-based models...")
    farm_power = all_data[:, :, 0].sum(axis=1)
    df = pd.DataFrame({'farm_power': farm_power}, index=timestamps)

    # Lag features
    for lag in [6, 12, 24, 48]:
        df[f'lag_{lag}'] = df['farm_power'].shift(lag)

    # Rolling window features
    for window in [6, 12, 24]:
        df[f'rolling_mean_{window}'] = df['farm_power'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df['farm_power'].shift(1).rolling(window).std()

    # Calendar features (cyclical)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)

    # Target variables (y)
    for h in range(1, horizon + 1):
        df[f'target_h{h}'] = df['farm_power'].shift(-h)

    df = df.dropna()

    target_cols = [f'target_h{h}' for h in range(1, horizon + 1)]
    feature_cols = [col for col in df.columns if col not in target_cols and col != 'farm_power']

    X_tab = df[feature_cols]
    y_tab = df[target_cols]

    print(f"Tabular feature creation complete. Shape: {X_tab.shape}")
    return X_tab, y_tab


class TimeSeriesDataset(Dataset):
    """Memory-efficient dataset for sequence-based models (NNs)."""
    def __init__(self, data: np.ndarray, indices: list, lookback: int, horizon: int, scaler_X=None, scaler_y=None):
        self.data = data
        self.indices = indices
        self.lookback = lookback
        self.horizon = horizon
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.indices[i]
        x_end_idx = start_idx + self.lookback
        y_end_idx = x_end_idx + self.horizon

        x = self.data[start_idx:x_end_idx, :, :]
        y = self.data[x_end_idx:y_end_idx, :, 0]

        if self.scaler_X:
            T, N, F = x.shape
            x = self.scaler_X.transform(x.reshape(-1, F)).reshape(T, N, F)
        if self.scaler_y:
            H, N = y.shape
            y = self.scaler_y.transform(y.reshape(-1, 1)).reshape(H, N)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()