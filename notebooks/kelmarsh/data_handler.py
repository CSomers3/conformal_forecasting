"""
Data loading, caching, and preprocessing module.
"""
import os
import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def load_or_create_zarr_cache(source_path: str, zarr_path: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Loads raw sequence data from a Zarr cache or creates it from a pickle file.
    This version is robust to inconsistent timestamps across turbines.
    """
    if os.path.exists(zarr_path):
        try:
            data_store = zarr.open(zarr_path, mode='r')
            return data_store['data'][:], pd.to_datetime(data_store['timestamps'][:], utc=True)
        except (KeyError, zarr.errors.PathNotFoundError):
            pass # Cache is invalid, proceed to recreate

    data_dict = pd.read_pickle(source_path)
    turbines = sorted(list(data_dict.keys()))

    # --- ROBUST TIMESTAMP CREATION ---
    master_index = pd.DatetimeIndex([], tz='UTC')
    for t in turbines:
        turbine_index = data_dict[t].index
        if turbine_index.tz is None:
            turbine_index = turbine_index.tz_localize('UTC')
        master_index = master_index.union(turbine_index)

    master_index = master_index.sort_values()

    # 2. Re-index and interpolate each turbine's data to the master index
    processed_dfs = []
    for t in turbines:
        turbine_df = data_dict[t]
        if turbine_df.index.tz is None:
            turbine_df.index = turbine_df.index.tz_localize('UTC')

        # Reindex to align data and fill gaps, then interpolate
        reindexed_df = turbine_df.reindex(master_index)
        interpolated_df = reindexed_df.interpolate(method='time', limit_direction='both')
        processed_dfs.append(interpolated_df)

    # 3. Stack the aligned data
    all_data = np.stack([df.values for df in processed_dfs], axis=1).astype(np.float32)

    # --- CACHE TO ZARR ---
    root = zarr.open(zarr_path, mode='w')
    chunks = (1000, all_data.shape[1], all_data.shape[2])
    root.create_dataset('data', data=all_data, chunks=chunks)
    root.create_dataset('timestamps', data=master_index.values.astype(str))

    return all_data, master_index

def create_tabular_features_for_window(data_seq: np.ndarray, timestamps: pd.DatetimeIndex, horizon: int, target_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a tabular feature set for a given window of data."""
    # Note: Assuming covariates are already in data_seq. This function creates time-based features.
    farm_power = data_seq[:, :, target_idx].sum(axis=1)
    df = pd.DataFrame({'farm_power': farm_power}, index=timestamps)
    
    # Lag features
    for lag in [6, 12, 24, 48, 72]:
        df[f'lag_{lag}'] = df['farm_power'].shift(lag)
    
    # Rolling window features
    for window in [6, 12, 24, 48]:
        df[f'rolling_mean_{window}'] = df['farm_power'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df['farm_power'].shift(1).rolling(window).std()

    # Time-based features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    # Target variables
    for h in range(1, horizon + 1):
        df[f'target_h{h}'] = df['farm_power'].shift(-h)
        
    df = df.dropna()
    target_cols = [f'target_h{h}' for h in range(1, horizon + 1)]
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    return df[feature_cols], df[target_cols]

class TimeSeriesDataset(Dataset):
    """Memory-efficient dataset for sequence-based models (NNs)."""
    def __init__(self, data: np.ndarray, indices: list, lookback: int, horizon: int,
                 scaler_X, scaler_y, target_idx: int):
        self.data = data
        self.indices = indices
        self.lookback = lookback
        self.horizon = horizon
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.target_idx = target_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.indices[i]
        x_end_idx = start_idx + self.lookback
        y_end_idx = x_end_idx + self.horizon

        x_raw = self.data[start_idx:x_end_idx]
        y_raw = self.data[x_end_idx:y_end_idx, :, self.target_idx]

        # Scale features
        T, N, F = x_raw.shape
        x_scaled = self.scaler_X.transform(x_raw.reshape(-1, F)).reshape(T, N, F)

        # Scale targets
        H, N_y = y_raw.shape
        y_scaled = self.scaler_y.transform(y_raw.reshape(-1, 1)).reshape(H, N_y)

        return torch.from_numpy(x_scaled).float(), torch.from_numpy(y_scaled).float()
