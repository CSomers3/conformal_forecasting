"""
Thank you to Simon Leszek's GitHub: https://github.com/sltzgs/OpenWindSCADA/
From which these utility functions are heavily inspired
"""

import pandas as pd
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def load_kelmarsh_data(base_path):
    """
    Load Kelmash SCADA and log data
    """
    print("Loading SCADA data & Logs...")
    
    dct_scada = {}
    dct_logs = {}
    for trb_id in range(1, 7):
        # Load SCADA data
        scada_files = glob.glob(f"{base_path}*/Turbine_Data_Kelmarsh_{trb_id}_*.csv")
        if scada_files:
            df = pd.concat([
                pd.read_csv(f, skiprows=9, low_memory=False, index_col='# Date and time')
                for f in sorted(scada_files)
            ])
            
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.names = ['timestamp']
            df = df[df['Data Availability'] == 1].dropna(axis=1, how='all')
            
            dct_scada[f'T0{trb_id}'] = df
        
        # Load status logs
        log_files = glob.glob(f"{base_path}*/Status_Kelmarsh_{trb_id}_*.csv")
        if log_files:
            df_logs = pd.concat([
                pd.read_csv(f, skiprows=9, low_memory=False, index_col='Timestamp start')
                for f in sorted(log_files)
            ])
            
            df_logs.index = pd.to_datetime(df_logs.index, utc=True)
            df_logs = df_logs.dropna(axis=1, how='all')
            
            dct_logs[f'T0{trb_id}'] = df_logs
    
    return dct_scada, dct_logs


def set_style():
    """Simple, clean plot style"""
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })

def prepare_power_curve(path_):
    """Prepare power curve from CSV"""
    df_powercurve = pd.read_csv(path_, index_col=0)
    df_pc = pd.DataFrame(
        index=[np.round(i, 2) for i in np.arange(0, 30, 0.01)], 
        columns=['power_norm']
    )
    df_pc[df_pc.index < df_powercurve.index[0]] = 0
    df_pc[df_pc.index > df_powercurve.index[-1]] = 0
    df_pc.loc[df_powercurve.index] = df_powercurve.values.reshape(-1, 1)
    return df_pc.astype(float).interpolate(method='linear')

def plot_turbine_power_curve(df, turbine_id, powercurve_path='powercurve.csv'):
    """
    Plot single turbine power curve - simple inputs only
    
    Args:
        df: DataFrame with 'Wind speed (m/s)', 'Wind direction (°)', 'Power (kW)'
        turbine_id: String ID for turbine (e.g. 'T01')
        powercurve_path: Path to power curve CSV
    """
    set_style()
    
    # Prepare data
    df_pc = prepare_power_curve(powercurve_path)
    df_pc = df_pc / 2000 * df['Power (kW)'].max()
    
    ws = df['Wind speed (m/s)'].values
    p = df['Power (kW)'].values
    
    # Calculate expected power
    rounded_ws = np.round(ws, 2)
    valid = np.isin(rounded_ws, df_pc.index)
    expected_p = np.full_like(ws, np.nan)
    expected_p[valid] = df_pc.loc[rounded_ws[valid]].values.flatten()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Residuals
    residuals = p - expected_p
    norm = Normalize(vmin=-500, vmax=500)
    
    # Scatter plot
    scatter = ax.scatter(ws, p, c=residuals, cmap='RdYlBu_r', norm=norm, 
                        alpha=0.6, s=10, edgecolors='none')
    
    # Power curve line
    ax.plot(df_pc.index, df_pc['power_norm'], 'k-', linewidth=2, label='Reference Curve')
    
    # Styling
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Turbine {turbine_id} Power Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Power Residual (kW)')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_multiple_turbines(turbine_data_dict, powercurve_path='powercurve.csv'):
    """
    Plot multiple turbines in grid layout
    
    Args:
        turbine_data_dict: Dict like {'T01': df1, 'T02': df2, ...}
        powercurve_path: Path to power curve CSV
    """
    set_style()
    
    n_turbines = len(turbine_data_dict)
    cols = 3
    rows = (n_turbines + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_turbines == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    # Prepare power curve once
    df_pc_base = prepare_power_curve(powercurve_path)
    
    for i, (turbine_id, df) in enumerate(turbine_data_dict.items()):
        ax = axes_flat[i]
        
        # Scale power curve to this turbine
        df_pc = df_pc_base / 2000 * df['Power (kW)'].max()
        
        ws = df['Wind speed (m/s)'].values
        p = df['Power (kW)'].values
        
        # Calculate expected power
        rounded_ws = np.round(ws, 2)
        valid = np.isin(rounded_ws, df_pc_base.index)
        expected_p = np.full_like(ws, np.nan)
        expected_p[valid] = df_pc.loc[rounded_ws[valid]].values.flatten()
        
        # Plot
        residuals = p - expected_p
        norm = Normalize(vmin=-500, vmax=500)
        
        scatter = ax.scatter(ws, p, c=residuals, cmap='RdYlBu_r', norm=norm, 
                            alpha=0.6, s=8, edgecolors='none')
        ax.plot(df_pc.index, df_pc['power_norm'], 'k-', linewidth=2)
        
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'Turbine {turbine_id}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_turbines, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Single colorbar for all
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.8)
    cbar.set_label('Power Residual (kW)')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes
