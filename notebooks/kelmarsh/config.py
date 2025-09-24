"""
Central configuration file for the forecasting pipeline.
"""
import torch
import logging

CONFIG = {
    "data": {
        "source_path": "short_kelmarsh.pkl",
        "zarr_path": "short_kelmarsh.zarr",
        "lookback": 48,
        "horizon": 6,
        "target_col_idx": 0
    },
    "run": {
        "seed": 133,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_level": logging.INFO,
        "num_workers": 4
    },
    "evaluation": {
        "test_start_date": "2023-01-01",
        "retrain_every_hours": 168, # Retrain weekly
    },
    "models": {
        "stgcn": {
            "wrapper": "NNModelWrapper",
            "class": "STGCN",
            "params": {"hidden_size": 128, "dropout": 0.2, "kernel_size": 7, "n_blocks": 3},
            "training": {"epochs": 75, "learning_rate": 1e-4, "batch_size": 128}
        },
        "lgbm_direct": {
            "wrapper": "DirectTreeWrapper",
            "params": {
                'objective': 'quantile',
                'metric': 'quantile',
                'n_estimators': 300,
                'max_depth': 7,
                'learning_rate': 0.05,
                'min_child_samples': 20,
                'verbose': -1,
                'n_jobs': -1,
            },
            "probabilistic": {
                "quantiles": [0.05, 0.5, 0.95] # Lower, Median, Upper
            },
            "training": {}
        },
        "arima": {
            "wrapper": "ARIMAModelWrapper",
            "class": "ARIMABaseline",
            "params": {"p_range": range(0, 5), "q_range": range(0, 5)},
            "training": { "rolling_window_size": 168 }
        }
    },
    "graph": {
        "k_neighbors": 2,
        "coords": {
            'T01': [52.4006, -0.94713, 145.598], 'T02': [52.40255, -0.94953, 156.577],
            'T03': [52.40383, -0.94419, 153.477], 'T04': [52.39878, -0.94115, 146.313],
            'T05': [52.40231, -0.94054, 142.901], 'T06': [52.40069, -0.93609, 135.039]
        }
    }
}

