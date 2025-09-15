"""
Central configuration file for the forecasting pipeline.

All parameters related to data, evaluation, models, and graph structure
are defined here.
"""
import torch

CONFIG = {
    "data": {
        "source_path": "data_masked.pkl",
        "zarr_path": "data_cache.zarr",
        "lookback": 48,  # Use 48 hours of past data to make a forecast
        "horizon": 6,     # Forecast 6 hours into the future
    },
    "run": {
        "seed": 133,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "evaluation": {
        "test_start_date": "2022-06-01",
        # Retrain models every 7 days (168 hours) in the test period.
        "retrain_every_hours": 168,
    },
    "models": {
        "gcn_gru": {
            "wrapper": "NNModelWrapper",
            "class": "GCN_GRU",
            "params": {"gcn_hidden": 32, "gru_hidden": 64},
            "training": {
                "epochs": 100, "learning_rate": 1e-3, "batch_size": 64,
            }
        },
        "lgbm_qr": {
            "wrapper": "QuantileRegressionWrapper",
            "params": {
                'objective': 'quantile', 'metric': 'quantile',
                'n_estimators': 300, 'max_depth': 7,
                'learning_rate': 0.05, 'min_child_samples': 20,
                'verbose': -1, 'n_jobs': -1,
            },
            "quantiles": [0.05, 0.5, 0.95],
            "training": {}
        },
        "arima": {
            "wrapper": "ARIMAModelWrapper",
            "class": "ARIMABaseline",
            "params": {"p_range": range(0, 6), "q_range": range(0, 3)},
            "training": {
                "rolling_window_size": 168, # 1 week
            }
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