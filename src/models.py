import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, List, Optional
from sklearn.base import clone
import warnings
import math


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions"""
        pass
    
    @abstractmethod
    def clone(self):
        """Create a clone of the model"""
        pass
    
    def update(self, X: pd.DataFrame, y: pd.Series):
        """Update model with new data (default: refit)"""
        self.fit(X, y)


class SklearnWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn compatible models"""
    
    def __init__(self, model, feature_cols: Optional[List[str]] = None):
        self.model = model
        self.feature_cols = feature_cols
        self.training_indices = set()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_features = self._prepare_features(X)
        self.model.fit(X_features, y)
        self.training_indices = set(X.index)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_features = self._prepare_features(X)
        return self.model.predict(X_features)

    def clone(self):
        return SklearnWrapper(clone(self.model), self.feature_cols)

    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_cols:
            return X[self.feature_cols].values
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        return X[numeric_cols].values
      

class ConformalMethod(ABC):
    """Abstract base class for conformal prediction methods"""
    
    def __init__(self, model: BaseModelWrapper, alpha: float):
        self.model = model
        self.alpha = alpha
    
    @abstractmethod
    def calibrate(self, calib_data: pd.DataFrame, target: str):
        """Calibrate using historical data"""
        pass
    
    @abstractmethod
    def predict_interval(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction intervals"""
        pass
    
    @abstractmethod
    def update(self, new_data: pd.DataFrame, target: str):
        """Update state with new observations"""
        pass

class OSSCP(ConformalMethod):
    """
    Online Sequential Split Conformal Prediction (OSSCP) (adapted from Lei et al., 2018)

    https://arxiv.org/abs/1604.04173
    """
    
    def __init__(self, model: BaseModelWrapper, alpha: float, max_calib_size: int = 1000):
        super().__init__(model, alpha)
        self.max_calib_size = max_calib_size
        self.residuals = []
        self.quantile = np.inf

    def calibrate(self, calib_data: pd.DataFrame, target: str):
        """Initialise calibration set with historical data"""
        X_calib = calib_data.drop(columns=[target])
        y_calib = calib_data[target]
        
        preds = self.model.predict(X_calib)
        residuals = np.abs(preds - y_calib.values)
        self.residuals = residuals.tolist()
        self._update_quantile()

    def predict_interval(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction intervals using current quantile"""
        preds = self.model.predict(X)
        return np.column_stack([preds - self.quantile, preds + self.quantile])

    def update(self, new_data: pd.DataFrame, target: str):
        """Update calibration set with new observations"""
        X_new = new_data.drop(columns=[target])
        y_new = new_data[target]
        
        # Process each new observation
        for i in range(len(new_data)):
            # Predict using current model
            pred = self.model.predict(X_new.iloc[[i]])[0]
            true_val = y_new.iloc[i]
            residual = np.abs(pred - true_val)
            
            # Update residuals with sliding window
            self.residuals.append(residual)
            if len(self.residuals) > self.max_calib_size:
                self.residuals.pop(0)
                
            # Update adaptive quantile
            self._update_quantile()

    def _update_quantile(self):
        """Quantile calculation per Lei et al. (2018)"""
        n = len(self.residuals)
        if n == 0:
            self.quantile = np.inf
            return
            
        # Calculate required quantile position
        k = math.ceil((n + 1) * (1 - self.alpha))
        k = min(k, n)  # Ensure k <= n
        k = max(k, 1)  # Ensure k >= 1
            
        # Get k-th largest residual
        sorted_residuals = np.sort(self.residuals)
        self.quantile = sorted_residuals[k-1]

class ACI(ConformalMethod):
    """
    Adaptive Conformal Inference (ACI) (Gibbs & Candès, 2021)

    https://arxiv.org/abs/2106.00170
    """
    
    def __init__(self, model: BaseModelWrapper, alpha: float, 
                 gamma: float = 0.01, window_size: int = 200):
        super().__init__(model, alpha)
        self.gamma = gamma
        self.window_size = window_size
        self.quantile = np.inf
        self.residuals = []

    def calibrate(self, calib_data: pd.DataFrame, target: str):
        X_calib = calib_data.drop(columns=[target])
        y_calib = calib_data[target]
        
        preds = self.model.predict(X_calib)
        self.residuals = np.abs(preds - y_calib.values).tolist()
        self._update_quantile()

    def predict_interval(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.column_stack([preds - self.quantile, preds + self.quantile])


    def update(self, new_data: pd.DataFrame, target: str):
        X_new = new_data.drop(columns=[target])
        y_new = new_data[target]
        
        # Get prediction using current model
        pred = self.model.predict(X_new)[0]
        actual = y_new.iloc[0]
        
        # Check coverage using CURRENT quantile (before updating)
        is_covered = (pred - self.quantile) <= actual <= (pred + self.quantile)
        
        # ACI update per Gibbs & Candès (2021)
        # err_t = 1 if NOT covered, 0 if covered
        err_t = int(not is_covered)
        
        # Update rule: α_{t+1} = α_t + γ(α - err_t)
        # Since we're updating quantile directly: quantile += γ(err_t - (1-α))
        error = err_t - (1 - self.alpha)
        self.quantile = max(0, self.quantile + self.gamma * error)
        
        # Update residuals for fallback
        residual = np.abs(pred - actual)
        self.residuals.append(residual)
        if len(self.residuals) > self.window_size:
            self.residuals.pop(0)

    def _update_quantile(self):
        """Standard quantile calculation"""
        if self.residuals:
            self.quantile = np.quantile(self.residuals, 1 - self.alpha)

            
class EnbPI(ConformalMethod):
    """
    Ensemble Batch Prediction Intervals (EnbPI) (Xu & Xie, 2021)
    
    https://arxiv.org/abs/2010.09107
    """
    
    def __init__(self, model: BaseModelWrapper, alpha: float, 
                 bootstrap_size: int = 20):
        super().__init__(model, alpha)
        self.bootstrap_size = bootstrap_size
        self.ensemble = []
        self.residuals = []
        self.bootstrap_indices = []
        
    def calibrate(self, calib_data: pd.DataFrame, target: str):
        # Initialise ensemble
        self.ensemble = [self.model.clone() for _ in range(self.bootstrap_size)]
        self.bootstrap_indices = []
        
        n_calib = len(calib_data)
        calib_idx_array = calib_data.index.values
        
        # Train on bootstrap samples
        for i, model in enumerate(self.ensemble):
            boot_positions = np.random.choice(n_calib, size=n_calib, replace=True)
            boot_indices = calib_idx_array[boot_positions]
            self.bootstrap_indices.append(set(boot_indices))
            
            boot_sample = calib_data.iloc[boot_positions]
            X_sample = boot_sample.drop(columns=[target])
            y_sample = boot_sample[target]
            model.fit(X_sample, y_sample)
        
        self.residuals = []
        for idx in calib_data.index:
            oob_preds = []
            for j, boot_indices in enumerate(self.bootstrap_indices):
                if idx not in boot_indices:  # Out-of-bag for this bootstrap
                    pred = self.ensemble[j].predict(calib_data.loc[[idx]].drop(columns=[target]))[0]
                    oob_preds.append(pred)
            
            if oob_preds:  # Only if we have OOB predictions
                oob_mean = np.mean(oob_preds)
                self.residuals.append(np.abs(oob_mean - calib_data.loc[idx, target]))

    def predict_interval(self, X: pd.DataFrame) -> np.ndarray:
        if not self.residuals:
            raise RuntimeError("Call calibrate() first")
            
        # Ensemble prediction
        preds_matrix = np.array([model.predict(X) for model in self.ensemble])
        mean_preds = np.mean(preds_matrix, axis=0)
        
        # Calculate interval using standard quantile
        quantile = np.quantile(self.residuals, 1 - self.alpha)
        return np.column_stack([mean_preds - quantile, mean_preds + quantile])

    def update(self, new_data: pd.DataFrame, target: str):
        X_new = new_data.drop(columns=[target])
        y_new = new_data[target]
        
        # Update each model
        for model in self.ensemble:
            model.update(X_new, y_new)
        
        # Update residuals with new observation
        ensemble_preds = np.array([model.predict(X_new)[0] for model in self.ensemble])
        mean_pred = np.mean(ensemble_preds)
        self.residuals.append(np.abs(mean_pred - y_new.iloc[0]))
        

class ConformalTimeSeries:
    def __init__(
        self,
        model: Union[Any, BaseModelWrapper],
        target: str,
        feature_cols: Optional[List[str]] = None,
        alpha: float = 0.1,
        method: str = 'oss',
        method_params: Optional[Dict] = None
    ):
        """
        Conformal prediction for time series forecasting
        """
        self.model = self._wrap_model(model, feature_cols)
        self.target = target
        self.alpha = alpha
        self.method = method
        self.method_params = method_params or {}
        self.conformal_method = self._init_conformal_method()

    def _wrap_model(self, model, feature_cols) -> BaseModelWrapper:
        """Wrap raw models into appropriate wrapper"""
        if isinstance(model, BaseModelWrapper):
            return model
        
        model_type = type(model).__module__
        if 'sklearn' in model_type:
            return SklearnWrapper(model, feature_cols)
        elif 'xgboost' in model_type:
            return XGBoostWrapper(model, feature_cols)
        else:
            warnings.warn("Unrecognised model type - using SklearnWrapper")
            return SklearnWrapper(model, feature_cols)

    def _init_conformal_method(self) -> ConformalMethod:
        """Initialise selected conformal method"""
        if self.method == 'oss':
            return OSSCP(
                model=self.model,
                alpha=self.alpha,
                max_calib_size=self.method_params.get('max_calib_size', 1000)
            )
        elif self.method == 'aci':
            return ACI(
                model=self.model,
                alpha=self.alpha,
                gamma=self.method_params.get('gamma', 0.01),
                window_size=self.method_params.get('window_size', 200)
            )
        elif self.method == 'enbpi':
            return EnbPI(
                model=self.model,
                alpha=self.alpha,
                bootstrap_size=self.method_params.get('bootstrap_size', 20)
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def fit(self, train_data: pd.DataFrame):
        """Train base model(s)"""
        X_train = train_data.drop(columns=[self.target])
        y_train = train_data[self.target]
        self.model.fit(X_train, y_train)

    def calibrate(self, calib_data: pd.DataFrame):
        """Calibrate conformal intervals"""
        self.conformal_method.calibrate(calib_data, self.target)

    def predict_interval(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction intervals"""
        return self.conformal_method.predict_interval(X)

    def update(self, new_data: pd.DataFrame):
        """Update state with new observations"""
        self.conformal_method.update(new_data, self.target)
