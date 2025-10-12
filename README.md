# Adaptive Conformal Prediction for Wind Farm Forecasting

**Turn any wind power forecaster into probabilistic intervals with theoretical coverage guarantees—even under regime changes.**

This repository implements adaptive conformal inference (ACI) methods for distribution-free uncertainty quantification in wind energy forecasting. Developed as an MSc Statistics thesis at UCL, it demonstrates how online conformal prediction can maintain reliable coverage under temporal dependence, distributional shift, and spatial correlation—without retraining the underlying forecaster.

## Why This Matters

Grid operators need accurate probabilistic forecasts to manage renewable variability. Traditional uncertainty methods (Quantile Regression, Bayesian NNs) either make unrealistic distributional assumptions or are computationally prohibitive for real-time deployment. Conformal prediction offers an alternative: distribution-free, finite-sample valid intervals that adapt on the fly.

## Key Results

- **Valid coverage under extreme conditions**: ACI methods maintain ~90% empirical coverage under strong temporal dependence ($\rho$=0.95), abrupt regime shifts, and complex spatial correlations—where standard methods fail.
- **Regime-shift robustness**: DtACI provides 15-20% sharper intervals than fixed-$\gamma$ ACI when distributions shift abruptly (common in wind).
- **Real-world validation**: STGCN point forecasts + ACI on Kelmarsh Wind Farm achieve lower Winkler Scores than OSSCP baseline and significantly outperform uncalibrated Quantile Regression (69.5% miscoverage).

## Structure

```
conformal_forecasting/
├── src/                           # Core implementations
│   ├── ACI.py, agACI.py, dtACI.py # Conformal methods
├── notebooks/
│   ├── synthetic/                 # Simulation studies (AR, MA, ARMA, regime-switching)
│   │   └── simulations.ipynb
│   └── kelmarsh/                  # Real wind farm case study
│       ├── data_prep.ipynb
│       ├── point_forecast_main.ipynb
│       └── conformal_main.ipynb
└── data/
    └── kelmarsh/                  # Open-source SCADA + NWP forecasts (2022-2024)
```

## Installation

```bash
git clone https://github.com/CSomers3/conformal_forecasting.git
cd conformal_forecasting
pip install -e .
```

**Requirements**: Python 3.8+, numpy, pandas, scikit-learn, pytorch, statsmodels

## Quick Start

### 1. Synthetic Experiments

Run controlled simulations to see ACI adapt to exchangeability violations:

```bash
cd notebooks/synthetic
jupyter notebook simulations.ipynb
```

Tests three violation types:
- **Temporal dependence**: AR(1), MA(1), ARMA(1,1) with increasing autocorrelation ($\rho \in$ [0.1, 0.95])
- **Regime shifts**: Calm→Normal→Stormy transitions; varying shift frequency
- **Spatial correlations**: $3\times 3$ wind farm with wake effects under 8 wind directions

### 2. Kelmarsh Case Study

Reproduce real-world results on Kelmarsh Wind Farm (6 turbines, Jan 2022–Jan 2024):

```bash
cd notebooks/kelmarsh
jupyter notebook data_prep.ipynb          # Load & clean SCADA + GFS data
jupyter notebook point_forecast_main.ipynb # Train STGCN, ARIMA, LGBM models
jupyter notebook conformal_main.ipynb     # Apply ACI, AgACI, DtACI
```

This follows a **rolling window protocol** (weekly retraining, expanding history) mimicking operational forecasting.

### 3. Core API

```python
import numpy as np
from src.ACI import aci

# Your data
y = np.array([...])       # Actuals
preds = np.array([...])   # Point forecasts

alpha = 0.1               # Target miscoverage rate
gamma = 0.05              # Learning rate (ACI tuning parameter)
residuals = []            # Running calibration set

betas = []                # Coverage history [0 or 1]

for t in range(warmup, len(y)):
    # 1. Compute quantile with adaptive alpha
    q = np.quantile(residuals, 1 - alpha_t)
    
    # 2. Form interval
    lower, upper = preds[t] - q, preds[t] + q
    
    # 3. Check coverage
    beta = int((y[t] >= lower) and (y[t] <= upper))
    betas.append(beta)
    
    # 4. [ACI METHOD EXTENSION] Update alpha for next step
    out = aci(np.array(betas), alpha, gamma)
    alpha_t = out["alphaSeq"][-1]
    
    residuals.append(abs(y[t] - preds[t]))
```

## Methods Implemented

| Method | Key Feature | Trade-off |
|--------|-------------|-----------|
| **ACI** | Fixed learning rate γ | Simple, fast; sensitive to γ choice |
| **AgACI** | Ensemble of experts with multiple γ | Robust to γ; ~10× slower |
| **DtACI** | Dynamic γ selection via meta-learner | Best regret bounds; moderate overhead |
| **OSSCP** | Rolling window baseline | Non-adaptive; fails under regime shift |

See Table 5.1 in thesis simulations for runtime comparisons.

## Limitations & Future Work

**Current limitations:**
- Symmetric intervals not ideal for wind (power $\geq$ 0); systematic upper-bound bias observed
- Multi-step coverage degrades with forecast horizon (see Figure 6.5)
- Small real-world dataset (one 6-turbine farm, 2 years)
- Computational overhead of AgACI/DtACI may limit real-time deployment

**Next steps:**
- Asymmetric/physically-constrained conformal methods
- Multi-step specific algorithms (e.g., Wang & Hyndman 2024)
- Physics-informed non-conformity scores
- Validation on larger farms and longer time horizons

## Data

**Kelmarsh Wind Farm** (open-source): 6 × 2050 kW turbines, Northamptonshire, UK  
- **SCADA**: 10-min turbine-level measurements (power, wind speed, direction, temperature, etc.)  
- **NWP**: NCEP GFS forecasts at hourly resolution, 0.11° grid (~13 km)  
- **License**: [Plumley (2022)](https://zenodo.org/records/16807551)

## Citation

If you use this code, please cite the thesis (pending publication):

```bibtex
@mastersthesis{somers2025conformal,
  author  = {Carl Somers},
  title   = {Reliable Uncertainty Quantification for Wind Farm Forecasting: 
             Adaptive Conformal Prediction under Distribution Shift},
  school  = {University College London},
  year    = {2025}
}
```

## References

- Gibbs, I. & Candès, E. (2021). [Adaptive conformal inference under distribution shift](https://arxiv.org/abs/2106.00170)
- Zaffran, M., Féron, O., et al. (2022). [Adaptive conformal predictions for time series](https://arxiv.org/abs/2202.07282)
- Gibbs, I. & Candès, E. (2024). [Conformal inference for online prediction with arbitrary distribution shifts](https://arxiv.org/abs/2208.08401)

## Issues & Questions

Found a bug? Have questions? [Open an issue](https://github.com/CSomers3/conformal_forecasting/issues) or reach out via LinkedIn.
