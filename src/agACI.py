"""
Aggregated Adaptive Conformal Inference.

Implementation of "Adaptive Conformal Predictions for Time Series". 
Zaffran et Al. (2022), https://arxiv.org/abs/2202.07282
"""


import numpy as np
from numpy.typing import NDArray

def vec_zero_max(x: NDArray) -> NDArray:
    """Vectorized max(., 0) operation."""
    return np.maximum(x, 0)

def vec_zero_min(x: NDArray) -> NDArray:
    """Vectorized min(., 0) operation."""
    return np.minimum(x, 0)

def pinball(u: NDArray, alpha: float) -> NDArray:
    """Pinball loss function used in AgACI."""
    return alpha * u - vec_zero_min(u)

def agaci(
    betas: NDArray,
    alpha: float,
    gammas: NDArray,
    alpha_init: float = None,
    eps: float = 1e-8,  # Increased for better stability
) -> dict:
    """
    Aggregate expert predictions for adaptive conformal inference.

    Args:
        betas: observed quantile levels that achieve smallest prediction sets
        alpha: target miscoverage level
        gammas: step-size candidates for expert algorithms
        alpha_init: initial miscoverage level (defaults to alpha if None)
        eps: small constant for numerical stability

    Returns:
        Dictionary containing:
            - alphaSeq: sequence of aggregated miscoverage levels
            - errSeq: sequence of coverage errors
            - gammaSeq: sequence of aggregated step sizes
    """
    betas = np.array(betas, dtype=float)
    gammas = np.array(gammas, dtype=float)
    T = len(betas)
    k = len(gammas)

    if alpha_init is None:
        alpha_init = alpha

    # output init
    alpha_seq = np.full(T, alpha_init)
    err_seq = np.zeros(T)
    gamma_seq = np.zeros(T)

    # expert init
    expert_alphas = np.full(k, alpha_init)
    expert_probs = np.full(k, 1.0 / k)
    expert_sq_losses = np.zeros(k)
    expert_etas = np.zeros(k)
    expert_l_values = np.zeros(k)
    expert_max_losses = np.zeros(k)

    for t in range(T):
        # === Compute predictions ===
        alpha_seq[t] = np.sum(expert_probs * expert_alphas)
        err_seq[t] = float(alpha_seq[t] > betas[t])
        gamma_seq[t] = np.sum(expert_probs * gammas)

        # === Update expert weights ===
        expert_losses = (err_seq[t] - alpha) * (expert_alphas - alpha_seq[t])
        expert_sq_losses += expert_losses ** 2
        expert_max_losses = np.maximum(expert_max_losses, np.abs(expert_losses))

        abs_max = np.abs(expert_max_losses) + eps
        log_term = np.log2(abs_max)
        expert_evals = np.power(2, np.ceil(log_term) + 1).clip(min=eps) 

        # Update l_values
        eta_loss_term = expert_etas * expert_losses
        expert_l_values += 0.5 * (
            expert_losses * (1 + eta_loss_term)
            + expert_evals * (eta_loss_term > 0.5)
        )

        # Stabilized etas: add eps to denominator, clip to prevent inf
        sqrt_term = np.sqrt(np.log(k) / (expert_sq_losses + eps) + eps)
        div_term = 1 / expert_evals + eps
        expert_etas = np.minimum(div_term, sqrt_term).clip(max=1.0)  # Upper clip for stability
        update_term = gammas * (alpha - (expert_alphas > betas[t]).astype(float))
        expert_alphas = np.clip(expert_alphas + update_term, 0, 1)

        # Stabilise
        eta_l = expert_etas * expert_l_values
        exp_arg = -eta_l + np.max(eta_l)
        exp_arg = np.clip(exp_arg, -700, 700)  # Prevent exp overflow/underflow
        expert_weights = expert_etas * np.exp(exp_arg)

        weight_sum = np.sum(expert_weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            expert_probs = np.full(k, 1.0 / k)
        else:
            expert_probs = expert_weights / weight_sum

    return {
        "alphaSeq": alpha_seq,
        "errSeq": err_seq,
        "gammaSeq": gamma_seq
    }

