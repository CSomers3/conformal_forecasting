"""
Adaptive Conformal Inference (ACI).

Implementation of "Adaptive Conformal Inference".
Gibbs and Candes (2021), https://arxiv.org/abs/2106.00170
"""

import numpy as np
from numpy.typing import NDArray


def aci(
    betas: NDArray,
    alpha: float,
    gamma: float,
    alpha_init: float = None,
) -> dict:
    """
    Online adaptive conformal inference (ACI).

    Args:
        betas: coverage indicators — i.e., whether the prediction set covered the true value (0/1 or float)
        alpha: target miscoverage rate (e.g., 0.1)
        gamma: learning rate (typically small, e.g., 0.01)
        alpha_init: initial guess for alpha (defaults to target alpha)

    Returns:
        Dictionary containing:
            - alphaSeq: sequence of adapted miscoverage levels
            - errSeq: sequence of coverage errors
    """
    betas = np.array(betas, dtype=float)
    T = len(betas)

    if alpha_init is None:
        alpha_init = alpha

    # Initialise sequences
    alpha_seq = np.full(T, alpha_init)
    err_seq = np.zeros(T)

    current_alpha = alpha_init

    for t in range(T):
        # Compute error: 1 if current alpha > beta (i.e., prediction set failed to cover)
        err_seq[t] = float(current_alpha > betas[t])

        # Update alpha via projected SGD
        current_alpha += gamma * (alpha - err_seq[t])
        current_alpha = np.clip(current_alpha, 0.0, 1.0)

        alpha_seq[t] = current_alpha

    return {
        "alphaSeq": alpha_seq,
        "errSeq": err_seq,
    }
