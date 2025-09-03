"""
Dynamically-tuned Adaptive Conformal Inference.

Implementation of 'Conformal Inference for Online Prediction with Arbitrary
Distribution Shifts', Gibbs and Candès (2023), https://arxiv.org/abs/2208.08401
"""

import numpy as np
from numpy.typing import NDArray


def loss(x: NDArray, alpha: float) -> NDArray:
    """Pinball loss used in DtACI."""
    return alpha * x - np.minimum(0, x)


def dtaci(
    betas: NDArray,
    alpha: float,
    gammas: NDArray,
    I: int = 30,
) -> dict:
    """
    Select miscoverage levels in an online fashion according to DtACI.

    Args:
        betas: observed quantile levels that achieve smallest prediction sets
        alpha: target miscoverage level
        gammas: step-size candidates
        I: size of local time interval

    Returns:
        Dictionary containing:
            - alphaSeq: sequence of aggregated miscoverage levels
            - errSeq: sequence of coverage errors
            - gammaSeq: placeholder (not defined for dtACI)
    """
    T = len(betas)
    k = len(gammas)

    assert I <= T, "Local window I must not exceed total time horizon T"

    denom = (1 - alpha) ** 2 * alpha ** 3 + alpha ** 2 * (1 - alpha) ** 3
    eta = np.sqrt(3 / I) * np.sqrt((np.log(k * I) + 2) / denom)
    sigma = 1 / (2 * I)

    w = np.ones(k)
    alphas = np.full(k, alpha)
    alpha_seq = np.empty(T)
    gamma_seq = np.full(T, np.nan)  # Not meaningful for dtACI
    err_seq = np.zeros(T)

    for t in range(T):
        p = w / w.sum()
        alpha_seq[t] = (p * alphas).sum()
        err_seq[t] = float(alpha_seq[t] > betas[t])

        w_bar = w * np.exp(-eta * loss(betas[t] - alphas, alpha))
        w = (1 - sigma) * w_bar / w_bar.sum() + sigma / k

        err = (alphas > betas[t]).astype(float)
        alphas += gammas * (alpha - err)

    alpha_seq = np.clip(alpha_seq, 0, 1)

    return {
        "alphaSeq": alpha_seq,
        "errSeq": err_seq,
        "gammaSeq": gamma_seq
    }