"""
Key rate models for BB84 QKD (ideal and simple noisy channel).

This module provides reusable functions for:
- Asymptotic ideal BB84 secret key fraction per sifted bit.
- A simple per-pulse key rate model including channel transmittance.
- A more detailed *noisy* model with:
  - channel transmittance η,
  - detector dark count probability per pulse y,
  - misalignment (intrinsic) error probability e_d,
  - error correction efficiency f_ec ≥ 1.

These functions are intended for:
- Stage 1: theoretical plots and monotonicity checks,
- later stages: dataset generation and ML-based parameter optimization.

-----------------------------------------------------------------------
Ideal asymptotic BB84 (per sifted bit)
-----------------------------------------------------------------------
Under the Shor–Preskill security proof in the asymptotic limit with
ideal single-photon sources and detectors (no side channels, no
finite-size effects), the secret key fraction per sifted bit is:

    K_ideal(Q) = max(0, 1 - 2 h2(Q)),

where Q is the quantum bit error rate (QBER) and h2 is the binary
entropy in bits:

    h2(x) = -x log2 x - (1 - x) log2(1 - x).

-----------------------------------------------------------------------
Simple realistic noisy model (decoy-style, single signal intensity)
-----------------------------------------------------------------------
For a very common asymptotic model of BB84 with weak coherent pulses,
transmittance η, dark counts, and misalignment, one often starts from
the key rate per *pulse* (per emitted signal) of the form
(see, e.g., Ma et al., PRA 72, 012326 (2005)):

    R ≈ q * [ Q1 * (1 - h2(e1)) - f_ec * Q_μ * h2(E_μ) ],

where (in a simplified single-intensity picture):
- q          : sifting factor (1/2 for unbiased BB84),
- Q1         : gain of single-photon states,
- e1         : error rate of single-photon states,
- Q_μ        : overall gain (probability that Bob gets a detection),
- E_μ        : overall QBER,
- f_ec ≥ 1   : error correction inefficiency factor (e.g., 1.1).

Here we implement a simplified *effective* version suitable for
Stage 1 experiments by modeling:

    Y0       : background (dark) count probability per pulse,
    η        : overall channel transmittance,
    e_d      : misalignment error probability (~ a few percent),

and:
    Q_μ(η, Y0)  = Y0 + η * (1 - Y0)   (prob. of a click),
    E_μ(η, Y0)  = [0.5 * Y0 + e_d * η * (1 - Y0)] / Q_μ(η, Y0),

so that the observed QBER E_μ arises from half-random dark counts
and misalignment on the signal component.

We then use a *single effective* key-fraction:

    K_eff(E_μ) = max(0, 1 - f_ec * h2(E_μ) - h2(E_μ)),

which mimics the structure of the Shor–Preskill rate with a penalty
f_ec on error correction. This keeps the model simple but gives the
correct qualitative monotonic dependences:
- Increasing η   → increasing key rate (up to saturation),
- Increasing Y0  → increasing QBER and decreasing key rate,
- Increasing e_d → increasing QBER and decreasing key rate.

This is intentionally lightweight and suitable as a starting point
for more realistic finite-key and full decoy-state analyses later.

-----------------------------------------------------------------------
References (ideal BB84 & noisy models)
-----------------------------------------------------------------------
- C. H. Bennett and G. Brassard,
  "Quantum cryptography: Public key distribution and coin tossing,"
  in Proceedings of IEEE International Conference on Computers, Systems
  and Signal Processing, Bangalore, India, pp. 175–179, 1984.
  (Original BB84 protocol.)

- P. W. Shor and J. Preskill,
  "Simple proof of security of the BB84 quantum key distribution protocol,"
  Physical Review Letters 85, 441–444 (2000).
  (Asymptotic security proof giving K(Q) = 1 - 2 h2(Q) for ideal BB84.)

- X. Ma, B. Qi, Y. Zhao, and H.-K. Lo,
  "Practical decoy state for quantum key distribution,"
  Physical Review A 72, 012326 (2005).
  (Standard reference for practical BB84 with decoy states and realistic
   channel/detector modeling; we follow its spirit for the noisy model,
   though here in a simplified single-intensity form.)
"""

from __future__ import annotations

import numpy as np


def binary_entropy(q: np.ndarray | float) -> np.ndarray:
    """
    Binary entropy function h2(q) in bits:
        h2(q) = -q log2 q - (1 - q) log2(1 - q)

    Defined for q in [0, 1]. We handle the endpoints by continuity:
        h2(0) = h2(1) = 0.
    """
    q_arr = np.asarray(q, dtype=float)
    eps = 1e-12
    q_clipped = np.clip(q_arr, eps, 1 - eps)
    h = -q_clipped * np.log2(q_clipped) - (1 - q_clipped) * np.log2(1 - q_clipped)
    h = np.where((q_arr <= 0) | (q_arr >= 1), 0.0, h)
    return h


# -------------------------------------------------------------------
# Ideal asymptotic BB84
# -------------------------------------------------------------------

def key_rate_ideal_per_sifted_bit(qber: np.ndarray | float) -> np.ndarray:
    """
    Asymptotic ideal BB84 secret key fraction per sifted bit:

        K_ideal(Q) = max(0, 1 - 2 h2(Q)),

    where Q is the QBER and h2 is the binary entropy in bits.
    """
    qber_arr = np.asarray(qber, dtype=float)
    k = 1.0 - 2.0 * binary_entropy(qber_arr)
    return np.maximum(k, 0.0)


def key_rate_ideal_per_pulse(
    qber: np.ndarray | float,
    eta: np.ndarray | float,
    q_sift: float = 0.5,
) -> np.ndarray:
    """
    Simple per-pulse key rate model for ideal BB84:

        R_ideal(Q, η) = q_sift * η * K_ideal(Q),

    where:
        - q_sift is the basis-sifting factor (default 1/2 for unbiased BB84),
        - η is the total channel transmittance,
        - K_ideal(Q) is the secret fraction per sifted bit.
    """
    eta_arr = np.asarray(eta, dtype=float)
    k = key_rate_ideal_per_sifted_bit(qber)
    return q_sift * eta_arr * k


# -------------------------------------------------------------------
# Simple noisy model (effective decoy-style)
# -------------------------------------------------------------------

def overall_gain(eta: np.ndarray | float, y0: np.ndarray | float) -> np.ndarray:
    """
    Overall gain (click probability) in a simple model:

        Q_μ(η, Y0) = Y0 + η * (1 - Y0),

    where:
        - η  : total channel transmittance,
        - Y0 : background (dark) count probability per pulse.
    """
    eta_arr = np.asarray(eta, dtype=float)
    y0_arr = np.asarray(y0, dtype=float)
    return y0_arr + eta_arr * (1.0 - y0_arr)


def overall_qber(
    eta: np.ndarray | float,
    y0: np.ndarray | float,
    e_d: np.ndarray | float,
) -> np.ndarray:
    """
    Overall QBER E_μ from dark counts and misalignment:

        E_μ(η, Y0) = [0.5 * Y0 + e_d * η * (1 - Y0)] / Q_μ(η, Y0),

    where:
        - Y0 : dark count probability per pulse,
        - η  : channel transmittance,
        - e_d: misalignment error probability on signal detections.

    We clip the result into [0, 0.5] for numerical stability.
    """
    eta_arr = np.asarray(eta, dtype=float)
    y0_arr = np.asarray(y0, dtype=float)
    e_d_arr = np.asarray(e_d, dtype=float)

    q_mu = overall_gain(eta_arr, y0_arr)
    # Avoid division by zero when gain is ~0 (extremely lossy channel).
    tiny = 1e-15
    q_mu_safe = np.where(q_mu < tiny, tiny, q_mu)

    numerator = 0.5 * y0_arr + e_d_arr * eta_arr * (1.0 - y0_arr)
    e_mu = numerator / q_mu_safe
    return np.clip(e_mu, 0.0, 0.5)


def key_rate_noisy_per_pulse(
    eta: np.ndarray | float,
    y0: np.ndarray | float,
    e_d: np.ndarray | float,
    q_sift: float = 0.5,
    f_ec: float = 1.1,
) -> np.ndarray:
    """
    Effective per-pulse key rate for a simple noisy BB84 model:

        R_noisy(η, Y0, e_d)
            ≈ q_sift * Q_μ(η, Y0)
               * max(0, 1 - h2(E_μ) - f_ec * h2(E_μ)),

    where:
        - Q_μ(η, Y0) is the overall gain (click probability),
        - E_μ is the resulting overall QBER,
        - f_ec ≥ 1 is the error-correction inefficiency factor.

    This is a deliberately simplified "effective" expression that
    captures correct qualitative dependencies for Stage 1 analysis.
    """
    eta_arr = np.asarray(eta, dtype=float)
    y0_arr = np.asarray(y0, dtype=float)
    e_d_arr = np.asarray(e_d, dtype=float)

    q_mu = overall_gain(eta_arr, y0_arr)
    e_mu = overall_qber(eta_arr, y0_arr, e_d_arr)

    penalty = (1.0 + f_ec) * binary_entropy(e_mu)
    k_eff = np.maximum(1.0 - penalty, 0.0)
    return q_sift * q_mu * k_eff


# Convenience aliases for external use
__all__ = [
    "binary_entropy",
    "key_rate_ideal_per_sifted_bit",
    "key_rate_ideal_per_pulse",
    "overall_gain",
    "overall_qber",
    "key_rate_noisy_per_pulse",
]

