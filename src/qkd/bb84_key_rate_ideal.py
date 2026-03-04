"""
Asymptotic ideal BB84 secret key rate and basic plots (demo).

This script now acts as a *demo* that uses the reusable functions
in `key_rate_models.py` to generate:
- Key rate vs. QBER (ideal asymptotic BB84).
- Key rate vs. channel transmittance (ideal model).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .key_rate_models import key_rate_ideal_per_sifted_bit, key_rate_ideal_per_pulse


def plot_key_rate_vs_qber(
    q_min: float = 0.0,
    q_max: float = 0.2,
    num_points: int = 500,
    save_path: str | None = "key_rate_vs_qber_ideal_bb84.png",
) -> None:
    """
    Plot key rate per sifted bit vs. QBER using the asymptotic ideal BB84 formula.
    """
    q_values = np.linspace(q_min, q_max, num_points)
    k_values = key_rate_ideal_per_sifted_bit(q_values)

    plt.figure(figsize=(6, 4))
    plt.plot(q_values * 100.0, k_values, label="Asymptotic ideal BB84")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("QBER Q [%]")
    plt.ylabel("Secret key per sifted bit K(Q) [bits]")
    plt.title("Asymptotic ideal BB84 key rate vs QBER")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    # Also show for interactive exploration.
    plt.show()


def plot_key_rate_vs_transmittance(
    qber: float = 0.02,
    q_sift: float = 0.5,
    eta_min: float = 0.0,
    eta_max: float = 1.0,
    num_points: int = 500,
    save_path: str | None = "key_rate_vs_transmittance_ideal_bb84.png",
) -> None:
    """
    Plot per-pulse key rate vs. channel transmittance η for a fixed QBER.

    This uses the simple model:
        R(Q, η) = q_sift * η * K(Q).
    """
    eta_values = np.linspace(eta_min, eta_max, num_points)
    r_values = key_rate_ideal_per_pulse(qber=qber, eta=eta_values, q_sift=q_sift)

    plt.figure(figsize=(6, 4))
    plt.plot(eta_values, r_values, label=f"QBER = {qber * 100:.1f} %, q = {q_sift}")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Channel transmittance η")
    plt.ylabel("Secret key per pulse R(Q, η) [bits]")
    plt.title("Asymptotic ideal BB84 key rate vs transmittance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def main() -> None:
    """
    Generate the basic theoretical plots for Stage 1:
    - Key rate vs QBER
    - Key rate vs transmittance (for a fixed QBER)
    """
    # First, key fraction vs QBER.
    plot_key_rate_vs_qber()

    # Then, key rate vs transmittance for a representative QBER (e.g. 2%).
    plot_key_rate_vs_transmittance(qber=0.02, q_sift=0.5)


if __name__ == "__main__":
    main()

