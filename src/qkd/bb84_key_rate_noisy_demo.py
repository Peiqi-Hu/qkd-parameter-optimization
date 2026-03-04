"""
Noisy BB84 key rate demo using reusable key-rate models.

This script demonstrates how to use the simple noisy-channel model
implemented in `key_rate_models.py` to generate:
- Key rate vs. transmittance for fixed noise parameters,
- Key rate vs. dark count probability for fixed transmittance,
- (optionally) key rate vs. misalignment error.

All the physics and references live in `key_rate_models.py`; this file
is focused on visualization and will be used in Stage 1 for sanity
checks and intuition-building plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .key_rate_models import key_rate_noisy_per_pulse, overall_qber


def plot_noisy_key_rate_vs_transmittance(
    y0: float = 1e-6,
    e_d: float = 0.015,
    q_sift: float = 0.5,
    f_ec: float = 1.1,
    eta_min: float = 0.0,
    eta_max: float = 1.0,
    num_points: int = 400,
    save_path: str | None = "noisy_key_rate_vs_transmittance.png",
) -> None:
    """
    Plot noisy per-pulse key rate vs channel transmittance η.
    """
    eta_values = np.linspace(eta_min, eta_max, num_points)
    r_values = key_rate_noisy_per_pulse(
        eta=eta_values,
        y0=y0,
        e_d=e_d,
        q_sift=q_sift,
        f_ec=f_ec,
    )
    e_mu_values = overall_qber(eta_values, y0, e_d)

    plt.figure(figsize=(6, 4))
    plt.plot(eta_values, r_values, label="Noisy BB84 key rate")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Channel transmittance η")
    plt.ylabel("Secret key per pulse R_noisy [bits]")
    plt.title(
        f"Noisy BB84 key rate vs transmittance\n"
        f"(Y0={y0:g}, e_d={e_d*100:.1f}%, f_ec={f_ec:.2f})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

    # Optionally, a quick look at QBER vs η on a separate figure.
    plt.figure(figsize=(6, 4))
    plt.plot(eta_values, e_mu_values * 100.0, label="QBER E_μ")
    plt.xlabel("Channel transmittance η")
    plt.ylabel("QBER E_μ [%]")
    plt.title("Noisy model QBER vs transmittance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        qber_path = save_path.replace(".png", "_qber.png")
        plt.savefig(qber_path, dpi=300)
    plt.show()


def plot_noisy_key_rate_vs_dark_counts(
    eta: float = 0.1,
    e_d: float = 0.015,
    q_sift: float = 0.5,
    f_ec: float = 1.1,
    y0_min: float = 1e-8,
    y0_max: float = 1e-4,
    num_points: int = 400,
    save_path: str | None = "noisy_key_rate_vs_dark_counts.png",
) -> None:
    """
    Plot noisy per-pulse key rate vs dark count probability Y0
    on a log scale, for a fixed transmittance η.
    """
    y0_values = np.logspace(np.log10(y0_min), np.log10(y0_max), num_points)
    r_values = key_rate_noisy_per_pulse(
        eta=eta,
        y0=y0_values,
        e_d=e_d,
        q_sift=q_sift,
        f_ec=f_ec,
    )

    plt.figure(figsize=(6, 4))
    plt.semilogx(y0_values, r_values, label=f"η = {eta}")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Dark count probability per pulse Y0")
    plt.ylabel("Secret key per pulse R_noisy [bits]")
    plt.title(
        f"Noisy BB84 key rate vs dark counts\n"
        f"(e_d={e_d*100:.1f}%, f_ec={f_ec:.2f})"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def main() -> None:
    """
    Generate noisy-model demonstration plots:
    - Key rate vs transmittance,
    - Key rate vs dark count probability.
    """
    plot_noisy_key_rate_vs_transmittance()
    plot_noisy_key_rate_vs_dark_counts()


if __name__ == "__main__":
    main()

