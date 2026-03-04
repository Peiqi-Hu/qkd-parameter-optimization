"""
BB84 key-rate dashboard: ideal vs noisy, multiple parameter views.

This script pulls together the reusable models in `key_rate_models.py`
and produces a "dashboard" of plots comparing **ideal** and **noisy**
BB84 key rates as functions of several parameters:

1. Key rate vs QBER.
2. Key rate vs transmittance η.
3. Key rate vs fiber distance.
4. Key rate vs dark-count noise Y0.
5. Key rate vs channel loss (in dB).

It also generates a small CSV summary table of representative parameter
combinations, including:
- input parameters (distance, η, Y0, e_d),
- resulting QBER (ideal input, noisy calculated),
- gains (ideal and noisy),
- ideal and noisy key rates.

The underlying formulas and references are documented in
`key_rate_models.py` (Bennett & Brassard 1984; Shor & Preskill 2000;
Ma et al. 2005).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .key_rate_models import (
    key_rate_ideal_per_sifted_bit,
    key_rate_ideal_per_pulse,
    key_rate_noisy_per_pulse,
    overall_gain,
    overall_qber,
    transmittance_from_distance_km,
    key_rate_ideal_vs_distance,
    key_rate_noisy_vs_distance,
    DEFAULT_FIBER_LOSS_DB_PER_KM,
    DEFAULT_DETECTOR_EFFICIENCY,
)


# Tuned default noise parameters for Stage 1 demos.
DEFAULT_Q_SIFT: float = 0.5
DEFAULT_Y0: float = 1e-6       # dark count probability per pulse
DEFAULT_E_D: float = 0.015     # misalignment error (~1.5 %)
DEFAULT_F_EC: float = 1.16     # realistic-ish error-correction efficiency


def create_dashboard(
    max_distance_km: float = 100.0,
    qber_ideal: float = 0.02,
    distance_for_qber_compare_km: float = 50.0,
) -> None:
    """
    Generate a dashboard of key-rate plots (5 views) and a summary table.
    """
    # Distance grid
    distances = np.linspace(0.0, max_distance_km, 400)

    # Transmittance grid
    eta_values = np.linspace(0.0, 1.0, 400)

    # QBER grid for ideal model
    q_values = np.linspace(0.0, 0.2, 400)

    # 1) Key rate vs QBER (ideal: direct; noisy: treat e_d ≈ Q for comparison)
    k_ideal_q = key_rate_ideal_per_sifted_bit(q_values)
    eta_fixed_for_q = transmittance_from_distance_km(distance_for_qber_compare_km)
    r_ideal_q = key_rate_ideal_per_pulse(
        qber=q_values,
        eta=eta_fixed_for_q,
        q_sift=DEFAULT_Q_SIFT,
    )
    # For the noisy curve, we sweep misalignment e_d and label x-axis by e_d ≈ Q.
    e_d_sweep = q_values
    r_noisy_q = key_rate_noisy_per_pulse(
        eta=eta_fixed_for_q,
        y0=DEFAULT_Y0,
        e_d=e_d_sweep,
        q_sift=DEFAULT_Q_SIFT,
        f_ec=DEFAULT_F_EC,
    )

    # 2) Key rate vs transmittance η
    r_ideal_eta = key_rate_ideal_per_pulse(
        qber=qber_ideal,
        eta=eta_values,
        q_sift=DEFAULT_Q_SIFT,
    )
    r_noisy_eta = key_rate_noisy_per_pulse(
        eta=eta_values,
        y0=DEFAULT_Y0,
        e_d=DEFAULT_E_D,
        q_sift=DEFAULT_Q_SIFT,
        f_ec=DEFAULT_F_EC,
    )

    # 3) Key rate vs distance
    r_ideal_dist = key_rate_ideal_vs_distance(
        distance_km=distances,
        qber=qber_ideal,
        q_sift=DEFAULT_Q_SIFT,
    )
    r_noisy_dist = key_rate_noisy_vs_distance(
        distance_km=distances,
        y0=DEFAULT_Y0,
        e_d=DEFAULT_E_D,
        q_sift=DEFAULT_Q_SIFT,
        f_ec=DEFAULT_F_EC,
    )

    # 4) Key rate vs dark-count noise Y0 (log-scale)
    y0_values = np.logspace(-8, -4, 300)
    eta_fixed_for_noise = eta_fixed_for_q
    r_ideal_noise = key_rate_ideal_per_pulse(
        qber=qber_ideal,
        eta=eta_fixed_for_noise,
        q_sift=DEFAULT_Q_SIFT,
    ) * np.ones_like(y0_values)
    r_noisy_noise = key_rate_noisy_per_pulse(
        eta=eta_fixed_for_noise,
        y0=y0_values,
        e_d=DEFAULT_E_D,
        q_sift=DEFAULT_Q_SIFT,
        f_ec=DEFAULT_F_EC,
    )

    # 5) Key rate vs channel loss (in dB)
    # Sweep η on a log-scale to get a broad loss range.
    eta_loss = np.logspace(-6, 0, 400)
    loss_db = -10.0 * np.log10(eta_loss)
    r_ideal_loss = key_rate_ideal_per_pulse(
        qber=qber_ideal,
        eta=eta_loss,
        q_sift=DEFAULT_Q_SIFT,
    )
    r_noisy_loss = key_rate_noisy_per_pulse(
        eta=eta_loss,
        y0=DEFAULT_Y0,
        e_d=DEFAULT_E_D,
        q_sift=DEFAULT_Q_SIFT,
        f_ec=DEFAULT_F_EC,
    )

    # --- Create a 3x2 grid of subplots (5 used) ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # (1) Key rate vs QBER
    ax = axes[0, 0]
    ax.plot(q_values * 100.0, r_ideal_q, label="Ideal")
    ax.plot(e_d_sweep * 100.0, r_noisy_q, label="Noisy (e_d ≈ Q)")
    ax.set_xlabel("QBER / misalignment [%]")
    ax.set_ylabel("Key rate [bits per pulse]")
    ax.set_title(
        "Key rate vs QBER / misalignment\n"
        f"(fixed distance {distance_for_qber_compare_km:.0f} km)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (2) Key rate vs transmittance η
    ax = axes[0, 1]
    ax.plot(eta_values, r_ideal_eta, label="Ideal")
    ax.plot(eta_values, r_noisy_eta, label="Noisy")
    ax.set_xlabel("Transmittance η")
    ax.set_ylabel("Key rate [bits per pulse]")
    ax.set_title("Key rate vs transmittance η")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (3) Key rate vs distance
    ax = axes[1, 0]
    ax.plot(distances, r_ideal_dist, label="Ideal")
    ax.plot(distances, r_noisy_dist, label="Noisy")
    ax.set_xlabel("Fiber distance [km]")
    ax.set_ylabel("Key rate [bits per pulse]")
    ax.set_title(
        f"Key rate vs distance\n"
        f"(QBER_ideal={qber_ideal*100:.1f}%, α={DEFAULT_FIBER_LOSS_DB_PER_KM} dB/km, "
        f"η_det={DEFAULT_DETECTOR_EFFICIENCY})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (4) Key rate vs dark-count noise Y0
    ax = axes[1, 1]
    ax.semilogx(y0_values, r_ideal_noise, label="Ideal (independent of Y0)")
    ax.semilogx(y0_values, r_noisy_noise, label="Noisy")
    ax.set_xlabel("Dark count probability per pulse Y0")
    ax.set_ylabel("Key rate [bits per pulse]")
    ax.set_title(
        "Key rate vs dark-count noise Y0\n"
        f"(distance {distance_for_qber_compare_km:.0f} km)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # (5) Key rate vs channel loss (dB)
    ax = axes[2, 0]
    ax.plot(loss_db, r_ideal_loss, label="Ideal")
    ax.plot(loss_db, r_noisy_loss, label="Noisy")
    ax.set_xlabel("Channel loss [dB]")
    ax.set_ylabel("Key rate [bits per pulse]")
    ax.set_title("Key rate vs channel loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Empty last subplot as a placeholder for text/legend if needed.
    axes[2, 1].axis("off")

    fig.tight_layout()
    plt.show()

    # Also generate a summary table for a small parameter grid.
    generate_summary_table(
        output_csv=Path("bb84_dashboard_summary.csv"),
        qber_ideal=qber_ideal,
    )


def generate_summary_table(
    output_csv: Path,
    qber_ideal: float,
    distances_km: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0),
    y0_values: tuple[float, ...] = (1e-8, 1e-6, 1e-4),
    e_d_values: tuple[float, ...] = (0.01, 0.02),
) -> None:
    """
    Generate a small CSV table of representative parameter combinations.

    Each row contains:
    - distance_km, eta_total,
    - Y0, e_d,
    - Q_ideal (input QBER), Q_noisy,
    - gain_ideal, gain_noisy,
    - R_ideal, R_noisy.
    """
    rows: list[dict[str, float]] = []

    for d in distances_km:
        eta = float(transmittance_from_distance_km(d))
        for y0 in y0_values:
            for e_d in e_d_values:
                # Ideal metrics
                gain_ideal = eta  # simple model: gain ≈ η
                r_ideal = float(
                    key_rate_ideal_per_pulse(
                        qber=qber_ideal,
                        eta=eta,
                        q_sift=DEFAULT_Q_SIFT,
                    )
                )

                # Noisy metrics
                gain_noisy = float(overall_gain(eta=eta, y0=y0))
                q_noisy = float(overall_qber(eta=eta, y0=y0, e_d=e_d))
                r_noisy = float(
                    key_rate_noisy_per_pulse(
                        eta=eta,
                        y0=y0,
                        e_d=e_d,
                        q_sift=DEFAULT_Q_SIFT,
                        f_ec=DEFAULT_F_EC,
                    )
                )

                rows.append(
                    {
                        "distance_km": d,
                        "eta_total": eta,
                        "Y0": y0,
                        "e_d": e_d,
                        "Q_ideal": qber_ideal,
                        "Q_noisy": q_noisy,
                        "gain_ideal": gain_ideal,
                        "gain_noisy": gain_noisy,
                        "R_ideal": r_ideal,
                        "R_noisy": r_noisy,
                    }
                )

    fieldnames = [
        "distance_km",
        "eta_total",
        "Y0",
        "e_d",
        "Q_ideal",
        "Q_noisy",
        "gain_ideal",
        "gain_noisy",
        "R_ideal",
        "R_noisy",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Pretty console preview in a compact table format.
    print(f"\nSaved summary table to {output_csv}\n")

    # Choose a subset of columns and a column width for visualization.
    display_cols = [
        "distance_km",
        "eta_total",
        "Y0",
        "e_d",
        "Q_noisy",
        "gain_noisy",
        "R_noisy",
    ]
    col_width = 12

    def fmt(val: float) -> str:
        # Compact numeric formatting suitable for console viewing.
        if abs(val) >= 1e-2 and abs(val) < 1e3:
            return f"{val: .4f}"
        else:
            return f"{val: .2e}"

    # Header
    header = "".join(name.center(col_width) for name in display_cols)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    # Show the first N rows in a formatted way.
    preview_rows = rows[:10]
    for row in preview_rows:
        line = "".join(
            (
                f"{fmt(row[c])}".rjust(col_width)
                if c not in ("distance_km",)
                else f"{row[c]: .1f}".rjust(col_width)
            )
            for c in display_cols
        )
        print(line)
    print(sep)
    print("(showing first 10 rows; see CSV for full table)\n")


def main() -> None:
    create_dashboard()


if __name__ == "__main__":
    main()

