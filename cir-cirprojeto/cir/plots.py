"""Plotting helpers for CIR analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

__all__ = [
    "plot_paths",
    "plot_hist_terminal",
    "plot_loglog_convergence",
    "plot_yield_curve",
]


def _resolve_path(path_png: str | Path) -> Path:
    """Ensure the target path lives under figures/ and has a .png suffix."""

    path = Path(path_png)
    if not path.suffix:
        path = path.with_suffix(".png")
    if not path.is_absolute():
        path = FIG_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, path_png: str | Path) -> Path:
    """Save and close a Matplotlib figure."""

    out_path = _resolve_path(path_png)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_paths(
    t: Sequence[float],
    R: np.ndarray,
    title: str,
    path_png: str | Path,
) -> Path:
    """Line plot of Monte Carlo short-rate paths."""

    t = np.asarray(t)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, R.T, linewidth=1.0, alpha=0.85)
    ax.set(title=title, xlabel="Time", ylabel="Short rate")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
    return _save(fig, path_png)


def plot_hist_terminal(
    R_T: Sequence[float],
    bins: int | Sequence[float],
    title: str,
    path_png: str | Path,
) -> Path:
    """Histogram of terminal short rates."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(R_T, bins=bins, density=True, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.set(title=title, xlabel="Terminal short rate", ylabel="Density")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
    return _save(fig, path_png)


def plot_loglog_convergence(
    dts: Iterable[float],
    errors: Iterable[float],
    slope: float,
    intercept: float,
    path_png: str | Path,
) -> Path:
    """Log-log convergence plot with fitted slope."""

    dts = np.asarray(list(dts), dtype=float)
    errors = np.asarray(list(errors), dtype=float)
    order_line = np.exp(intercept) * dts**slope

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dts, errors, "o-", label="Observed error")
    ax.loglog(dts, order_line, "--", label=f"Fit slope={slope:.2f}")
    ax.set(
        title="Strong convergence order",
        xlabel="Δt",
        ylabel="Error",
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.legend()
    return _save(fig, path_png)


def plot_yield_curve(
    maturities: Sequence[float],
    prices: Sequence[float],
    yields: Sequence[float],
    path_png: str | Path,
) -> Path:
    """Plot price and yield curves on dual axes."""

    maturities = np.asarray(maturities)
    prices = np.asarray(prices)
    yields = np.asarray(yields)

    fig, ax_price = plt.subplots(figsize=(7, 4))
    ax_yield = ax_price.twinx()

    ax_price.plot(maturities, prices, "o-", color="#1f77b4", label="Bond price")
    ax_yield.plot(
        maturities,
        yields,
        "s--",
        color="#d62728",
        label="Zero-coupon yield",
    )

    ax_price.set_xlabel("Maturity")
    ax_price.set_ylabel("Price")
    ax_yield.set_ylabel("Yield")
    ax_price.set_title("Zero-coupon curve")
    ax_price.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)

    lines = ax_price.get_lines() + ax_yield.get_lines()
    labels = [line.get_label() for line in lines]
    ax_price.legend(lines, labels, loc="best")

    return _save(fig, path_png)
