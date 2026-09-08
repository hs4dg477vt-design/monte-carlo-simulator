from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulator import SimulationResult


def save_equity_fan_chart(
    result: SimulationResult,
    output_path: Path,
    *,
    initial_capital: float,
    actual_trade_pnl: np.ndarray | None = None,
    seed: int = 42,
    max_paths: int = 150,
) -> None:
    """Save a readable sample of paths plus percentile confidence bands."""

    paths = result.equity_paths
    rng = np.random.default_rng(seed)
    count = min(max_paths, paths.shape[0])
    selected = rng.choice(paths.shape[0], size=count, replace=False)

    x = np.arange(paths.shape[1] + 1)
    with_start = np.concatenate(
        [np.full((paths.shape[0], 1), initial_capital), paths],
        axis=1,
    )
    p05, p50, p95 = np.quantile(with_start, [0.05, 0.50, 0.95], axis=0)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor("#f3efe7")
    ax.set_facecolor("#f3efe7")

    for path in with_start[selected]:
        ax.plot(x, path, color="#2f6b5f", alpha=0.07, linewidth=0.7)

    ax.fill_between(x, p05, p95, color="#4c8c7d", alpha=0.18, label="5th-95th percentile")
    ax.plot(x, p50, color="#d35432", linewidth=2.2, label="Median simulation")
    ax.plot(x, p05, color="#2f6b5f", linewidth=1.0, alpha=0.8)
    ax.plot(x, p95, color="#2f6b5f", linewidth=1.0, alpha=0.8)

    if actual_trade_pnl is not None and len(actual_trade_pnl) == paths.shape[1]:
        actual = initial_capital + np.cumsum(actual_trade_pnl)
        ax.plot(
            x,
            np.concatenate([[initial_capital], actual]),
            color="#1d2830",
            linewidth=1.8,
            label="Observed ordering",
        )

    ax.set_title("Monte Carlo Equity Paths", fontsize=18, fontweight="bold", loc="left")
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Account balance ($)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
