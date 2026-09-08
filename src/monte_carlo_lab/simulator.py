from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SimulationMethod = Literal["block-bootstrap", "iid-bootstrap", "shuffle"]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a set of Monte Carlo equity paths."""

    simulations: int = 5_000
    horizon: int | None = None
    method: SimulationMethod = "block-bootstrap"
    block_size: int = 5
    initial_capital: float = 5_000.0
    ruin_floor: float = 1_000.0
    seed: int = 42


@dataclass(frozen=True)
class SimulationResult:
    """Paths, path-level metrics, and aggregate statistics."""

    pnl_paths: np.ndarray
    equity_paths: np.ndarray
    ending_balance: np.ndarray
    max_drawdown: np.ndarray
    max_drawdown_pct: np.ndarray
    summary: dict[str, object]


def _validated_pnl(trade_pnl: np.ndarray) -> np.ndarray:
    pnl = np.asarray(trade_pnl, dtype=np.float64)
    if pnl.ndim != 1:
        raise ValueError("trade_pnl must be a one-dimensional array")
    if pnl.size < 2:
        raise ValueError("at least two trade outcomes are required")
    if not np.isfinite(pnl).all():
        raise ValueError("trade_pnl contains missing or non-finite values")
    return pnl


def _block_bootstrap(
    pnl: np.ndarray,
    simulations: int,
    horizon: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_size < 1 or block_size > pnl.size:
        raise ValueError("block_size must be between 1 and the number of trades")

    blocks_needed = int(np.ceil(horizon / block_size))
    starts = rng.integers(
        0,
        pnl.size - block_size + 1,
        size=(simulations, blocks_needed),
    )
    offsets = np.arange(block_size)
    block_indices = starts[..., None] + offsets
    return pnl[block_indices].reshape(simulations, -1)[:, :horizon]


def generate_pnl_paths(
    trade_pnl: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Generate resampled P&L paths according to the selected method."""

    pnl = _validated_pnl(trade_pnl)
    if config.simulations < 1:
        raise ValueError("simulations must be positive")

    horizon = pnl.size if config.horizon is None else config.horizon
    if horizon < 1:
        raise ValueError("horizon must be positive")

    rng = np.random.default_rng(config.seed)
    if config.method == "block-bootstrap":
        return _block_bootstrap(
            pnl,
            config.simulations,
            horizon,
            config.block_size,
            rng,
        )
    if config.method == "iid-bootstrap":
        indices = rng.integers(0, pnl.size, size=(config.simulations, horizon))
        return pnl[indices]
    if config.method == "shuffle":
        if horizon != pnl.size:
            raise ValueError("shuffle requires horizon to equal the input trade count")
        return np.vstack([rng.permutation(pnl) for _ in range(config.simulations)])
    raise ValueError(f"unsupported simulation method: {config.method}")


def _path_metrics(
    pnl_paths: np.ndarray,
    initial_capital: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equity = initial_capital + np.cumsum(pnl_paths, axis=1)
    initial = np.full((equity.shape[0], 1), initial_capital, dtype=np.float64)
    equity_with_start = np.concatenate([initial, equity], axis=1)
    running_peak = np.maximum.accumulate(equity_with_start, axis=1)
    drawdown = running_peak - equity_with_start
    max_drawdown = drawdown.max(axis=1)

    drawdown_pct = np.divide(
        drawdown,
        running_peak,
        out=np.zeros_like(drawdown),
        where=running_peak > 0,
    )
    max_drawdown_pct = drawdown_pct.max(axis=1) * 100.0
    ending_balance = equity[:, -1]
    return equity, ending_balance, max_drawdown, max_drawdown_pct


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.01, 0.05, 0.50, 0.95, 0.99)
    labels = ("p01", "p05", "p50", "p95", "p99")
    return {
        label: float(value)
        for label, value in zip(labels, np.quantile(values, levels), strict=True)
    }


def run_simulation(
    trade_pnl: np.ndarray,
    config: SimulationConfig,
) -> SimulationResult:
    """Run the simulation and calculate risk-focused summary statistics."""

    pnl = _validated_pnl(trade_pnl)
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if config.ruin_floor >= config.initial_capital:
        raise ValueError("ruin_floor must be below initial_capital")

    pnl_paths = generate_pnl_paths(pnl, config)
    equity, ending_balance, max_drawdown, max_drawdown_pct = _path_metrics(
        pnl_paths,
        config.initial_capital,
    )

    path_minimum = equity.min(axis=1)
    terminal_returns = (ending_balance / config.initial_capital - 1.0) * 100.0
    tail_count = max(1, int(np.ceil(0.05 * config.simulations)))
    worst_terminal = np.sort(ending_balance)[:tail_count]

    summary: dict[str, object] = {
        "input": {
            "trade_count": int(pnl.size),
            "mean_trade_pnl": float(pnl.mean()),
            "median_trade_pnl": float(np.median(pnl)),
            "win_rate_pct": float((pnl > 0).mean() * 100.0),
        },
        "configuration": {
            "method": config.method,
            "simulations": config.simulations,
            "horizon_trades": int(pnl_paths.shape[1]),
            "block_size": config.block_size if config.method == "block-bootstrap" else None,
            "initial_capital": config.initial_capital,
            "ruin_floor": config.ruin_floor,
            "seed": config.seed,
        },
        "ending_balance": _quantiles(ending_balance),
        "terminal_return_pct": _quantiles(terminal_returns),
        "max_drawdown": _quantiles(max_drawdown),
        "max_drawdown_pct": _quantiles(max_drawdown_pct),
        "probability_of_loss_pct": float((ending_balance < config.initial_capital).mean() * 100.0),
        "probability_of_ruin_pct": float((path_minimum <= config.ruin_floor).mean() * 100.0),
        "expected_shortfall_5pct_balance": float(worst_terminal.mean()),
    }

    return SimulationResult(
        pnl_paths=pnl_paths,
        equity_paths=equity,
        ending_balance=ending_balance,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        summary=summary,
    )
