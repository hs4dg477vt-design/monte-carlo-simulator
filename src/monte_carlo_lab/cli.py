from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .plotting import save_equity_fan_chart
from .simulator import SimulationConfig, run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test a trade-level P&L series with Monte Carlo simulation."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV containing trade P&L")
    parser.add_argument("--pnl-column", default="pnl_points")
    parser.add_argument("--point-value", type=float, default=1.0)
    parser.add_argument("--cost-per-trade", type=float, default=0.0)
    parser.add_argument(
        "--method",
        choices=("block-bootstrap", "iid-bootstrap", "shuffle"),
        default="block-bootstrap",
    )
    parser.add_argument("--simulations", type=int, default=5_000)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=5_000.0)
    parser.add_argument("--ruin-floor", type=float, default=1_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if args.pnl_column not in frame.columns:
        raise ValueError(f"missing P&L column: {args.pnl_column}")

    pnl = pd.to_numeric(frame[args.pnl_column], errors="coerce")
    if pnl.isna().any():
        raise ValueError("P&L column contains missing or non-numeric values")
    trade_pnl = pnl.to_numpy(dtype=float) * args.point_value - args.cost_per_trade

    config = SimulationConfig(
        simulations=args.simulations,
        horizon=args.horizon,
        method=args.method,
        block_size=args.block_size,
        initial_capital=args.initial_capital,
        ruin_floor=args.ruin_floor,
        seed=args.seed,
    )
    result = run_simulation(trade_pnl, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    metrics_path = args.output_dir / "path_metrics.csv"
    chart_path = args.output_dir / "equity_paths.png"

    summary_path.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "simulation": range(1, args.simulations + 1),
            "ending_balance": result.ending_balance,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
        }
    ).to_csv(metrics_path, index=False)

    observed = trade_pnl if result.pnl_paths.shape[1] == trade_pnl.size else None
    save_equity_fan_chart(
        result,
        chart_path,
        initial_capital=config.initial_capital,
        actual_trade_pnl=observed,
        seed=config.seed,
    )

    print(json.dumps(result.summary, indent=2))
    print(f"\nWrote {summary_path}, {metrics_path}, and {chart_path}")


if __name__ == "__main__":
    main()
