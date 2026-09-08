import numpy as np
import pytest

from monte_carlo_lab.simulator import (
    SimulationConfig,
    generate_pnl_paths,
    run_simulation,
)


def test_block_bootstrap_is_reproducible() -> None:
    pnl = np.array([10.0, -5.0, 7.0, -2.0, 4.0])
    config = SimulationConfig(simulations=20, horizon=12, block_size=2, seed=7)

    first = generate_pnl_paths(pnl, config)
    second = generate_pnl_paths(pnl, config)

    assert first.shape == (20, 12)
    np.testing.assert_array_equal(first, second)


def test_shuffle_preserves_terminal_pnl_but_changes_path_risk() -> None:
    pnl = np.array([100.0, -90.0, 40.0, -10.0, 15.0])
    config = SimulationConfig(
        simulations=50,
        method="shuffle",
        initial_capital=1_000.0,
        ruin_floor=0.0,
    )
    result = run_simulation(pnl, config)

    np.testing.assert_allclose(result.ending_balance, 1_000.0 + pnl.sum())
    assert np.unique(result.max_drawdown).size > 1


def test_ruin_probability_detects_floor_breaches() -> None:
    pnl = np.array([-100.0, -100.0, 10.0])
    config = SimulationConfig(
        simulations=500,
        horizon=10,
        method="iid-bootstrap",
        initial_capital=250.0,
        ruin_floor=100.0,
        seed=3,
    )
    result = run_simulation(pnl, config)

    probability = result.summary["probability_of_ruin_pct"]
    assert isinstance(probability, float)
    assert 0.0 < probability <= 100.0


def test_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        run_simulation(np.array([1.0, np.nan]), SimulationConfig())
