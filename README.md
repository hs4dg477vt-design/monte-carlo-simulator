# Monte Carlo Simulator

This is a Python project I made to test how a trading strategy might perform if
its trades happened in different sequences. It reads a CSV of trade profits and
losses, creates randomized equity paths, and measures the results.

The simulator can calculate:

- Ending balance ranges
- Maximum drawdown
- Probability of losing money
- Probability of reaching a chosen account floor
- Expected shortfall

It has three simulation options: block bootstrap, independent bootstrap, and
trade-order shuffle. I normally use block bootstrap because it keeps short
groups of trades together instead of assuming every trade is independent.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

monte-carlo-simulator \
  --input sample_data/anonymized_trade_pnl.csv \
  --pnl-column pnl_points \
  --point-value 2 \
  --cost-per-trade 1.50 \
  --method block-bootstrap \
  --simulations 5000 \
  --block-size 5 \
  --initial-capital 5000 \
  --ruin-floor 1000
```

To run the tests:

```bash
pytest
```

The sample CSV only contains anonymized trade P&L. It does not include account
information, market data, or the rules used to generate the trades.

This is an educational project, not investment advice.
