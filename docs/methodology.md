# Methodology

## Goal

Historical backtests provide one realized ordering of trades. This project asks a
different question: how sensitive are the strategy's outcomes to the order and
local clustering of wins and losses?

## Resampling methods

### Block bootstrap

The default method samples contiguous blocks of trades with replacement until a
new path reaches the requested horizon. Sampling blocks, rather than isolated
trades, preserves some short-run dependence such as clustered wins, losses, and
volatility. Block size is a modeling choice and should be tested rather than
optimized around one favorable result.

### IID bootstrap

This method samples individual trades with replacement. It is simple and allows
terminal P&L to vary, but assumes each trade is independent and identically
distributed. That assumption is often unrealistic when market regimes cluster.

### Trade shuffle

This method permutes the observed trades without replacement. It is useful for
studying sequence risk and drawdown sensitivity. It cannot estimate uncertainty
in total P&L because every path contains exactly the same trades, so all paths
end with the same cumulative P&L.

## Reported metrics

- Terminal balance and return percentiles
- Maximum dollar and percentage drawdown percentiles
- Probability of ending below initial capital
- Probability of touching a user-defined ruin floor
- Five-percent expected shortfall of terminal balance

## Limitations

- Simulations inherit any bias, overfitting, or execution errors in the input ledger.
- A bootstrap assumes the historical sample is informative about future outcomes.
- Transaction costs must be included through `--cost-per-trade` if the ledger is gross.
- The model does not simulate order-book fills, latency, changing position size, or market impact.
- Results are research diagnostics, not forecasts or investment advice.
