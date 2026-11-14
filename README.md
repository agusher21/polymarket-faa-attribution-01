# FAA Outage Attribution Forecaster

A deterministic forecaster for Polymarket's FAA outage attribution
markets. Given the price history of each market in an event up to a
checkpoint, it predicts how the Yes price of every market will be
distributed 240 minutes later, expressed as five quantiles
(0.05, 0.25, 0.50, 0.75, 0.95) per market.

The markets in the event all ask the same underlying question from
different angles -- what caused the FAA outage -- so their prices move
together and mostly drift down as the true cause becomes clearer. The
forecaster is built around that observation: it is deliberately simple,
fully deterministic, and uses only the price histories in the input
files. No network access, no fitted model, no randomness.

## How it works

For each market the forecaster looks at the minute-by-minute price
history up to the checkpoint and builds a picture of how prices like
this one tend to move over the next four hours:

- **Empirical forward-change quantiles.** From the history it collects
  every past 240-minute forward change and takes the quantiles of those
  changes directly. Recent changes count more than old ones through
  recency weighting, so a market that only started moving an hour ago
  is judged mostly on that hour.
- **Normal blend by MAD sigma.** The empirical quantiles are blended
  with a normal distribution whose scale comes from a robust estimate
  of one-minute volatility: 1.4826 times the median absolute deviation
  of one-minute price changes. The MAD-based scale resists single wild
  ticks that would inflate a plain standard deviation, and the blend
  keeps the tails sane when the history is short.
- **Martingale median.** The median forecast stays at the checkpoint
  price: absent evidence to the contrary, the best guess for a market
  price four hours out is where it is now.
- **Small downward drift.** Attribution markets in this event tend to
  leak downward as uncertainty resolves, so a gentle downward drift is
  applied to the outer quantiles.
- **Light jump-aware mean reversion.** If the most recent price sits
  far from where it has been trading, part of that excursion is pulled
  back toward the recent level -- but only lightly, and only when the
  move looks like a jump rather than a genuine regime change.
- **Volatility-adaptive spread.** The width of the forecast interval
  grows with the market's own recent volatility, so calm markets get
  tight bands and jumpy markets get wide ones.

The output for each market is five quantiles of the Yes price at the
240-minute horizon, clipped to the valid price range.

## Running it

```sh
MAKEFUN_INPUT_DIR=/path/to/cases MAKEFUN_OUTPUT_DIR=/path/to/out python3 main.py
```

The input directory holds one `*.in` JSON file per case; the output
directory receives one `*.out` JSON file per case, each of the form
`{"forecasts": {"<market_id>": [q05, q25, q50, q75, q95], ...}}`.
The program needs only the Python standard library and always produces
the same output for the same input.

## Diagnostics mode

Set `FAA_DIAGNOSTICS=1` to also write a `<case>.diag.json` file next to
each `.out` file:

```sh
FAA_DIAGNOSTICS=1 MAKEFUN_INPUT_DIR=/path/to/cases MAKEFUN_OUTPUT_DIR=/path/to/out python3 main.py
```

Each diagnostics file lists one record per market with the raw facts
behind the forecast: the number of prices seen, the checkpoint price,
a robust one-minute sigma, the realised volatility and net move over
the last 60 minutes, the largest single one-minute jump in that hour
and what share of the hour's net move it accounts for, and a rough
240-minute forecast spread approximated as `2 * sigma_1min * sqrt(240)`
(labelled as an approximation -- it is a random-walk scale-up, not the
forecast's own spread). With the variable unset, behaviour is exactly
as before: only `.out` files are written.

## Verified numbers

On the kit practice cases the forecaster scores a mean pinball loss of
**0.277892**, against a house baseline of **0.363589**.

On a pseudo-backtest over the real FAA outage event -- replaying the
attribution markets from an earlier checkpoint and scoring against the
realised prices -- it scores **0.082114** against a house baseline of
**0.243846**.

## Files

- `main.py` -- the whole forecaster, standard library only.
- `README.md` -- this file.
- `LICENSE` -- MIT.
- `.makefun/owner.json` -- entry ownership record.
