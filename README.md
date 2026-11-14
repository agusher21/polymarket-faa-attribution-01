# Polymarket FAA Outage Attribution Forecaster

Entry for make.fun competition `polymarket-faa-attribution-01`: forecast the
Yes price of each of the four Polymarket FAA outage attribution markets
(Iran, a foreign actor, a hack, an AI sandbox escape) at every minute of
the 4 hours after each checkpoint, as five quantiles (5%, 25%, 50%, 75%,
95%), for the lowest mean pinball loss.

## Program

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `*.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`), in the
`{"forecasts": [{"id", "q": [[5 numbers] x 240]}]}` format.

Method, per market, in price-change units added to the checkpoint price:

1. **Empirical forward changes.** At anchor horizons (1, 2, 3, 5, 8, 13, 21,
   34, 55, 89, 144, 180, 233, 240 minutes), the weighted quantiles of every
   past forward change, weighted by recency (half-life ensemble 90/120/180
   minutes averaged).
2. **Normal blend.** Where history is thin, blend toward a normal forecast
   whose 1-minute sigma is the market's robust 1-minute MAD (floored at
   0.0002, capped at 0.05). Blend weight `m / (m + 300)`.
3. **Martingale median.** Recentre so the median change is zero; drift is
   added back explicitly, not smuggled in through a noisy empirical median.
4. **Small downward drift** (-0.1 sigma per sqrt-minute): attribution
   markets decay slightly with no news.
5. **Light jump-aware mean reversion.** A saturating pull against the last
   hour's slope, scaled by the jump fraction of that move, up-moves reverting
   harder than down-moves (0.2 vs 0.05), capped at 0.04. Deliberately
   light: the NVIDIA practice markets are strongly mean-reverting, but the
   FAA markets' full-window 60-min autocorrelation is positive (momentum),
   so strong reversion loses badly on the actual final markets.
6. **Volatility-adaptive width.** The spread widens when the last hour's
   realised volatility exceeds the full-history scale (clipped [0.7, 1.5])
   and is reshaped (inner quantiles 0.85, outer 1.2, overall 0.85).

`FAA_PARAMS` (JSON, env var) overrides any tuned constant for local
grading only; the grader never sets it.

## Local results

Practice instance (the six NVIDIA cases the dev phase grades on,
153,600 scored items): **pinball loss 0.277892 vs house baseline 0.363589**
— beats the baseline on every case. Runtime ~2.4 s per case (cap 8 s).

FAA pseudo-backtest (four cases cut from the live FAA markets' own
1-minute history, checkpoints 2026-11-14 02:00/06:00/09:00/10:00 UTC,
19,200 items): **pinball loss 0.082114 vs house baseline 0.243846** —
beats the baseline on every case, 90% coverage 0.984.

Backtest scripts live outside this repo (they fetch live Polymarket data):
`~/workspace/makefun/faa-work/faa_backtest.py` and `regime_check.py`.

## License

MIT — see `LICENSE`.
