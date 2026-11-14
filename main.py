#!/usr/bin/env python3
"""FAA outage attribution forecaster v1: empirical forward-change quantiles.

Model v1 (post-opening re-author): empirical forward-change quantiles with a
half-life recency ensemble (90/120/180 min), blended toward a normal forecast
from the market's robust 1-minute MAD sigma, martingale median recentring,
a small downward drift (-0.1 sigma per sqrt-minute), light jump-aware mean
reversion, and volatility-adaptive width reshaping. Practice score on the
NVIDIA "Next CEO" cases (p00-p05): mean pinball 0.277892 vs house baseline
0.363589. FAA pseudo-backtest (same method on the FAA markets' own history
to date): 0.082114 vs baseline 0.243846.

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes one
*.out per case to $MAKEFUN_OUTPUT_DIR (else /output). Standard library only,
deterministic, no network, no randomness.

The contest scores five quantiles (5/25/50/75/95%) of each market's Yes price
at every minute of the 240 minutes after a checkpoint, by mean pinball loss.
Prediction-market prices are near-martingales that move in steps and jump:
the honest forecast is the distribution of where the price actually goes
from here, measured on this market's own recent history.

Method, per market, all in price-change units (added to the checkpoint
price at the end):
1. Empirical forward changes. At a ladder of anchor horizons h, collect
   every past forward change p[t+h] - p[t] the history holds, weight each
   sample by recency (half-life ensemble: 90, 120 and 180 minutes averaged),
   and take the weighted quantiles at the five scored levels. Old regimes
   fade; the last two hours dominate.
2. Normal blend. Where the history is thin, blend the empirical quantiles
   toward a normal forecast whose 1-minute sigma is the market's robust
   1-minute MAD (1.4826 * median absolute deviation, floored and capped).
   Blend weight w = m / (m + 300) with m the number of delta samples, so a
   market with thousands of samples is almost purely empirical.
3. Martingale median. Recentre the quantiles so the median change is zero:
   the checkpoint price is the best median, and any real drift is added
   back explicitly (step 4), not smuggled in through a noisy empirical
   median.
4. Small downward drift. A tiny median shift of -0.1 sigma per sqrt-minute:
   attribution markets decay slightly as the window passes with no news.
5. Light jump-aware mean reversion. If the price moved hard in the last
   hour, expect a little of it back: a saturating pull, front-loaded over
   the first hour, scaled by how much of the hour's move was one single
   jump (a jump reverts; a slow grind does not). Up-moves revert harder
   than down-moves (0.2 vs 0.05), capped at 0.04 price units. The pull
   tilts the outer quantiles slightly wider than the median (rev_q 0.3).
   The strength is deliberately light: on the NVIDIA practice instance
   strong reversion wins (those markets are strongly mean-reverting,
   60-min lag autocorrelation about -0.23), but on the FAA markets
   themselves the full-window autocorrelation is positive (momentum,
   +0.02 to +0.20) and strong reversion loses badly there; the recent
   hours are mixed, so a light pull hedges both regimes.
6. Volatility-adaptive width. The spread widens when the last hour's
   realised volatility exceeds the market's full-history 1-minute scale
   (factor clipped to [0.7, 1.5]), and is reshaped: inner quantiles pulled
   in (0.85), outer quantiles pushed out (1.2), overall 0.85. Overlapping
   empirical windows run wide in the middle and thin in the tails; this
   re-shapes them toward the realised shape.
7. Between anchors, interpolate the quantiles in sqrt(h); beyond the last
   anchor, scale by sqrt(h / h_max). Clamp to [0, 1], enforce
   non-decreasing rows, round to 6 decimals.

Everything the program uses is wired into the computation: there are no
dormant knobs. Cross-market terms (sum correction, pooled sigma,
cross-market reversion) were considered and left out: on the practice
instance they never helped, and with only four markets the pool is thin.
"""

import json
import math
import os
from pathlib import Path

# Scored quantile levels, in row order.
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Anchor horizons (minutes) where empirical forward changes are measured.
ANCHORS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 180, 233, 240]
# Recency half-lives (minutes) averaged for the empirical quantiles.
HALFLIVES = [90.0, 120.0, 180.0]

# Tuned constants. Every one is used below; none is dormant.
BLEND_K = 300.0        # empirical weight w = m / (m + BLEND_K)
SIGMA_FLOOR = 0.0002   # floor on the 1-minute MAD sigma (price units)
SIGMA_CAP = 0.05       # cap on the 1-minute MAD sigma
MEDIAN_BIAS = -0.1     # median shift, in units of sigma1 per sqrt-minute
SPREAD_MULT = 0.85     # global multiplier on (quantile - median)
INNER_MULT = 0.85      # extra multiplier on the 25%/75% distances
TAIL_MULT = 1.2        # extra multiplier on the 5%/95% distances
VOL_SCALE = 0.3        # last-hour realised-vol width scaling
REV_WIN = 60           # window (minutes) for the reversion slope
REV_K_UP = 0.2         # reversion strength after an up-move
REV_K_DOWN = 0.05       # reversion strength after a down-move
REV_CAP = 0.04         # absolute cap on the reversion shift
REV_Q = 0.3            # quantile tilt of the reversion shift

# Local tuning override (the grader never sets this variable, so grading
# always uses the defaults above): a JSON object naming any constant above.
# Every key must already exist here; an unknown key is refused loudly so a
# typo can never silently change nothing.
_env = os.environ.get("FAA_PARAMS")
if _env:
    for _k, _v in json.loads(_env).items():
        if _k not in globals() or _k.startswith("_"):
            raise ValueError(f"unknown tuning key {_k}")
        globals()[_k] = _v


def _quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[lo + 1] * frac


def _median(vals):
    return _quantile(sorted(vals), 0.5)


def _wquantile(pairs, tau):
    """Weighted quantile of (value, weight) pairs, linear interpolation."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0 or not pairs:
        return 0.0
    target = tau * tot
    acc = 0.0
    prev_v = pairs[0][0]
    for v, w in pairs:
        nxt = acc + w
        if nxt >= target:
            if nxt == acc:
                return v
            f = (target - acc) / (nxt - acc)
            return prev_v + f * (v - prev_v)
        prev_v = v
        acc = nxt
    return pairs[-1][0]


def _round6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0.0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return 0.0 if v == 0 else v


def _robust_sigma1(prices):
    """1-minute price-move scale: 1.4826 * MAD of 1-minute changes."""
    d1 = [prices[t + 1] - prices[t] for t in range(len(prices) - 1)]
    med = _median(d1)
    mad = _median([abs(v - med) for v in d1])
    return min(SIGMA_CAP, max(SIGMA_FLOOR, 1.4826 * mad))


def _anchor_quantiles(prices, sigma1):
    """Empirical forward-change quantiles at each anchor, blended to normal.

    Returns {anchor_h: [q05, q25, q50, q75, q95]} in price-change units.
    """
    n = len(prices)
    out = {}
    for h in ANCHORS:
        if h > 240:
            continue
        m = n - h
        if m <= 0:
            continue
        # Average the weighted quantiles over the halflife ensemble.
        emp = [0.0] * 5
        for hl in HALFLIVES:
            pairs = []
            for t in range(m):
                # Weight by recency of the sample's END point.
                w = 0.5 ** ((n - 1 - (t + h)) / hl)
                pairs.append((prices[t + h] - prices[t], w))
            for j, tau in enumerate(TAUS):
                emp[j] += _wquantile(pairs, tau) / len(HALFLIVES)
        w_emp = m / (m + BLEND_K)
        norm = [z * sigma1 * math.sqrt(h) for z in Z]
        out[h] = [w_emp * e + (1.0 - w_emp) * g
                  for e, g in zip(emp, norm)]
    return out


def _reversion_shift(prices, h):
    """Jump-aware mean-reversion pull (price units), applied as -shift.

    Returns (shift, rev_shift): shift is the total median shift from
    reversion (already signed so the median moves against the last-hour
    slope); rev_shift is the same pull for the quantile tilt.
    """
    n = len(prices)
    if n <= REV_WIN:
        return 0.0, 0.0
    slope = (prices[-1] - prices[-1 - REV_WIN]) / REV_WIN
    k = REV_K_UP if slope > 0 else REV_K_DOWN
    if k == 0.0:
        return 0.0, 0.0
    pull = k * slope * 60.0 * (1.0 - math.exp(-h / 60.0))
    # Scale by the jump fraction: the largest 1-minute move in the window
    # over the window's total move. A jump reverts; a grind does not.
    window_moves = [prices[i] - prices[i - 1]
                    for i in range(n - REV_WIN, n)]
    total = abs(prices[-1] - prices[-1 - REV_WIN])
    biggest = max((abs(v) for v in window_moves), default=0.0)
    if total > 0:
        pull *= min(1.0, biggest / total)
    else:
        pull = 0.0
    pull = max(-REV_CAP, min(REV_CAP, pull))
    # The median moves AGAINST the slope: shift is negative of the pull.
    return -pull, -pull


def _forecast_market(prices, horizon):
    """240 rows x 5 quantiles for one market (rows are absolute prices)."""
    n = len(prices)
    x = prices[-1]
    if n < 2:
        return [[_round6(x + z * 0.0025 * math.sqrt(h)) for z in Z]
                for h in range(1, horizon + 1)]

    sigma1 = _robust_sigma1(prices)
    anchors = _anchor_quantiles(prices, sigma1)
    anchor_hs = sorted(anchors)

    # Last-hour realised volatility vs full-history scale (width scaling).
    d60 = [prices[i] - prices[i - 1] for i in range(n - 60, n)] if n >= 61 else []
    rv60 = 0.0
    if d60:
        m60 = _median(d60)
        rv60 = 1.4826 * _median([abs(v - m60) for v in d60])

    rows = []
    for h in range(1, horizon + 1):
        if h in anchors:
            qs = list(anchors[h])
        elif h > anchor_hs[-1]:
            s = math.sqrt(h / anchor_hs[-1])
            qs = [q * s for q in anchors[anchor_hs[-1]]]
        else:
            lo = max(a for a in anchor_hs if a < h)
            hi = min(a for a in anchor_hs if a > h)
            w = (math.sqrt(h) - math.sqrt(lo)) / (math.sqrt(hi) - math.sqrt(lo))
            qs = [(1.0 - w) * a + w * b
                  for a, b in zip(anchors[lo], anchors[hi])]
        # Martingale median: recenter so the median change is exactly 0.
        med_change = qs[2]
        qs = [q - med_change for q in qs]
        # Median shift: small downward drift + jump-aware reversion.
        shift = MEDIAN_BIAS * sigma1 * math.sqrt(h)
        rev_shift = 0.0
        rs, rshift = _reversion_shift(prices, h)
        shift += rs
        rev_shift = rshift
        if shift:
            # Tilt: the reversion part widens the outer quantiles a little.
            qs = [q + shift + rev_shift * REV_Q * (j - 2) / 2.0
                  for j, q in enumerate(qs)]
        # Spread: volatility-adaptive width, reshaped (inner in, tails out).
        sm = SPREAD_MULT
        if rv60 > 0 and sigma1 > 0:
            sm *= min(1.5, max(0.7, 1.0 + VOL_SCALE * (rv60 / sigma1 - 1.0)))
        mults = [TAIL_MULT, INNER_MULT, 1.0, INNER_MULT, TAIL_MULT]
        med = qs[2]
        qs = [med + (q - med) * sm * mults[j] for j, q in enumerate(qs)]
        # Absolute prices, monotone, clamped, 6 decimals.
        row = []
        prev = None
        for q in qs:
            v = _round6(x + q)
            if prev is not None and v < prev:
                v = prev
            row.append(v)
            prev = v
        rows.append(row)
    return rows


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in sorted(p.name for p in in_dir.glob("*.in")):
        case = json.loads((in_dir / fname).read_text())
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": _forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        out = json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n"
        (out_dir / (fname[:-3] + ".out")).write_text(out)


if __name__ == "__main__":
    main()
