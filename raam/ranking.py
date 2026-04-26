"""Total Rank scoring and 5-asset selection per Giordano (2018).

Convention used here (rank 1 = best):
    Rank(M)  highest momentum  -> rank 1
    Rank(V)  lowest volatility -> rank 1
    Rank(C)  lowest avg corr   -> rank 1

Total Rank = wM*Rank(M) + wV*Rank(V) + wC*Rank(C) - T - M/x

T is the ATR breakout state (+2 long / -2 neutral). Subtracting T means a
long-trend signal *lowers* (improves) the total rank, which matches the
paper's intent. The paper's printed formula has "+ M/x" but that pushes
*higher* momentum to a *higher* (worse) total, contradicting the rest of
the paper -- treated here as a typo and flipped to "- M/x" so the
tiebreaker rewards higher momentum.

Selection: 5 lowest Total Ranks at month-end -> 20% each. Any of those
five with non-positive M is replaced with cash (SHY).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from .universe import RANKED_TICKERS, CASH_TICKER
from .indicators import (
    absolute_momentum,
    riskmetrics_volatility,
    average_relative_correlation,
    atr_breakout_signal,
)

DEFAULT_WEIGHTS = {"wM": 1.0 / 3, "wV": 1.0 / 3, "wC": 1.0 / 3}
DEFAULT_X = 100.0
TOP_N = 5
SLOT_WEIGHT = 1.0 / TOP_N  # 0.20


@dataclass
class Signal:
    as_of: pd.Timestamp
    weights: dict[str, float]
    table: pd.DataFrame  # one row per ranked ticker, sorted by total_rank asc
    cash_fallbacks: list[str] = field(default_factory=list)

    @property
    def held_tickers(self) -> list[str]:
        return [t for t, w in self.weights.items() if w > 0]


def _rank_series(s: pd.Series, *, ascending: bool) -> pd.Series:
    """Rank with 1 = best. ascending=True means smaller raw value -> rank 1."""
    return s.rank(method="min", ascending=ascending, na_option="bottom").astype(float)


def compute_signal(panel: dict[str, pd.DataFrame],
                   as_of: pd.Timestamp | str | None = None,
                   weights: dict[str, float] | None = None,
                   x: float = DEFAULT_X) -> Signal:
    """Compute the RAAM allocation as of `as_of` (defaults to latest available)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    closes = pd.DataFrame({t: panel[t]["Close"] for t in RANKED_TICKERS}).dropna(how="all")
    rets = closes.pct_change()

    # --- Per-asset daily series ---
    M = pd.DataFrame({t: absolute_momentum(closes[t]) for t in RANKED_TICKERS})
    V = pd.DataFrame({t: riskmetrics_volatility(closes[t]) for t in RANKED_TICKERS})
    C = average_relative_correlation(rets[RANKED_TICKERS])
    T = pd.DataFrame({t: atr_breakout_signal(panel[t]) for t in RANKED_TICKERS})

    # Pick the as-of date.
    common_idx = M.index.intersection(V.index).intersection(C.index).intersection(T.index)
    if as_of is None:
        as_of_ts = common_idx.max()
    else:
        as_of_ts = pd.Timestamp(as_of)
        common_idx = common_idx[common_idx <= as_of_ts]
        if len(common_idx) == 0:
            raise ValueError(f"No data on/before {as_of_ts}")
        as_of_ts = common_idx.max()

    m_row = M.loc[as_of_ts].astype(float)
    v_row = V.loc[as_of_ts].astype(float)
    c_row = C.loc[as_of_ts].astype(float)
    t_row = T.loc[as_of_ts].astype(float)

    rank_m = _rank_series(m_row, ascending=False)  # high mom -> rank 1
    rank_v = _rank_series(v_row, ascending=True)   # low vol  -> rank 1
    rank_c = _rank_series(c_row, ascending=True)   # low corr -> rank 1

    total = (
        w["wM"] * rank_m
        + w["wV"] * rank_v
        + w["wC"] * rank_c
        - t_row
        - m_row / x  # tiebreaker: higher M -> lower (better) total
    )

    table = pd.DataFrame({
        "M": m_row, "V": v_row, "C": c_row, "T": t_row.astype(int),
        "Rank(M)": rank_m, "Rank(V)": rank_v, "Rank(C)": rank_c,
        "total_rank": total,
    }).sort_values("total_rank")

    top = table.head(TOP_N).copy()
    top["selected"] = True
    table["selected"] = table.index.isin(top.index)

    cash_fallbacks = []
    weights_out: dict[str, float] = {t: 0.0 for t in RANKED_TICKERS}
    weights_out[CASH_TICKER] = 0.0
    for tkr in top.index:
        if top.loc[tkr, "M"] > 0:
            weights_out[tkr] = SLOT_WEIGHT
        else:
            cash_fallbacks.append(tkr)
            weights_out[CASH_TICKER] += SLOT_WEIGHT

    return Signal(as_of=as_of_ts, weights=weights_out,
                  table=table, cash_fallbacks=cash_fallbacks)
