"""The four signal components from Giordano (2018).

M  Absolute Momentum            4-month ROC on daily closes
V  Volatility Model              RiskMetrics EWMA variance (lambda=0.94),
                                 10-day SMA on annualized stdev
C  Average Relative Correlation  Mean pairwise corr of daily returns vs the
                                 rest of the ranked universe, 4-month window
T  ATR Trend/Breakout            42-period ATR + Donchian-style channel.
                                 +2 long, -2 neutral/short, 0 before first signal.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS_4M = 84
EWMA_LAMBDA = 0.94
VOL_SMOOTH = 10
ATR_LEN = 42
DONCHIAN_UP = 63
DONCHIAN_DN = 105


def absolute_momentum(close: pd.Series, lookback: int = TRADING_DAYS_4M) -> pd.Series:
    """4-month ROC. Returned in percent (e.g. 5.0 == +5%)."""
    return (close / close.shift(lookback) - 1.0) * 100.0


def riskmetrics_volatility(close: pd.Series,
                           lam: float = EWMA_LAMBDA,
                           smooth: int = VOL_SMOOTH) -> pd.Series:
    """Annualized EWMA volatility (RiskMetrics) with SMA smoothing.

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2
    Returned in percent (annualized stdev * 100), then 10-day SMA.
    """
    r = np.log(close / close.shift(1)).dropna()
    var = pd.Series(index=r.index, dtype=float)
    if len(r) == 0:
        return var
    # Seed with sample variance of the first ~30 returns; small bias dies out fast.
    seed = r.iloc[: min(30, len(r))].var()
    prev = float(seed) if pd.notna(seed) else float(r.iloc[0] ** 2)
    for ts, ret in r.items():
        prev = lam * prev + (1.0 - lam) * (ret ** 2)
        var.loc[ts] = prev
    ann_vol_pct = np.sqrt(var * 252.0) * 100.0
    return ann_vol_pct.rolling(smooth, min_periods=1).mean()


def average_relative_correlation(returns: pd.DataFrame,
                                 window: int = TRADING_DAYS_4M) -> pd.DataFrame:
    """For each column, mean pairwise correlation with every other column over
    a rolling window. Returns a DataFrame indexed like `returns`.
    """
    out = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    cols = list(returns.columns)
    for i in range(window, len(returns) + 1):
        w = returns.iloc[i - window:i]
        if w.dropna(how="any").shape[0] < window // 2:
            continue
        corr = w.corr()
        np.fill_diagonal(corr.values, np.nan)
        out.iloc[i - 1] = corr.mean(axis=1)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    """Wilder's ATR (RMA / EMA with alpha = 1/length)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def atr_breakout_signal(df: pd.DataFrame,
                        atr_len: int = ATR_LEN,
                        up_lookback: int = DONCHIAN_UP,
                        dn_lookback: int = DONCHIAN_DN) -> pd.Series:
    """Trend signal in {-2, 0, +2}.

    The paper's literal band formula contains typos. We implement the
    financially-coherent Donchian + ATR variant that matches Figure 10:

        UpperBand_t = HighestHigh(up_lookback) + ATR(atr_len)
        LowerBand_t = LowestLow(dn_lookback)  - ATR(atr_len)

    Going long (+2) on the next session when today's high > yesterday's upper
    band; flipping neutral/short (-2) when today's low < yesterday's lower
    band. The state is sticky between events. Signal value 0 only before the
    first ever breakout (model not yet initialized).
    """
    a = atr(df, atr_len)
    upper = df["High"].rolling(up_lookback).max() + a
    lower = df["Low"].rolling(dn_lookback).min() - a
    upper_prev = upper.shift(1)
    lower_prev = lower.shift(1)

    long_break = df["High"] > upper_prev
    short_break = df["Low"] < lower_prev

    state = pd.Series(0, index=df.index, dtype=int)
    cur = 0
    for i, ts in enumerate(df.index):
        if long_break.iloc[i]:
            cur = 2
        elif short_break.iloc[i]:
            cur = -2
        state.iloc[i] = cur
    # Signal applies on the *next* session.
    return state.shift(1).fillna(0).astype(int)
