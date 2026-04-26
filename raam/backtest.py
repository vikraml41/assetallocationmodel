"""Monthly rebalance backtest. Allocation set on month-end, applied next month."""

from __future__ import annotations
import numpy as np
import pandas as pd

from .universe import RANKED_TICKERS, CASH_TICKER
from .ranking import compute_signal


def _month_end_dates(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(1, index=idx)
    return list(s.resample("ME").last().dropna().index.intersection(idx))


def run_backtest(panel: dict[str, pd.DataFrame], start: str = "2004-07-01",
                 end: str | None = None, weights: dict[str, float] | None = None,
                 verbose: bool = False) -> dict:
    closes = pd.DataFrame({t: panel[t]["Close"] for t in [*RANKED_TICKERS, CASH_TICKER]}).dropna()
    closes = closes.loc[(closes.index >= pd.Timestamp(start)) &
                        (closes.index <= (pd.Timestamp(end) if end else closes.index.max()))]
    if closes.empty:
        raise RuntimeError("No overlapping price data in requested window.")

    rets = closes.pct_change().fillna(0.0)
    rebal_dates = _month_end_dates(closes.index)

    nav = pd.Series(index=closes.index, dtype=float)
    nav.iloc[0] = 100.0
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]] = []
    current_w = pd.Series(0.0, index=closes.columns)
    current_w[CASH_TICKER] = 1.0  # start in cash until first signal

    next_rebal_idx = 0
    for i in range(1, len(closes.index)):
        date = closes.index[i]
        # Apply yesterday's weights to today's returns.
        nav.iloc[i] = nav.iloc[i - 1] * (1.0 + (current_w * rets.iloc[i]).sum())

        # Drift weights.
        gross = current_w * (1.0 + rets.iloc[i])
        s = gross.sum()
        if s > 0:
            current_w = gross / s

        # Rebalance at month-end using signal computed *as of* this date.
        if next_rebal_idx < len(rebal_dates) and date == rebal_dates[next_rebal_idx]:
            try:
                sig = compute_signal(panel, as_of=date, weights=weights)
                new_w = pd.Series(sig.weights).reindex(closes.columns).fillna(0.0)
                if new_w.sum() == 0:
                    new_w[CASH_TICKER] = 1.0
                current_w = new_w
                weights_history.append((date, sig.weights.copy()))
                if verbose:
                    held = {k: v for k, v in sig.weights.items() if v > 0}
                    print(f"{date.date()}  {held}")
            except Exception as e:
                if verbose:
                    print(f"{date.date()}  signal error: {e}")
            next_rebal_idx += 1

    nav = nav.dropna()
    return {
        "nav": nav,
        "returns": nav.pct_change().fillna(0.0),
        "weights_history": weights_history,
        "stats": _stats(nav),
    }


def _stats(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / ann_vol if ann_vol > 0 else np.nan
    rolling_max = nav.cummax()
    dd = (nav / rolling_max) - 1.0
    return {
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1.0,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": dd.min(),
        "start": nav.index[0].date().isoformat(),
        "end": nav.index[-1].date().isoformat(),
    }
