"""Daily OHLC loader with on-disk parquet cache."""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.csv"


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index_label="Date")


def load_ohlc(ticker: str, start: str = "2003-01-01", end: str | None = None,
              refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns Open, High, Low, Close (split/div-adjusted)."""
    path = _cache_path(ticker)
    if path.exists() and not refresh:
        df = _read_cache(path)
        last = df.index.max()
        # Refresh tail if cache is more than a day stale and we're requesting recent data.
        target_end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        if last < target_end - pd.Timedelta(days=2):
            tail = _download(ticker, last - pd.Timedelta(days=10), end)
            if tail is not None and not tail.empty:
                merged = pd.concat([df, tail])
                df = merged.loc[~merged.index.duplicated(keep="last")].sort_index()
                _write_cache(df, path)
        return _slice(df, start, end)

    df = _download(ticker, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    _write_cache(df, path)
    return _slice(df, start, end)


def _download(ticker: str, start, end) -> pd.DataFrame | None:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()
    return df


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def load_panel(tickers: list[str], start: str = "2003-01-01",
               end: str | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {t: load_ohlc(t, start, end, refresh=refresh) for t in tickers}


def closes_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({t: df["Close"] for t, df in panel.items()}).dropna(how="all")
