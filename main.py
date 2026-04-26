"""RAAM (Ranked Asset Allocation Model) CLI.

Subcommands:
  signal      Compute today's allocation and print it. No notification, no state write.
  rebalance   Compute month-end allocation, send notification, persist state.
              Idempotent: re-running on the same as-of date will re-notify.
  backtest    Run the monthly backtest from a start date.

Examples:
  python main.py signal
  python main.py signal --as-of 2017-11-30
  python main.py rebalance --channel stdout
  python main.py backtest --start 2005-01-01 --end 2017-11-30
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from raam.universe import RANKED_TICKERS, CASH_TICKER
from raam.data import load_panel
from raam.ranking import compute_signal
from raam.backtest import run_backtest
from raam.notify import notify, format_signal_text
from raam import state


ALL_TICKERS = [*RANKED_TICKERS, CASH_TICKER]


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="raam")
    sub = p.add_subparsers(dest="cmd", required=True)

    sig = sub.add_parser("signal", help="Compute and print the latest allocation.")
    sig.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to latest.")
    sig.add_argument("--start", default="2003-01-01", help="History start for indicators.")
    sig.add_argument("--refresh", action="store_true", help="Bypass cache.")

    rb = sub.add_parser("rebalance", help="Compute, notify, persist state.")
    rb.add_argument("--as-of", default=None)
    rb.add_argument("--start", default="2003-01-01")
    rb.add_argument("--refresh", action="store_true")
    rb.add_argument("--channel", default=None,
                    help="Override NOTIFY_CHANNEL. Use 'stdout' to skip email.")

    bt = sub.add_parser("backtest", help="Run the monthly backtest.")
    bt.add_argument("--start", default="2004-07-01")
    bt.add_argument("--end", default=None)
    bt.add_argument("--refresh", action="store_true")
    bt.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def _load(start, refresh):
    return load_panel(ALL_TICKERS, start=start, refresh=refresh)


def cmd_signal(args):
    panel = _load(args.start, args.refresh)
    sig = compute_signal(panel, as_of=args.as_of)
    print(format_signal_text(sig, prior_weights=state.load_prior()))


def cmd_rebalance(args):
    panel = _load(args.start, args.refresh)
    sig = compute_signal(panel, as_of=args.as_of)
    prior = state.load_prior()
    notify(sig, prior_weights=prior, channel=args.channel)
    state.save_current(sig.as_of.date().isoformat(), sig.weights)
    print(f"\nState saved. Next run will diff against this allocation.")


def cmd_backtest(args):
    panel = _load("2003-01-01", args.refresh)
    res = run_backtest(panel, start=args.start, end=args.end, verbose=args.verbose)
    s = res["stats"]
    print()
    print(f"  Period:        {s['start']}  ->  {s['end']}")
    print(f"  Total return:  {s['total_return']*100:8.2f}%")
    print(f"  CAGR:          {s['cagr']*100:8.2f}%")
    print(f"  Ann vol:       {s['ann_vol']*100:8.2f}%")
    print(f"  Sharpe:        {s['sharpe']:8.2f}")
    print(f"  Max drawdown:  {s['max_drawdown']*100:8.2f}%")
    print(f"  Final NAV:     {res['nav'].iloc[-1]:8.2f}  (start = 100)")


COMMANDS = {"signal": cmd_signal, "rebalance": cmd_rebalance, "backtest": cmd_backtest}


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
