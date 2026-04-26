"""Persist last allocation between runs so we can show what to BUY/SELL."""

from __future__ import annotations
import json
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "last_allocation.json"


def load_prior() -> dict[str, float] | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text()).get("weights")
    except Exception:
        return None


def save_current(as_of: str, weights: dict[str, float]) -> None:
    STATE_FILE.write_text(json.dumps({"as_of": as_of, "weights": weights}, indent=2))
