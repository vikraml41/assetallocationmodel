"""Notification layer. Email via SMTP, with a stdout fallback for testing.

Add another channel by writing a function with the same signature
(subject: str, body_text: str, body_html: str | None = None) and dispatching
on NOTIFY_CHANNEL.
"""

from __future__ import annotations
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Callable

import pandas as pd

from .ranking import Signal
from .universe import NAMES, ASSET_CLASS, CASH_TICKER


# ---------- formatting ----------

def format_signal_text(sig: Signal, *, prior_weights: dict[str, float] | None = None) -> str:
    lines = [
        f"RAAM rebalance signal as of {sig.as_of.date()}",
        "",
        "Target portfolio:",
    ]
    held = {k: v for k, v in sig.weights.items() if v > 0}
    if not held:
        lines.append("  (all cash)")
    else:
        for tkr, wt in held.items():
            tag = "" if tkr != CASH_TICKER else "  [CASH]"
            lines.append(f"  {tkr:<5}  {wt*100:5.1f}%   {ASSET_CLASS[tkr]:<22}  {NAMES[tkr]}{tag}")

    if sig.cash_fallbacks:
        lines += ["", "Selected but routed to cash (negative momentum):"]
        for t in sig.cash_fallbacks:
            lines.append(f"  {t}  ({NAMES[t]})")

    if prior_weights:
        lines += ["", "Trades from prior allocation:"]
        all_t = sorted(set(prior_weights) | set(sig.weights))
        for t in all_t:
            old = prior_weights.get(t, 0.0)
            new = sig.weights.get(t, 0.0)
            if abs(new - old) > 1e-9:
                arrow = "BUY " if new > old else "SELL"
                lines.append(f"  {arrow} {t:<5}  {old*100:5.1f}%  ->  {new*100:5.1f}%")

    lines += ["", "Top of ranking table:"]
    lines.append(sig.table.head(8).to_string(float_format=lambda v: f"{v:7.2f}"))
    return "\n".join(lines)


def format_signal_html(sig: Signal, *, prior_weights: dict[str, float] | None = None) -> str:
    held = {k: v for k, v in sig.weights.items() if v > 0}
    rows = "".join(
        f"<tr><td><b>{t}</b></td><td>{w*100:.1f}%</td>"
        f"<td>{ASSET_CLASS[t]}</td><td>{NAMES[t]}</td></tr>"
        for t, w in held.items()
    ) or "<tr><td colspan='4'><i>All cash</i></td></tr>"

    trades_block = ""
    if prior_weights:
        diffs = []
        for t in sorted(set(prior_weights) | set(sig.weights)):
            old = prior_weights.get(t, 0.0)
            new = sig.weights.get(t, 0.0)
            if abs(new - old) > 1e-9:
                arrow = "BUY" if new > old else "SELL"
                color = "#1a7f37" if new > old else "#b91c1c"
                diffs.append(
                    f"<tr><td style='color:{color}'><b>{arrow}</b></td>"
                    f"<td>{t}</td><td>{old*100:.1f}%</td><td>&rarr;</td>"
                    f"<td>{new*100:.1f}%</td></tr>"
                )
        if diffs:
            trades_block = (
                "<h3>Trades</h3><table cellpadding='4' style='border-collapse:collapse'>"
                + "".join(diffs) + "</table>"
            )

    table_html = sig.table.head(8).to_html(float_format=lambda v: f"{v:.2f}",
                                           border=0, classes="rk")
    return f"""<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif">
<h2>RAAM Rebalance Signal &mdash; {sig.as_of.date()}</h2>
<h3>Target Portfolio</h3>
<table cellpadding="6" style="border-collapse:collapse">
<thead><tr style="background:#f0f0f0"><th>Ticker</th><th>Weight</th><th>Class</th><th>Name</th></tr></thead>
<tbody>{rows}</tbody></table>
{trades_block}
<h3>Top of Ranking</h3>
{table_html}
<p style="color:#888;font-size:12px">Computed by the Ranked Asset Allocation Model
(Giordano, 2018). Apply at next session's open.</p>
</body></html>"""


# ---------- channels ----------

def _send_email(subject: str, body_text: str, body_html: str | None = None) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASS"]
    sender = os.environ.get("SMTP_FROM", user)
    recipient = os.environ["SMTP_TO"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body_text)
    if body_html:
        msg.add_alternative(body_html, subtype="html")

    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls(context=ctx)
        smtp.login(user, password)
        smtp.send_message(msg)


def _print_to_stdout(subject: str, body_text: str, body_html: str | None = None) -> None:
    print("=" * 70)
    print(subject)
    print("=" * 70)
    print(body_text)


CHANNELS: dict[str, Callable[[str, str, str | None], None]] = {
    "email": _send_email,
    "stdout": _print_to_stdout,
}


def notify(sig: Signal, *, prior_weights: dict[str, float] | None = None,
           channel: str | None = None) -> None:
    channel = channel or os.environ.get("NOTIFY_CHANNEL", "email")
    send = CHANNELS.get(channel)
    if send is None:
        raise ValueError(f"Unknown NOTIFY_CHANNEL: {channel}")

    held = ", ".join(f"{t} {w*100:.0f}%" for t, w in sig.weights.items() if w > 0) or "all cash"
    subject = f"[RAAM] {sig.as_of.date()} rebalance: {held}"
    body = format_signal_text(sig, prior_weights=prior_weights)
    html = format_signal_html(sig, prior_weights=prior_weights) if channel == "email" else None
    send(subject, body, html)
