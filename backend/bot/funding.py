"""
Perpetual funding-rate + open-interest tracking.

"Extreme" funding is symbol-relative (BTC and ETH run different typical funding
ranges), so this keeps a rolling history in Mongo (db.funding_history — the same
durability pattern bot_state/trade_outcomes already use) and classifies the CURRENT
reading against its own percentile rank rather than a fixed absolute threshold.
Needs a minimum sample count before calling anything "extreme".
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional

from bot.delta_client import DeltaClient
from config import settings
from db import db


async def record_and_bias(symbol: str, delta: DeltaClient) -> Optional[dict]:
    """Fetch current funding/OI, persist a history point, and return the
    symbol-relative bias read: {funding_rate, oi, oi_change_6h, percentile, extreme}.
    `extreme` is "high"/"low"/None; None until enough history has accumulated."""
    try:
        cur = await delta.get_funding_and_oi(symbol)
    except Exception:
        return None
    if cur.get("funding_rate") is None:
        return None

    now = datetime.now(timezone.utc)
    try:
        await db.funding_history.insert_one({
            "symbol": symbol, "ts": now,
            "funding_rate": cur["funding_rate"], "oi": cur.get("oi"),
        })
    except Exception:
        pass  # a missed history point degrades the percentile, not correctness

    since = now - timedelta(days=settings.funding_history_window_days)
    try:
        rates = [d["funding_rate"] async for d in db.funding_history.find(
            {"symbol": symbol, "ts": {"$gte": since}}, {"funding_rate": 1, "_id": 0})
            if isinstance(d.get("funding_rate"), (int, float))]
    except Exception:
        rates = []

    extreme = percentile = None
    if len(rates) >= settings.funding_min_history_samples:
        rank = sum(1 for r in rates if r <= cur["funding_rate"]) / len(rates)
        percentile = round(rank, 3)
        if rank >= settings.funding_extreme_percentile:
            extreme = "high"
        elif rank <= (1 - settings.funding_extreme_percentile):
            extreme = "low"

    return {
        "funding_rate": cur["funding_rate"],
        "oi": cur.get("oi"),
        "oi_change_6h": cur.get("oi_change_6h"),
        "percentile": percentile,
        "extreme": extreme,
        "samples": len(rates),
    }
