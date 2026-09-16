"""
Portfolio-level risk: realized correlation between two traded symbols.

With trade_symbols defaulting to BTCUSD,ETHUSD and max_concurrent_positions=2, both
can be open same-direction at once — which silently doubles correlated exposure
rather than diversifying it. This computes realized correlation on demand so the
sizing step in scheduler.py can dampen (not block) the second same-direction entry.
"""
from __future__ import annotations
import time
from typing import Optional

import pandas as pd

from bot.delta_client import DeltaClient

_cache: dict[tuple, tuple[float, Optional[float]]] = {}  # (sym_a, sym_b, tf) -> (ts, corr)
_TTL = 300.0  # correlation drifts slowly; no need to recompute more than every 5 min


async def realized_correlation(delta: DeltaClient, sym_a: str, sym_b: str,
                               timeframe_min: int, lookback: int = 200) -> Optional[float]:
    """Pearson correlation of the two symbols' recent bar-over-bar returns, or None
    if either candle series is unavailable/too short. Cached per symbol pair+timeframe."""
    a, b = sorted((sym_a.upper(), sym_b.upper()))
    key = (a, b, timeframe_min)
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached[0] < _TTL:
        return cached[1]
    try:
        ca = await delta.get_candles(a, timeframe_min, lookback)
        cb = await delta.get_candles(b, timeframe_min, lookback)
    except Exception:
        return cached[1] if cached else None
    if len(ca) < 20 or len(cb) < 20:
        return cached[1] if cached else None

    da = {int(c["time"]): c["close"] for c in ca}
    db_ = {int(c["time"]): c["close"] for c in cb}
    common = sorted(set(da) & set(db_))
    if len(common) < 20:
        return cached[1] if cached else None

    sa = pd.Series([da[t] for t in common]).pct_change().dropna()
    sb = pd.Series([db_[t] for t in common]).pct_change().dropna()
    n = min(len(sa), len(sb))
    if n < 15:
        return cached[1] if cached else None
    corr = sa.tail(n).reset_index(drop=True).corr(sb.tail(n).reset_index(drop=True))
    result = float(corr) if pd.notna(corr) else None
    _cache[key] = (now, result)
    return result
