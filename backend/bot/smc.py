"""
Smart Money Concepts (SMC) detection engine.

Computes the structural facts an SMC trader looks for — deterministically, from
candles — so the AI brain reasons over real levels instead of hallucinating them:

  • Market structure   — swing points labelled HH / HL / LH / LL  → trend
  • BOS vs CHoCH       — break of structure (continuation) vs change of character (reversal)
  • Liquidity          — equal highs/lows, prev-day high/low, swing pools (buy/sell-side)
  • Liquidity sweep    — wick beyond a level that closes back (stop hunt)
  • Order blocks       — last opposing candle before an impulsive, structure-breaking move
  • Premium / Discount — fib equilibrium (50%) + OTE zone of the current dealing range

`analyze(candles)` bundles everything into one compact, model-friendly dict.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from bot.lux_indicators import _atr
from bot.swings import find_swings as _swings


def _label(swings: list[dict]) -> list[dict]:
    """Tag each swing high HH/LH and each swing low HL/LL vs the prior same-type swing."""
    last_h = last_l = None
    labeled = []
    for s in swings:
        s = dict(s)
        if s["kind"] == "high":
            s["label"] = "HH" if (last_h is not None and s["price"] > last_h) else ("LH" if last_h is not None else "H")
            last_h = s["price"]
        else:
            s["label"] = "HL" if (last_l is not None and s["price"] > last_l) else ("LL" if last_l is not None else "L")
            last_l = s["price"]
        labeled.append(s)
    return labeled


def _trend(labeled: list[dict]) -> str:
    highs = [s["label"] for s in labeled if s["kind"] == "high"][-2:]
    lows = [s["label"] for s in labeled if s["kind"] == "low"][-2:]
    bull = ("HH" in highs) and ("HL" in lows)
    bear = ("LH" in highs) and ("LL" in lows)
    if bull and not bear:
        return "bullish"
    if bear and not bull:
        return "bearish"
    if highs[-1:] == ["HH"] or lows[-1:] == ["HL"]:
        return "bullish"
    if highs[-1:] == ["LH"] or lows[-1:] == ["LL"]:
        return "bearish"
    return "ranging"


def _breaks(candles: list[dict], swings: list[dict]) -> list[dict]:
    """Scan closes against the standing swing high/low to emit BOS / CHoCH events."""
    events: list[dict] = []
    trend = 0  # +1 up, -1 down
    standing_high = standing_low = None
    si = 0
    for i, c in enumerate(candles):
        while si < len(swings) and swings[si]["i"] <= i:
            s = swings[si]
            if s["kind"] == "high":
                standing_high = s["price"]
            else:
                standing_low = s["price"]
            si += 1
        close = c["close"]
        if standing_high is not None and close > standing_high:
            events.append({"type": "BOS" if trend >= 0 else "CHoCH", "dir": "bullish",
                           "level": round(standing_high, 2), "time": int(c["time"]), "i": i})
            trend, standing_high = 1, None
        elif standing_low is not None and close < standing_low:
            events.append({"type": "BOS" if trend <= 0 else "CHoCH", "dir": "bearish",
                           "level": round(standing_low, 2), "time": int(c["time"]), "i": i})
            trend, standing_low = -1, None
    return events


def _clusters(vals: list[float], tol: float) -> list[float]:
    """Average of groups of near-equal levels (>= 2 members) — 'equal' highs/lows."""
    if not vals:
        return []
    vals = sorted(vals)
    out, group = [], [vals[0]]
    for v in vals[1:]:
        if v - group[-1] <= tol:
            group.append(v)
        else:
            if len(group) >= 2:
                out.append(round(sum(group) / len(group), 2))
            group = [v]
    if len(group) >= 2:
        out.append(round(sum(group) / len(group), 2))
    return out


def _prev_day_hl(candles: list[dict]) -> tuple[Optional[float], Optional[float]]:
    by_day: dict[str, list[dict]] = {}
    for c in candles:
        d = datetime.fromtimestamp(int(c["time"]), tz=timezone.utc).date().isoformat()
        by_day.setdefault(d, []).append(c)
    days = sorted(by_day.keys())
    if len(days) < 2:
        return None, None
    prev = by_day[days[-2]]
    return round(max(x["high"] for x in prev), 2), round(min(x["low"] for x in prev), 2)


def _sweep(candles: list[dict], up_levels: list[float], dn_levels: list[float], lookback: int = 6) -> Optional[dict]:
    """Most recent stop-hunt: wick beyond a level that closes back inside."""
    for c in reversed(candles[-lookback:]):
        for lv in up_levels:
            if c["high"] > lv and c["close"] < lv:
                return {"side": "buyside", "level": round(lv, 2), "dir": "bearish", "time": int(c["time"])}
        for lv in dn_levels:
            if c["low"] < lv and c["close"] > lv:
                return {"side": "sellside", "level": round(lv, 2), "dir": "bullish", "time": int(c["time"])}
    return None


def _order_blocks(candles: list[dict], breaks: list[dict], price: float, atr: float) -> dict:
    """Last opposing candle before each recent structure-breaking impulse."""
    obs = {"bullish": [], "bearish": []}
    near = max(atr * 4, price * 0.02)
    for ev in reversed(breaks[-8:]):
        i = ev["i"]
        if ev["dir"] == "bullish" and len(obs["bullish"]) < 2:
            for j in range(i - 1, max(i - 12, -1), -1):
                if candles[j]["close"] < candles[j]["open"]:
                    lo, hi = round(candles[j]["low"], 2), round(candles[j]["high"], 2)
                    if abs((lo + hi) / 2 - price) <= near:
                        obs["bullish"].append({"low": lo, "high": hi, "time": int(candles[j]["time"]),
                                               "mitigated": price < hi})
                    break
        elif ev["dir"] == "bearish" and len(obs["bearish"]) < 2:
            for j in range(i - 1, max(i - 12, -1), -1):
                if candles[j]["close"] > candles[j]["open"]:
                    lo, hi = round(candles[j]["low"], 2), round(candles[j]["high"], 2)
                    if abs((lo + hi) / 2 - price) <= near:
                        obs["bearish"].append({"low": lo, "high": hi, "time": int(candles[j]["time"]),
                                               "mitigated": price > lo})
                    break
    return obs


def _premium_discount(swings: list[dict], price: float) -> Optional[dict]:
    highs = [s["price"] for s in swings if s["kind"] == "high"]
    lows = [s["price"] for s in swings if s["kind"] == "low"]
    if not highs or not lows:
        return None
    hi, lo = highs[-1], lows[-1]
    hi, lo = max(hi, lo), min(hi, lo)
    if hi <= lo:
        return None
    eq = (hi + lo) / 2
    zone = "premium" if price > eq else "discount" if price < eq else "equilibrium"
    # OTE (optimal trade entry) = 0.618–0.79 retracement of the range
    ote_low = round(hi - (hi - lo) * 0.79, 2)
    ote_high = round(hi - (hi - lo) * 0.618, 2)
    return {"range_high": round(hi, 2), "range_low": round(lo, 2), "equilibrium": round(eq, 2),
            "zone": zone, "pct_of_range": round((price - lo) / (hi - lo) * 100, 1),
            "discount_ote": [ote_low, ote_high]}


def analyze(candles: list[dict], atr_period: int = 14) -> Optional[dict]:
    """Full SMC read of one timeframe. Returns a compact dict (or None if too few bars)."""
    if not candles or len(candles) < 25:
        return None
    price = candles[-1]["close"]
    atr_list = _atr(candles, atr_period)
    atr = atr_list[-1] if atr_list else max(price * 0.002, 1.0)

    swings = _swings(candles)
    labeled = _label(swings)
    trend = _trend(labeled)
    breaks = _breaks(candles, swings)

    tol = max(atr * 0.15, price * 0.0006)
    highs = [s["price"] for s in swings if s["kind"] == "high"]
    lows = [s["price"] for s in swings if s["kind"] == "low"]
    equal_highs = _clusters(highs, tol)
    equal_lows = _clusters(lows, tol)
    pdh, pdl = _prev_day_hl(candles)

    buyside = sorted([h for h in highs if h > price])[:3]           # liquidity resting above
    sellside = sorted([l for l in lows if l < price], reverse=True)[:3]  # liquidity resting below
    up_levels = list(dict.fromkeys(buyside + equal_highs + ([pdh] if pdh else [])))
    dn_levels = list(dict.fromkeys(sellside + equal_lows + ([pdl] if pdl else [])))

    return {
        "trend": trend,
        "swings": [{"label": s["label"], "price": s["price"], "kind": s["kind"], "time": s["time"]} for s in labeled[-8:]],
        "structure_break": (breaks[-1] and {k: breaks[-1][k] for k in ("type", "dir", "level", "time")}) if breaks else None,
        "liquidity": {
            "buyside": buyside, "sellside": sellside,
            "equal_highs": equal_highs[-3:], "equal_lows": equal_lows[-3:],
            "prev_day_high": pdh, "prev_day_low": pdl,
        },
        "recent_sweep": _sweep(candles, up_levels, dn_levels),
        "order_blocks": _order_blocks(candles, breaks, price, atr),
        "premium_discount": _premium_discount(swings, price),
    }


def bias(smc: Optional[dict]) -> int:
    """Coarse directional read for a strategy vote: +1 bullish / -1 bearish / 0 neutral."""
    if not smc:
        return 0
    t = smc.get("trend")
    return 1 if t == "bullish" else -1 if t == "bearish" else 0
