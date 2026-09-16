"""
Auto-tune: learn each strategy's influence from LIVE results.

When a trade closes, the strategies that voted for that trade's direction are
credited with the trade's R-multiple (profit/loss ÷ risk). Cumulative per-strategy
expectancy then sets a vote WEIGHT (strategies that make money vote louder; ones
that lose money vote quieter — but never fully silenced). Weights feed the
decision engine so the bot self-tunes its strategy mix over time.
"""
import time
from db import db
from config import settings
from bot import strategies

_weights_cache = {"ts": 0.0, "w": {}}


async def record_outcome(votes: dict, action: str, r_multiple: float):
    """Credit every strategy that voted `action` (BUY/SELL) with this trade's R."""
    if not votes:
        return
    for sid, vote in votes.items():
        if vote == action:
            await db.strategy_perf.update_one(
                {"_id": sid},
                {"$inc": {"count": 1, "sum_r": float(r_multiple), "wins": 1 if r_multiple > 0 else 0}},
                upsert=True,
            )
    _weights_cache["ts"] = 0.0  # invalidate so weights refresh next read


async def get_perf() -> dict:
    out = {}
    async for d in db.strategy_perf.find({}):
        c = d.get("count", 0)
        out[d["_id"]] = {
            "count": c,
            "sum_r": round(d.get("sum_r", 0.0), 2),
            "wins": d.get("wins", 0),
            "win_rate": round(d.get("wins", 0) / c * 100, 1) if c else 0,
            "expectancy": round(d.get("sum_r", 0.0) / c, 3) if c else 0,
        }
    return out


def _weight_from(expectancy: float) -> float:
    w = 1.0 + settings.autotune_gain * expectancy
    return max(settings.weight_min, min(settings.weight_max, round(w, 3)))


async def get_weights(force: bool = False) -> dict:
    """Per-strategy vote weight (1.0 until enough trades, then driven by expectancy)."""
    now = time.time()
    if not force and now - _weights_cache["ts"] < 30 and _weights_cache["w"]:
        return _weights_cache["w"]
    try:
        perf = await get_perf()
    except Exception:
        # Mongo unreachable → fall back to neutral weights so AI+Delta trading
        # keeps running (don't let a DB outage abort the whole tick).
        return _weights_cache["w"] or {sid.value: 1.0 for sid in strategies.StrategyId}
    w = {}
    for sid in strategies.StrategyId:
        p = perf.get(sid.value)
        if not settings.autotune_enabled or not p or p["count"] < settings.autotune_min_trades:
            w[sid.value] = 1.0
        else:
            w[sid.value] = _weight_from(p["expectancy"])
    _weights_cache.update(ts=now, w=w)
    return w


async def status() -> dict:
    perf = await get_perf()
    weights = await get_weights(force=True)
    return {
        "enabled": settings.autotune_enabled,
        "min_trades": settings.autotune_min_trades,
        "strategies": [
            {"id": sid.value, "label": strategies.LABELS.get(sid, sid.value),
             "weight": weights.get(sid.value, 1.0), **(perf.get(sid.value) or {"count": 0, "sum_r": 0, "wins": 0, "win_rate": 0, "expectancy": 0})}
            for sid in strategies.StrategyId
        ],
    }


async def record_shadow_outcome(shadow_votes: dict, action: str, r_multiple: float):
    """Same idea as record_outcome(), but on a side ledger (db.shadow_strategy_perf)
    that never feeds get_weights() — shadow strategies (config.shadow_strategies)
    build a real track record against realized trade R-multiples without ever
    being able to influence a live decision."""
    if not shadow_votes:
        return
    for sid, vote in shadow_votes.items():
        if vote == action:
            await db.shadow_strategy_perf.update_one(
                {"_id": sid},
                {"$inc": {"count": 1, "sum_r": float(r_multiple), "wins": 1 if r_multiple > 0 else 0}},
                upsert=True,
            )


async def shadow_status() -> dict:
    """Performance of shadow-only strategies — the evidence used to decide whether
    to promote one into the live, voting `strategies` CSV."""
    out = []
    async for d in db.shadow_strategy_perf.find({}):
        c = d.get("count", 0)
        sid = d["_id"]
        try:
            enum_id = strategies.StrategyId(sid)
        except ValueError:
            enum_id = None
        out.append({
            "id": sid,
            "label": strategies.LABELS.get(enum_id, sid) if enum_id else sid,
            "count": c,
            "sum_r": round(d.get("sum_r", 0.0), 2),
            "wins": d.get("wins", 0),
            "win_rate": round(d.get("wins", 0) / c * 100, 1) if c else 0,
            "expectancy": round(d.get("sum_r", 0.0) / c, 3) if c else 0,
        })
    return {"shadow_strategies": settings.shadow_strategies, "strategies": out}
