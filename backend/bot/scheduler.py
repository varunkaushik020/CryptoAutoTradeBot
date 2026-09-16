"""
APScheduler job that fires every CHECK_INTERVAL_MINUTES.
On each tick:
  1. Fetch latest candles from Delta Exchange
  2. Compute indicators
  3. Run strategy → BUY / SELL / HOLD
  4. If BUY/SELL: place order via Delta Exchange API
  5. Log everything to MongoDB
"""
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from bot.delta_client import DeltaClient, vwap_fill
from bot.indicators import compute_indicators
from bot.lux_indicators import supertrend_ai, trendline_breakout_navigator, fair_value_gaps, inverse_fvg, _atr
from bot import strategies, autotune, ai_brain, smc, edge as edge_mod, news, agents as agents_mod, ensemble
from bot import divergence as divergence_mod, funding as funding_mod, orderbook as orderbook_mod
from bot import portfolio_risk
from config import settings
from db import db

logger = logging.getLogger("bot.scheduler")

scheduler = AsyncIOScheduler()
_bot_running = False
_delta = DeltaClient()
_last_ai_call: dict[str, float] = {}  # symbol -> monotonic ts of last LLM fallback (cooldown)


def _analyze(candles, vwap_candles=None):
    """Compute every indicator for one timeframe."""
    ind = compute_indicators(
        candles,
        ema_fast=settings.ema_fast,
        ema_slow=settings.ema_slow,
        rsi_period=settings.rsi_period,
        rsi_oversold=settings.rsi_oversold,
        rsi_overbought=settings.rsi_overbought,
        bb_period=settings.bb_period,
        bb_mult=settings.bb_mult,
        kc_period=settings.kc_period,
        kc_atr_mult=settings.kc_atr_mult,
        kc_atr_len=settings.kc_atr_len,
        adx_period=settings.adx_period,
        vwap_enabled=settings.vwap_enabled,
        vwap_candles=vwap_candles,
    )
    st = supertrend_ai(candles)
    tn = trendline_breakout_navigator(candles, term=settings.trendline_term)
    fvg = fair_value_gaps(candles)
    ifvg = inverse_fvg(candles)
    return ind, st, tn, fvg, ifvg


def _trend_bias(ind, st, tn) -> int:
    """Higher-TF continuous trend: +1 bullish / -1 bearish / 0 neutral."""
    score = 0
    if ind.get("ema_fast") is not None and ind.get("ema_slow") is not None:
        score += 1 if ind["ema_fast"] > ind["ema_slow"] else -1
    if st:
        d = st["latest"].get("dir")
        score += 1 if d == "long" else -1 if d == "short" else 0
    if tn:
        score += int(tn["latest"].get("trend") or 0)
    return 1 if score > 0 else -1 if score < 0 else 0


def _combined_sl(long: bool, entry: float, candles: list, st) -> tuple[float, dict, str]:
    """
    Stop-loss = the MOST PROTECTIVE of three methods, clamped to a sane range:
      • ATR:        entry -/+ k*ATR (volatility)
      • SuperTrend: the SuperTrend AI trailing-stop line (trend flip level)
      • Structure:  most recent swing low/high (market structure)
    Returns (sl_price, candidate_prices, chosen_method).
    """
    cands: dict[str, float] = {}
    atr_list = _atr(candles, settings.atr_period)
    atr = atr_list[-1] if atr_list else 0.0
    if atr > 0:
        cands["atr"] = entry - settings.atr_k * atr if long else entry + settings.atr_k * atr
    if st and st.get("latest", {}).get("ts"):
        ts = float(st["latest"]["ts"])
        if (long and ts < entry) or (not long and ts > entry):
            cands["supertrend"] = ts
    lb = candles[-settings.sl_lookback:]
    if lb:
        cands["structure"] = min(c["low"] for c in lb) if long else max(c["high"] for c in lb)

    if not cands:
        sld = settings.stop_loss_pct / 100
        sl = entry * (1 - sld) if long else entry * (1 + sld)
        return sl, {"fixed": round(sl, 2)}, "fixed"

    # most protective = furthest from entry
    sl = min(cands.values()) if long else max(cands.values())
    method = min(cands, key=cands.get) if long else max(cands, key=cands.get)

    # clamp risk to [min, max] % of price so sizing & targets stay sane
    min_d = entry * settings.min_sl_pct / 100
    max_d = entry * settings.max_sl_pct / 100
    risk = abs(entry - sl)
    if risk < min_d:
        sl = entry - min_d if long else entry + min_d
    elif risk > max_d:
        sl = entry - max_d if long else entry + max_d
        method += "+capped"
    return sl, {k: round(v, 2) for k, v in cands.items()}, method


def _room_for_1to2(long: bool, entry: float, risk: float, trend_candles: list) -> tuple[bool, str]:
    """Only trade if a 1:2 is reachable before the next 1h structural level."""
    if risk <= 0:
        return False, "no risk"
    lb = trend_candles[-settings.target_lookback:] if trend_candles else []
    if not lb:
        return True, "open (no structure)"
    need = 2 * risk
    if long:
        res = max(c["high"] for c in lb)
        if res <= entry:
            return True, "open upside"
        room = res - entry
        return room >= need, f"{room:.0f} to 1h resistance vs need {need:.0f}"
    sup = min(c["low"] for c in lb)
    if sup >= entry:
        return True, "open downside"
    room = entry - sup
    return room >= need, f"{room:.0f} to 1h support vs need {need:.0f}"


def _rr_target(agree: int, strength: int, aligned: bool) -> float:
    """Dynamic reward:risk. More agreement + stronger aligned trend = bigger target."""
    base = {2: 2.0, 3: 3.0, 4: 5.0}.get(agree, 10.0 if agree >= 5 else 2.0)
    if aligned:
        if strength >= 8:
            base = max(base, 5.0)
        if strength >= 9 and agree >= 4:
            base = 10.0
    return base


def _swings(candles, highs: bool) -> list[float]:
    """Local swing highs (highs=True) or lows over a 2-bar window each side."""
    out = []
    n = len(candles)
    for i in range(2, n - 2):
        c = candles[i]
        if highs:
            h = c["high"]
            if h > candles[i - 1]["high"] and h > candles[i - 2]["high"] and h >= candles[i + 1]["high"] and h >= candles[i + 2]["high"]:
                out.append(h)
        else:
            l = c["low"]
            if l < candles[i - 1]["low"] and l < candles[i - 2]["low"] and l <= candles[i + 1]["low"] and l <= candles[i + 2]["low"]:
                out.append(l)
    return out


def _target_levels(long: bool, entry: float, risk: float, c_entry, c_trend, fvg, rr_top: float) -> list[float]:
    """
    TP1/TP2/TP3 snapped to logical structure (swing highs/lows + FVG edges),
    each at least 1:2 / 1:3 / (5 or rr_top) reward:risk. Falls back to the
    ratio price when no structure level is available.
    """
    levels = _swings(c_entry[-120:], long) + _swings(c_trend[-60:], long)
    if fvg:
        for z in fvg.get("unmitigated", []):
            levels.append(z["bottom"] if long else z["top"])  # opposite gap edge = magnet/target
    levels = [l for l in levels if (l > entry if long else l < entry)]
    levels = sorted(set(round(l, 1) for l in levels), reverse=not long)  # nearest-first

    floors = [2.0, 3.0, max(5.0, rr_top)][:max(1, settings.max_tps)]
    tps = []
    last = entry
    for fl in floors:
        floor_price = entry + fl * risk if long else entry - fl * risk
        # target must clear both the RR floor AND the previous TP (monotonic)
        min_price = max(floor_price, last + risk) if long else min(floor_price, last - risk)
        chosen = None
        for l in levels:  # nearest-first
            if (long and l >= min_price) or (not long and l <= min_price):
                chosen = l
                break
        if chosen is None:
            chosen = min_price  # ratio fallback (already beyond last)
        tps.append(round(chosen, 1))
        last = chosen
    return tps


def _clamp_sl(long: bool, entry: float, sl: float) -> float | None:
    """Force an AI-proposed stop onto the correct side of entry and into the
    allowed distance band. Returns None if it's on the wrong side (unusable)."""
    if sl is None:
        return None
    min_d = entry * settings.min_sl_pct / 100
    max_d = entry * settings.max_sl_pct / 100
    if long:
        if sl >= entry:
            return None
        d = min(max(entry - sl, min_d), max_d)
        return round(entry - d, 2)
    if sl <= entry:
        return None
    d = min(max(sl - entry, min_d), max_d)
    return round(entry + d, 2)


def _valid_ai_tps(long: bool, entry: float, risk: float, ai_tps: list) -> list[tuple]:
    """Keep only AI take-profits that are on the right side, strictly progressive,
    and whose first level clears the 2R floor. Returns [(price, size_pct|None), ...]."""
    if not ai_tps or risk <= 0:
        return []
    need1 = settings.risk_reward * risk
    out, last = [], entry
    for t in ai_tps:
        p = t.get("price")
        if p is None:
            continue
        beyond = (p > last) if long else (p < last)
        if not beyond:
            continue
        if not out:  # first accepted TP must clear the R:R floor
            r1 = (p - entry) if long else (entry - p)
            if r1 < need1:
                continue
        out.append((round(p, 1), t.get("size_pct")))
        last = p
        if len(out) >= settings.max_tps:  # never more than max_tps take-profits
            break
    return out


def _tp_sizes(lots: int, accepted: list[tuple]) -> list[int]:
    """Split `lots` across TPs using AI size_pct when all present & sane, else config splits."""
    pcts = [s for _, s in accepted]
    if accepted and all(isinstance(s, (int, float)) and s > 0 for s in pcts) and 0.5 <= sum(pcts) <= 1.5:
        fracs = [s / sum(pcts) for s in pcts]
    else:
        cfg = [float(x) for x in settings.tp_splits.split(",")]
        fracs = (cfg + [0] * len(accepted))[:len(accepted)] or [1.0]
    sizes = [int(lots * f) for f in fracs]
    rem = lots - sum(sizes)
    for j in range(len(sizes)):
        if rem <= 0:
            break
        sizes[j] += 1
        rem -= 1
    return sizes


async def _daily_loss_exceeded(balance: float) -> tuple[bool, float, float]:
    """Circuit breaker: True once today's REALIZED loss across all traded symbols
    reaches daily_loss_limit_pct of capital. Returns (exceeded, today_realized, limit).
    Fails OPEN (allows trading) if it can't be computed — every trade still carries its
    own exchange-side stop."""
    limit_pct = settings.daily_loss_limit_pct
    if limit_pct <= 0 or balance <= 0:
        return False, 0.0, 0.0
    from routers.trades import _build_view  # local import avoids a circular import
    syms = {s.strip().upper() for s in settings.trade_symbols.split(",") if s.strip()}
    total = 0.0
    for sym in syms:
        try:
            v = await _build_view(sym)
            total += float(v.get("today_realized") or 0.0)
        except Exception:
            pass
    limit = -abs(limit_pct) / 100 * balance
    return (total <= limit), round(total, 2), round(limit, 2)


async def _liquidity_check(symbol: str, want_long: bool, lots: int = 0) -> tuple[bool, str]:
    """Is this market tradeable right now?

    Two questions, both asked BEFORE committing capital:
      1. Is the quoted spread sane? A wide book means the entry itself is a loss.
      2. Could we actually EXIT this size? Never enter a market you can't leave —
         a position you can only close at a real concession is a trap, not a trade.
    """
    try:
        book = await _delta.get_orderbook(symbol)
        t = await _delta.get_ticker(symbol)
        mark = float(t.get("mark_price") or t.get("close") or 0)
    except Exception as e:
        return False, f"order book unavailable ({type(e).__name__})"
    bids, asks = book.get("buy") or [], book.get("sell") or []
    if not bids or not asks or not mark:
        return False, "empty order book"

    bid, ask = float(bids[0]["price"]), float(asks[0]["price"])
    spread_pct = (ask - bid) / mark * 100
    if spread_pct > settings.max_entry_spread_pct:
        return False, f"spread {spread_pct:.2f}% > {settings.max_entry_spread_pct:g}% (illiquid)"

    if lots > 0:
        # exiting a long sells into bids; exiting a short buys from asks
        exit_levels = bids if want_long else asks
        px, got = vwap_fill(exit_levels, lots)
        if got < lots - 1e-9 or not px:
            return False, f"book too thin to exit {lots} lots"
        exit_slip = abs(px - mark) / mark * 100
        if exit_slip > settings.max_exit_slippage_pct:
            return False, (f"exit would cost {exit_slip:.2f}% "
                           f"(> {settings.max_exit_slippage_pct:g}%) — unexitable")
    return True, f"spread {spread_pct:.2f}%"


async def _liquidity_gate(symbol: str, want_long: bool, lots: int = 0) -> tuple[bool, str, dict]:
    """Wraps `_liquidity_check` with the shadow/enforce mode switch.

    In "shadow" mode the check still runs and is still logged, but never blocks a
    trade — this exists to gather real block-rate data (the testnet book, ETH's
    especially, is sometimes empty) before ever flipping to "enforce".
    Returns (allow_trade, status_text, log_meta).
    """
    if not settings.liquidity_gate_enabled or settings.liquidity_gate_mode == "off":
        return True, "", {"mode": "off"}
    ok, txt = await _liquidity_check(symbol, want_long, lots)
    meta = {"mode": settings.liquidity_gate_mode, "ok": ok, "reason": txt, "lots": lots}
    if not ok and settings.liquidity_gate_mode == "enforce":
        return False, txt, meta
    return True, txt, meta


def _symbol_profile(symbol: str, big: bool) -> dict | None:
    """POINT-based SL/TP limits for symbols that use them (ETH). Returns None for
    symbols that keep the percent-based logic (e.g. BTC)."""
    s = symbol.upper()
    if s.startswith("ETH"):
        if big:
            return {"sl_min": settings.eth_sl_min_pts, "sl_max": settings.eth_big_sl_max_pts,
                    "tp_min": settings.eth_big_tp_min_pts, "tp_max": settings.eth_big_tp_max_pts}
        return {"sl_min": settings.eth_sl_min_pts, "sl_max": settings.eth_sl_max_pts,
                "tp_min": settings.eth_tp_min_pts, "tp_max": settings.eth_tp_max_pts}
    return None


def _safe_leverage(entry: float, sl_dist: float) -> int:
    """Largest leverage (<= configured max) whose liquidation price stays BEYOND the
    stop-loss by liq_buffer_pct — so the stop always triggers before liquidation."""
    if entry <= 0 or sl_dist <= 0:
        return settings.leverage
    sl_frac = sl_dist / entry + settings.liq_buffer_pct / 100.0
    lev = int(1.0 / sl_frac) if sl_frac > 0 else settings.leverage
    return max(1, min(settings.leverage, lev))


class _SkipEntry(Exception):
    """Raised to cleanly skip opening a new entry (not an error)."""


def _parse_iso(v):
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


async def _closed_trade_realized(symbol: str, opened_at) -> tuple[float, datetime | None]:
    """Realized USD P/L of the just-closed trade, plus the time of its last fill.

    The exit timestamp is what lets the trade be scored against the MARK price at the
    moment it actually closed. With no fills we know neither number, so the caller
    stores the row as unverified rather than guessing — a guess here is what once
    produced 129 phantom "wins" and made the training panel lie.
    """
    cv = await _delta.get_contract_value(symbol)
    try:
        fills = await _delta.get_fills(page_size=500)
    except Exception:
        return 0.0, None
    rows = []
    start = _parse_iso(opened_at)
    for f in fills:
        if f.get("product_symbol") != symbol:
            continue
        ts = _parse_iso(f.get("created_at"))
        if ts < start - timedelta(seconds=2):
            continue
        side = 1 if str(f.get("side", "")).lower() == "buy" else -1
        qty = float(f.get("size") or 0)
        price = float(f.get("price") or 0)
        if qty > 0 and price > 0:
            rows.append((ts, side, qty, price, float(f.get("commission") or 0)))
    rows.sort(key=lambda x: x[0])
    pos = avg = realized = 0.0
    for _, side, qty, price, comm in rows:
        signed = side * qty
        if pos == 0:
            pos, avg = signed, price
        elif (pos > 0) == (side > 0):
            avg = (avg * abs(pos) + price * abs(signed)) / (abs(pos) + abs(signed))
            pos += signed
        else:
            cq = min(abs(pos), qty)
            realized += (price - avg) * (1 if pos > 0 else -1) * cq * cv
            new = pos + signed
            if new != 0 and (new > 0) != (pos > 0):
                avg = price
            pos = new
        realized -= comm
    return realized, (rows[-1][0] if rows else None)


async def _mark_at(symbol: str, when: datetime | None) -> tuple[float, bool]:
    """MARK price at `when`, plus whether that price really came from `when`.

    Attribution must be judged on the mark price the bot actually traded around, not
    on the fill — a thin book can put the fill percent(s) away from any real price.

    The flag matters. The candle feed is always anchored to *now*, so a `when` older
    than the window we fetch simply cannot be answered. Quietly substituting the live
    mark there would score a trade against a price it never saw — the same shape of
    error as the phantom "wins" this panel was built to stop. So the miss is reported
    instead, and the caller records such a row as unverified.
    """
    if when is not None:
        try:
            # The window ends at now, so ask for enough 1m bars to actually reach back
            # to `when` (a fixed 300 only ever covered ~5h).
            age_min = (datetime.now(timezone.utc) - when).total_seconds() / 60
            need = int(min(max(age_min + 10, 60), 2000))
            candles = await _delta.get_candles(symbol, 1, need, mark=True)
            ts = int(when.timestamp())
            prior = [c for c in candles if c["time"] <= ts]
            if prior:
                return float(prior[-1]["close"]), True
        except Exception:
            pass
    try:
        t = await _delta.get_ticker(symbol)
        # With no timestamp asked for, the live mark IS the answer; otherwise this is
        # a fallback that does not describe `when`.
        return float(t.get("mark_price") or t.get("close") or 0), when is None
    except Exception:
        return 0.0, False


async def _record_trade_outcome(symbol: str, state: dict) -> dict:
    """Score a just-closed trade two ways and persist both to `trade_outcomes`.

    strategy R  — mark entry -> mark exit. Measures the DECISION: was the direction
                  right? This is what trains the strategy weights.
    execution R — real fills incl. commission. Measures the VENUE: what the book
                  actually paid. Kept for visibility, never fed to the tuner, because
                  an empty order book would otherwise punish correct calls.

    This is the only writer of `trade_outcomes`; the /bot/training panel is a pure
    reader of it. If this stops being called, that panel silently freezes.
    """
    cv = await _delta.get_contract_value(symbol)
    realized_exec, exit_ts = await _closed_trade_realized(symbol, state.get("opened_at"))
    entry_mark = float(state.get("entry") or 0)
    size = abs(float(state.get("size") or 0))
    sign = 1 if state.get("side") == "buy" else -1
    exit_mark, exit_mark_exact = await _mark_at(symbol, exit_ts)

    strategy_pnl = (exit_mark - entry_mark) * sign * size * cv if (entry_mark and exit_mark) else 0.0
    risk_d = float(state.get("risk_dollars") or 0)

    def _r(pnl: float) -> float:
        if risk_d > 0:
            return round(pnl / risk_d, 3)
        return 1.0 if pnl > 0 else -1.0 if pnl < 0 else 0.0

    doc = {
        "symbol": symbol,
        # Without fills — or without a mark price from the actual exit moment — we
        # cannot know what happened; such rows are stored for visibility but excluded
        # from training and from the summary figures.
        "verified": exit_ts is not None and exit_mark_exact,
        # Kept apart so a failure says which half went missing.
        "fills_matched": exit_ts is not None,
        "exit_mark_exact": exit_mark_exact,
        "side": "BUY" if sign > 0 else "SELL",
        "size": size,
        "opened_at": state.get("opened_at"),
        "closed_at": exit_ts or datetime.now(timezone.utc),
        "entry_mark": round(entry_mark, 2),
        "exit_mark": round(exit_mark, 2),
        "fill_entry": state.get("fill_price"),
        "risk_dollars": round(risk_d, 2),
        "strategy_pnl": round(strategy_pnl, 2),
        "execution_pnl": round(realized_exec, 2),
        "slippage_cost": round(realized_exec - strategy_pnl, 2),
        "strategy_r": _r(strategy_pnl),
        "execution_r": _r(realized_exec),
        "votes": state.get("votes") or {},
        "sl_method": state.get("sl_method"),
        "tp_source": state.get("tp_source"),
        "ai_confidence": ((state.get("ai") or {}) or {}).get("confidence"),
    }
    try:
        await db.trade_outcomes.insert_one(dict(doc))
    except Exception as e:
        logger.error(f"outcome persist failed: {e}")
    return doc


async def _manage_open_position(symbol: str):
    """Move SL to breakeven after the first partial TP fills; clean up when flat."""
    try:
        state = await db.bot_state.find_one({"_id": symbol})
        pos = await _delta.get_position_size(symbol)
        pid = await _delta.get_product_id(symbol)

        if abs(pos) < 1e-9:
            # flat -> attribute the trade's result to its strategies, then clean up
            if state:
                try:
                    o = await _record_trade_outcome(symbol, state)
                    action = o["side"]
                    if o["verified"]:
                        # Train on the DECISION (mark->mark), not on what a thin book paid.
                        train_r = o["strategy_r"] if settings.autotune_use_mark_pnl else o["execution_r"]
                        await autotune.record_outcome(state.get("votes") or {}, action, train_r)
                        await autotune.record_shadow_outcome(state.get("shadow_votes") or {}, action, train_r)
                        # continuous learning: credit/debit the agents that drove this trade
                        agent_ids = state.get("agents") or []
                        if agent_ids:
                            await ensemble.record_outcome(agent_ids, train_r)
                        logger.info(
                            f"{symbol} closed — strategy {o['strategy_pnl']:+.2f} ({o['strategy_r']:+.2f}R) "
                            f"| execution {o['execution_pnl']:+.2f} ({o['execution_r']:+.2f}R) "
                            f"| slippage {o['slippage_cost']:+.2f} → attributed to {action} voters"
                            + (f" + agents {agent_ids}" if agent_ids else "")
                        )
                    else:
                        # No fills matched this trade: we cannot say what it did, so it
                        # must not teach the tuner anything.
                        why = ("no fills matched" if not o["fills_matched"]
                               else "no mark price for the exit moment")
                        logger.warning(
                            f"{symbol} closed but {why} — outcome recorded "
                            f"unverified and excluded from training."
                        )
                except Exception as e:
                    logger.error(f"attribution error: {e}")
                for o in await _delta.get_live_orders(symbol):
                    if o.get("reduce_only"):
                        await _delta.cancel_order(o["id"], pid)
                await db.bot_state.delete_one({"_id": symbol})
                logger.info(f"{symbol} flat — cleared trade state and leftover orders.")
            return

        if not state or state.get("be_moved"):
            return

        # a partial TP filled if current size < original size
        if abs(pos) < state["size"] - 1e-9:
            # true breakeven = the actual FILL price (zero PnL), not the mark entry
            entry = state.get("fill_price") or state["entry"]
            close_side = "buy" if pos < 0 else "sell"
            for o in await _delta.get_live_orders(symbol):
                if o.get("stop_order_type") == "stop_loss_order" and o.get("reduce_only"):
                    await _delta.cancel_order(o["id"], pid)
            await _delta.place_stop_order(symbol, close_side, abs(int(round(pos))), entry, "stop_loss_order")
            await db.bot_state.update_one({"_id": symbol}, {"$set": {"be_moved": True}})
            logger.info(f"{symbol} TP1 hit — moved stop-loss to breakeven ({entry:.1f}).")
    except Exception as e:
        logger.error(f"manage_position error: {e}")


#: Serialises every order-placing path. bot_tick holds it for its whole run, so
#: fast_tick can never interleave an entry with the deep tick's own execution.
_tick_lock = asyncio.Lock()

#: symbol -> {side, trigger, expires, armed_at}. Written by the deep tick when a
#: setup looks reachable before the next one; read by fast_tick.
_watch: dict[str, dict] = {}


def _expected_move(c_ltf: list[dict], horizon_sec: int) -> float | None:
    """How far price is expected to travel in `horizon_sec`, from 5m ATR.

    ATR is per 5m bar, so scale it to the horizon. Used to answer the question
    "could this trade become executable before the next deep tick?" — if the
    trigger is further away than the market is likely to move, don't arm.
    """
    try:
        series = _atr(c_ltf, settings.atr_period)   # per-bar series, newest last
    except Exception:
        return None
    if not series:
        return None
    atr = series[-1]
    if not atr or atr <= 0:
        return None
    bar_sec = max(settings.ltf_timeframe, 1) * 60
    return float(atr) * (horizon_sec / bar_sec)


def _update_watch(symbol: str, ai_plan: dict | None, price: float,
                  c_ltf: list[dict], allow_entry: bool, in_position: bool) -> None:
    """Arm/disarm the fast loop for one symbol, right after a deep analysis.

    Armed only when the AI wants a direction, names an entry level we have not
    reached, we are flat and allowed to enter, and that level is within reach at
    current volatility. Anything else clears the watch so it cannot fire stale.
    """
    _watch.pop(symbol, None)
    if not settings.fast_check_enabled or not allow_entry or in_position or not ai_plan:
        return
    side = ai_plan.get("action")
    trigger = ai_plan.get("entry")
    if side not in ("BUY", "SELL") or not trigger:
        return
    # Already through the level — the deep tick either took it or declined it;
    # arming here would re-enter on analysis that has already been acted on.
    if (side == "BUY" and price >= trigger) or (side == "SELL" and price <= trigger):
        return
    horizon = settings.check_interval_seconds or settings.check_interval_minutes * 60
    reach = _expected_move(c_ltf, horizon)
    if reach is None:
        return
    distance = abs(price - trigger)
    if distance > settings.fast_arm_atr_mult * reach:
        return
    now = datetime.now(timezone.utc)
    _watch[symbol] = {
        "side": side,
        "trigger": float(trigger),
        "confidence": ai_plan.get("confidence"),
        "armed_at": now,
        "expires": now + timedelta(seconds=settings.fast_arm_ttl_sec),
    }
    logger.info(f"[{symbol}] fast-watch ARMED {side} @ {trigger} "
                f"(price {price}, {distance:.2f} away, ~{reach:.2f} expected in {horizon}s)")


def _trigger_hit(watch: dict, price: float) -> bool:
    return ((watch["side"] == "BUY" and price >= watch["trigger"])
            or (watch["side"] == "SELL" and price <= watch["trigger"]))


async def _process_symbol(symbol: str, allow_entry: bool, weights: dict | None = None,
                          fast: bool = False):
    """Manage + (optionally) trade ONE symbol. `allow_entry` gates new positions
    so non-active symbols are still managed/closed but don't get fresh entries.

    `fast=True` is the execution loop re-running this path after an armed trigger
    was hit: identical guardrails and order code, but the AI step uses the fast
    provider chain so the entry is not delayed by a reasoning model."""
    try:
        # 0. Manage any open position (breakeven after TP1, cleanup when flat)
        await _manage_open_position(symbol)

        # 1. Fetch all THREE timeframes (1h bias, 15m decision, 5m timing)
        c_entry = await _delta.get_candles(symbol, settings.entry_timeframe, settings.candle_limit)
        c_trend = await _delta.get_candles(symbol, settings.trend_timeframe, settings.candle_limit)
        c_ltf = await _delta.get_candles(symbol, settings.ltf_timeframe, settings.candle_limit)
        if len(c_entry) < settings.ema_slow + 5:
            logger.warning("Not enough entry-timeframe candles yet, skipping tick.")
            return

        # VWAP needs REAL trade volume — mark-price candles (the default candle
        # source everywhere else) carry none. Fetch a parallel traded-price set just
        # for that, and only on the deep pass (the latency-sensitive fast pass skips
        # this extra fetch, same as it already skips historical_edge/news).
        c_entry_vol = None
        if settings.vwap_enabled and not fast:
            try:
                c_entry_vol = await _delta.get_candles(symbol, settings.entry_timeframe, settings.candle_limit, mark=False)
            except Exception as e:
                logger.warning(f"[{symbol}] traded-volume candles for VWAP failed: {e}")

        # 2. Analyze each timeframe (off the event loop — CPU-heavy)
        (ind, st, tn, fvg, ifvg), (ind_t, st_t, tn_t, _, _), (ind_l, st_l, tn_l, _, _), smc_e, smc_t, smc_l, divergence = await asyncio.gather(
            asyncio.to_thread(_analyze, c_entry, c_entry_vol),  # entry / decision (15m)
            asyncio.to_thread(_analyze, c_trend),         # trend / bias (1h)
            asyncio.to_thread(_analyze, c_ltf),           # timing (5m)
            asyncio.to_thread(smc.analyze, c_entry),      # SMC read (15m)
            asyncio.to_thread(smc.analyze, c_trend),      # SMC read (1h)
            asyncio.to_thread(smc.analyze, c_ltf),        # SMC read (5m)
            asyncio.to_thread(divergence_mod.detect_divergence, c_entry,
                              settings.divergence_swing_left, settings.divergence_swing_right),
        )
        bias = _trend_bias(ind_t, st_t, tn_t)
        bias_txt = {1: "bullish", -1: "bearish", 0: "neutral"}[bias]

        # Crypto-native shadow signals: funding-rate bias (skipped on fast passes —
        # a slow-moving signal that doesn't need 15s-cadence history writes) and
        # order-book imbalance (cheap; get_orderbook has its own short-TTL cache).
        funding_read = None
        if not fast:
            try:
                funding_read = await funding_mod.record_and_bias(symbol, _delta)
            except Exception as e:
                logger.warning(f"[{symbol}] funding read failed: {e}")
        orderbook_read = None
        try:
            orderbook_read = orderbook_mod.imbalance(await _delta.get_orderbook(symbol))
        except Exception as e:
            logger.warning(f"[{symbol}] orderbook read failed: {e}")

        # 3. Entry votes on the lower timeframe (need >= min_signals)
        enabled = strategies.parse_enabled(settings.strategies)
        shadow = strategies.parse_enabled(settings.shadow_strategies)
        ctx = strategies.StrategyContext(candles=c_entry, ind=ind, supertrend=st, trendline=tn, fvg=fvg, ifvg=ifvg,
                                         extra={"smc": smc_e, "smc_trend": smc_t, "divergence": divergence,
                                                "funding": funding_read, "orderbook_imbalance": orderbook_read})
        result = strategies.evaluate(ctx, enabled, settings.min_signals, weights, shadow=shadow)
        entry_action = result["action"]

        # ADX regime gate: in a non-trending 1h market, trend-following entries have
        # no edge — block BOTH the mechanical vote and (below) an AI-driven decide.
        # Off by default until backtest-validated (see GET /bot/backtest COMBINED_ADX_GATED).
        adx_now = (ind_t or {}).get("adx")
        adx_blocks = bool(settings.adx_gate_enabled and adx_now is not None and adx_now < settings.adx_min_trend)

        # 3b. Higher-TF filter: don't fight the 1h trend
        action = entry_action
        if entry_action == "BUY" and bias < 0:
            action = "HOLD"
            reason = f"Blocked — 15m wanted BUY but 1h trend is bearish | {result['reason']}"
        elif entry_action == "SELL" and bias > 0:
            action = "HOLD"
            reason = f"Blocked — 15m wanted SELL but 1h trend is bullish | {result['reason']}"
        elif entry_action in ("BUY", "SELL") and adx_blocks:
            action = "HOLD"
            reason = f"Blocked — 1h ADX {adx_now:.1f} < {settings.adx_min_trend:g} (ranging) | {result['reason']}"
        else:
            reason = f"1h trend {bias_txt} | {result['reason']}"

        # 3c. AGENTS — the PRIMARY decision-maker. Book-derived trading agents each
        # propose a side; the learning ensemble weights them by their LIVE win-rate,
        # decides take/skip (meta-labeling) and the position-size multiplier. This is
        # what makes the bot improve from its own trades. The LLM brain (3d) runs ONLY
        # as a fallback when the agents abstain.
        agent_drove = False
        agent_size_mult = 1.0
        agent_sl_hint = None
        agent_ids: list[str] = []
        ens = None
        try:
            agent_ctx = agents_mod.AgentContext(
                symbol=symbol, price=ind["close"], c_entry=c_entry, c_trend=c_trend,
                ind=ind, supertrend=st, trendline=tn, fvg=fvg, ifvg=ifvg,
                smc=smc_e, smc_trend=smc_t, bias=bias, confluence=result)
            ens = await ensemble.decide(agent_ctx)
        except Exception as e:
            logger.error(f"[{symbol}] agent ensemble failed: {e}")
        if ens and ens["action"] in ("BUY", "SELL"):
            cand = ens["action"]
            if settings.ai_respect_trend_filter and (
                (cand == "BUY" and bias < 0) or (cand == "SELL" and bias > 0)
            ):
                action = "HOLD"
                reason = f"Agents wanted {cand} but blocked by 1h {bias_txt} trend | {ens['reason']}"
            else:
                action = cand
                agent_drove = True
                agent_size_mult = ens["size_mult"]
                agent_sl_hint = ens["sl_hint"]
                agent_ids = ens["agents"]
                reason = f"{ens['reason']} · 1h {bias_txt}"

        # 3d. AI brain (FALLBACK ONLY) — runs when the agents abstained. In 'decide'
        # mode it makes the call + structure SL/TP; otherwise it refines SL/TP. Falls
        # back to the mechanical result above if unavailable/timed out.
        # COOLDOWN: an LLM call can take 15–150s, so at a fast tick cadence we must NOT
        # call it every tick (that overlaps ticks and triggers provider rate-limit storms).
        # Consult it at most once per ai_min_interval_sec PER SYMBOL; agents cover the rest.
        ai_plan = None
        ai_drove = False
        _now = time.monotonic()
        _ai_cooldown_ok = (settings.ai_min_interval_sec <= 0 or
                           _now - _last_ai_call.get(symbol, 0.0) >= settings.ai_min_interval_sec)
        if not agent_drove and _ai_cooldown_ok and ai_brain.available() \
                and settings.ai_mode in ("decide", "refine", "advisory"):
            _last_ai_call[symbol] = _now
            try:
                # The fast pass is confirming a level the deep pass already reasoned
                # about, so it ships a lean snapshot: no backtested edge, no news
                # block. That is not just speed — the full payload trips Groq's
                # free-tier tokens-per-minute ceiling with a 413.
                hist_edge = None if fast else await edge_mod.get_edge(symbol)
                news_ctx = None if fast else await news.ai_context(symbol)
                snapshot = ai_brain.build_snapshot(
                    symbol, ind["close"], c_entry, c_trend, ind, st, tn, fvg, ifvg,
                    ind_t, st_t, tn_t, bias_txt, result["votes"], weights, {},
                    smc_entry=smc_e, smc_trend=smc_t,
                    c_ltf=c_ltf, ind_l=ind_l, st_l=st_l, tn_l=tn_l, smc_ltf=smc_l,
                    historical_edge=hist_edge, news=news_ctx,
                    divergence=divergence, funding=funding_read, orderbook=orderbook_read)
                chain = ai_brain.FAST_PROVIDERS if fast else ai_brain.DEFAULT_PROVIDERS
                ai_plan = await asyncio.to_thread(ai_brain.analyze, snapshot, chain)
            except Exception as e:
                logger.error(f"[{symbol}] AI analysis failed: {e}")

        if ai_plan and settings.ai_mode == "decide":
            cand = ai_plan["action"]
            if cand in ("BUY", "SELL") and ai_plan["confidence"] < settings.ai_min_confidence:
                cand = "HOLD"
            if settings.ai_respect_trend_filter and (
                (cand == "BUY" and bias < 0) or (cand == "SELL" and bias > 0)
            ):
                action = "HOLD"
                reason = f"AI wanted {cand} but blocked by 1h {bias_txt} trend | {ai_plan['reasoning']}"
            elif cand in ("BUY", "SELL") and adx_blocks:
                action = "HOLD"
                reason = f"AI wanted {cand} but blocked by 1h ADX {adx_now:.1f} < {settings.adx_min_trend:g} (ranging) | {ai_plan['reasoning']}"
            else:
                action = cand
                ai_drove = cand in ("BUY", "SELL")
                reason = f"AI {ai_plan['confidence']:.0%} → {cand} · 1h {bias_txt} | {ai_plan['reasoning']}"
        elif ai_plan and settings.ai_mode == "refine":
            ai_drove = action in ("BUY", "SELL")  # mechanical decides; AI supplies SL/TP
            if ai_drove:
                reason = f"{reason} | AI SL/TP · {ai_plan['reasoning']}"

        ai_meta = ({"model": settings.ai_model, "mode": settings.ai_mode,
                    "confidence": ai_plan["confidence"], "reasoning": ai_plan["reasoning"],
                    "invalidation": ai_plan["invalidation"], "proposed_action": ai_plan["action"],
                    "stop_loss": ai_plan["stop_loss"], "take_profits": ai_plan["take_profits"],
                    "drove": ai_drove} if ai_plan else None)
        agent_meta = ({"decision": ens["action"], "confidence": ens["confidence"],
                       "size_mult": ens["size_mult"], "agents": ens["agents"],
                       "proposals": ens["proposals"], "reason": ens["reason"],
                       "drove": agent_drove} if ens else None)
        logger.info(f"[{symbol}] {action} (1h bias={bias_txt}, votes={result['votes']}, "
                    f"agents={'drove' if agent_drove else (ens['action'] if ens else 'off')}, "
                    f"ai={'on' if ai_plan else 'off'}{f' {ai_plan['confidence']:.0%}' if ai_plan else ''})")

        # 4. Execute (position-aware: no pyramiding; bracketed entry from flat)
        order_id = None
        order_status = None
        lots = 0
        sl_price = tp_price = None
        fraction = 0.0
        rr = float(settings.risk_reward)
        price = ind["close"]
        liq_meta = None
        expectancy_meta = None
        if action in ("BUY", "SELL"):
            side = action.lower()
            want_long = action == "BUY"
            try:
                pos = await _delta.get_position_size(symbol)
                if pos != 0 and ((pos > 0) != want_long):
                    # opposite signal -> cancel protective orders, then close
                    pid = await _delta.get_product_id(symbol)
                    for o in await _delta.get_live_orders(symbol):
                        if o.get("reduce_only"):
                            await _delta.cancel_order(o["id"], pid)
                    close_side = "buy" if pos < 0 else "sell"
                    order = await _delta.place_order(symbol, close_side, abs(int(round(pos))), reduce_only=True)
                    await db.bot_state.delete_one({"_id": symbol})
                    order_id = str(order.get("id", ""))
                    order_status = "closed_position"
                    logger.info(f"Closed {pos} {symbol} via {close_side} {abs(int(round(pos)))} (cancelled brackets)")
                elif pos != 0:
                    order_status = "held (already in position)"
                    logger.info(f"Signal {action} but already {('LONG' if pos>0 else 'SHORT')} {abs(pos)} — holding (no pyramiding)")
                elif not allow_entry:
                    order_status = "flat (managed only — not the active chart symbol)"
                else:
                    # concurrency cap across all coins
                    n_open = sum(1 for p in await _delta.get_positions() if p.get("size"))
                    if n_open >= settings.max_concurrent_positions:
                        order_status = f"skipped: max {settings.max_concurrent_positions} concurrent positions open"
                        logger.info(f"{symbol} flat + {action} signal but {n_open} positions already open — skipping")
                        raise _SkipEntry()
                    # liquidity gate (spread): a market this wide costs more to enter and
                    # exit than the edge is worth. Shadow mode logs this without blocking.
                    ok_liq, liq_txt, liq_meta = await _liquidity_gate(symbol, want_long)
                    if not ok_liq:
                        order_status = f"skipped: {liq_txt}"
                        logger.info(f"{symbol} entry blocked — liquidity: {liq_txt}")
                        raise _SkipEntry()
                    # news blackout: don't open a NEW trade into a high-impact release (whipsaw risk)
                    blocked, ev = await news.in_blackout()
                    if blocked:
                        order_status = (f"skipped: news blackout — {ev['impact']}-impact {ev['currency']} "
                                        f"'{ev['title']}' within {settings.news_blackout_min}m")
                        logger.info(f"{symbol} entry blocked — news blackout: {ev['currency']} {ev['title']}")
                        raise _SkipEntry()
                    # daily loss circuit breaker: stop opening new trades once down the day's limit
                    wallet = await _delta.get_wallet()
                    total_bal = float(wallet.get("balance") or 0)
                    avail = float(wallet.get("available_balance") or total_bal)
                    breached, day_pnl, day_limit = await _daily_loss_exceeded(total_bal)
                    if breached:
                        order_status = f"skipped: daily loss limit hit (today {day_pnl} <= {day_limit})"
                        logger.warning(f"{symbol} entry blocked — daily loss limit reached (today ${day_pnl} <= ${day_limit})")
                        raise _SkipEntry()
                    # expectancy/profitability gate: only trade vote combinations that have
                    # REAL backtested edge on this symbol (not just enough votes agreeing).
                    expectancy_meta = await edge_mod.blended_expectancy(symbol, result["votes"], action, weights)
                    if settings.expectancy_gate_enabled:
                        if expectancy_meta["blended_exp"] is None:
                            if not settings.expectancy_gate_fail_open:
                                order_status = "skipped: expectancy gate — no qualifying backtested evidence for this vote"
                                logger.info(f"{symbol} entry blocked — expectancy gate: no qualifying evidence")
                                raise _SkipEntry()
                        elif expectancy_meta["blended_exp"] < settings.expectancy_gate_min_R:
                            order_status = (f"skipped: expectancy gate — blended {expectancy_meta['blended_exp']:+.3f}R "
                                            f"< min {settings.expectancy_gate_min_R:+.3f}R")
                            logger.info(f"{symbol} entry blocked — expectancy gate: "
                                        f"{expectancy_meta['blended_exp']:+.3f}R < {settings.expectancy_gate_min_R:+.3f}R")
                            raise _SkipEntry()
                    # "big" trade: most indicators agree -> allow a wider stop + larger
                    # target with a smaller position (special case).
                    agree = result["buy_score"] if want_long else result["sell_score"]
                    total = max(len(enabled), 1)
                    big = (agree / total) >= settings.big_trade_min_agree

                    # flat -> place SL where the idea is invalidated. Use the AI's
                    # structure stop (clamped to a sane band) when it drove the call,
                    # otherwise the mechanical ATR/SuperTrend/structure combination.
                    sl_price = sl_method = None
                    if ai_drove and ai_plan:
                        clamped = _clamp_sl(want_long, price, ai_plan.get("stop_loss"))
                        if clamped is not None:
                            sl_price, sl_method = clamped, "ai"
                    elif agent_drove and agent_sl_hint is not None:
                        clamped = _clamp_sl(want_long, price, agent_sl_hint)
                        if clamped is not None:
                            sl_price, sl_method = clamped, "agent"
                    if sl_price is None:
                        sl_price, _sl_cands, sl_method = _combined_sl(want_long, price, c_entry, st)
                        if ai_drove:
                            sl_method += "+ai-fallback"
                        elif agent_drove:
                            sl_method += "+agent-fallback"

                    # Per-symbol POINT limits (ETH): cap the stop distance (tight for normal,
                    # wider for big trades). Keeps the structure stop if it's already tighter.
                    prof = _symbol_profile(symbol, big)
                    if prof:
                        sl_dist = abs(price - sl_price)
                        sl_dist = min(max(sl_dist, prof["sl_min"]), prof["sl_max"])
                        sl_price = price - sl_dist if want_long else price + sl_dist
                        sl_method = f"{sl_method}|pts<= {prof['sl_max']:g}" + ("|BIG" if big else "")
                    risk = abs(price - sl_price)
                    if prof:
                        # ETH uses a fixed point target ladder (R:R already >= 2.5 by config),
                        # so the structure-reachability gate is skipped — take the trade.
                        ok, room_info = True, "point-based target"
                    else:
                        ok, room_info = _room_for_1to2(want_long, price, risk, c_trend)
                    if not ok:
                        order_status = f"skipped: 1:2 not reachable ({room_info})"
                        logger.info(f"Skip {side}: {room_info}")
                    else:
                        cv = await _delta.get_contract_value(symbol)
                        # Leverage: as high as configured (50x), but auto-lowered so the
                        # liquidation price sits beyond the stop (SL always triggers first).
                        lev = _safe_leverage(price, risk)
                        await _delta.set_leverage(symbol, lev)

                        # CAPITAL-BASED sizing: deploy a fixed % of balance as margin
                        # (50% normal, less for big trades -> smaller lots). When agents
                        # drove the call, scale by the ensemble's Kelly/probability size
                        # multiplier (AFML bet sizing) — conviction trades get more capital.
                        cap_pct = settings.big_trade_capital_pct if big else settings.position_capital_pct
                        if agent_drove:
                            cap_pct *= agent_size_mult

                        # Correlation-aware dampening: don't silently double correlated
                        # exposure when another open position (different symbol, same
                        # direction) is highly correlated with this one.
                        corr_used = None
                        if settings.correlation_check_enabled:
                            try:
                                for p in await _delta.get_positions():
                                    other_sym = p.get("product_symbol")
                                    other_size = float(p.get("size") or 0)
                                    if not other_sym or other_sym == symbol or not other_size:
                                        continue
                                    if (other_size > 0) != want_long:
                                        continue  # opposite direction — no correlated stacking
                                    corr_used = await portfolio_risk.realized_correlation(
                                        _delta, symbol, other_sym, settings.correlation_timeframe_min,
                                        settings.correlation_lookback_bars)
                                    if corr_used is not None and corr_used >= settings.correlation_high_threshold:
                                        cap_pct *= settings.correlation_dampen_factor
                                        logger.info(f"{symbol} sizing dampened {settings.correlation_dampen_factor:g}x "
                                                    f"— {corr_used:.2f} correlated with open {other_sym} {side}")
                                        break
                            except Exception as e:
                                logger.warning(f"{symbol} correlation check failed ({type(e).__name__}) — using full size")

                        # Volatility-adjusted sizing: scale capital deployed inversely with
                        # current ATR%, so risk normalizes across volatility regimes instead
                        # of a flat capital allocation regardless of how choppy price is.
                        vol_scalar_used = None
                        if settings.vol_sizing_enabled and price > 0:
                            atr_list = _atr(c_entry, settings.atr_period)
                            atr_now = atr_list[-1] if atr_list else 0.0
                            if atr_now > 0:
                                atr_pct = atr_now / price * 100
                                vol_scalar_used = min(max(settings.vol_ref_atr_pct / atr_pct,
                                                          settings.vol_scalar_min), settings.vol_scalar_max)
                                cap_pct *= vol_scalar_used

                        margin_usd = total_bal * cap_pct / 100.0
                        # respect the available-balance ceiling
                        margin_usd = min(margin_usd, avail * settings.margin_cap_pct)
                        notional = margin_usd * lev
                        lots = int(notional / (price * cv)) if price > 0 and cv > 0 else 0
                        lots = max(lots, 1)
                        fraction = round(cap_pct, 2)  # for logging: % of capital used as margin
                        risk_dollars = risk * lots * cv  # $ at risk if the stop is hit

                        # dynamic reward:risk ceiling (floor 1:2) from conviction + 1h strength
                        strength = st_t["latest"].get("strength", 0) if st_t else 0
                        aligned = (bias > 0 and want_long) or (bias < 0 and not want_long)
                        rr = max(_rr_target(agree, strength, aligned), settings.risk_reward)

                        # size-aware exit check: now that lots are known, confirm the book
                        # could actually absorb closing this position. Shadow mode logs
                        # this without blocking (see liq_meta on the earlier spread check).
                        ok_exit, exit_txt, liq_meta = await _liquidity_gate(symbol, want_long, lots)
                        if not ok_exit:
                            order_status = f"skipped: {exit_txt}"
                            logger.info(f"{symbol} entry blocked — {exit_txt}")
                            raise _SkipEntry()

                        # 1) ENTRY (market, no bracket — we manage TP/SL ourselves)
                        order = await _delta.place_order(symbol, side, lots)
                        order_id = str(order.get("id", ""))
                        order_status = order.get("state", "unknown")

                        # The traded fill price (recorded for PnL/breakeven only). The trade
                        # is MANAGED on the MARK price so SL/TP match the chart and trigger on
                        # mark — on the thin testnet the fill can diverge from mark, but the
                        # geometry (20pt stop, 50-60pt target) stays anchored to mark.
                        posdoc = await _delta.get_position(symbol)
                        fill_price = float(posdoc.get("entry_price") or price)
                        entry_ref = price            # mark decision price = management basis
                        # risk stays mark-based (abs(price - sl_price)); do NOT re-anchor to the fill

                        # 2) TP levels (anchored to the mark entry reference).
                        if prof:
                            # ETH: fixed target ladder inside the point band (50–60 pts
                            # normal, wider for big). 1 TP -> just the top; 2 TPs -> band edges.
                            lo, hi = prof["tp_min"], prof["tp_max"]
                            dists = [hi] if settings.max_tps <= 1 else [lo, hi]
                            tps = [round(entry_ref + d, 1) if want_long else round(entry_ref - d, 1) for d in dists]
                            sizes = _tp_sizes(lots, [(p, None) for p in tps])
                            tp_source = "eth-points" + ("-BIG" if big else "")
                        else:
                            # other symbols: AI structure targets (validated), else structure snap
                            accepted = _valid_ai_tps(want_long, entry_ref, risk, ai_plan["take_profits"]) if (ai_drove and ai_plan) else []
                            if accepted:
                                tps = [p for p, _ in accepted]
                                sizes = _tp_sizes(lots, accepted)
                                tp_source = "ai"
                            else:
                                tps = _target_levels(want_long, entry_ref, risk, c_entry, c_trend, fvg, rr)
                                sizes = _tp_sizes(lots, [(p, None) for p in tps])
                                tp_source = "ai-fallback" if ai_drove else "structure"
                        tp_price = tps[0] if tps else None
                        close_side = "sell" if want_long else "buy"

                        # 3) place partial take-profits + full stop-loss (reduce-only stops)
                        placed_tps = []
                        for lvl, sz in zip(tps, sizes):
                            if sz <= 0:
                                continue
                            try:
                                await _delta.place_stop_order(symbol, close_side, sz, lvl, "take_profit_order")
                                placed_tps.append({"price": lvl, "size": sz})
                            except Exception as e:
                                logger.error(f"TP order failed @ {lvl}: {e}")
                        try:
                            await _delta.place_stop_order(symbol, close_side, lots, sl_price, "stop_loss_order")
                        except Exception as e:
                            logger.error(f"SL order failed @ {sl_price}: {e} — position is UNPROTECTED")

                        # 4) persist trade state for breakeven management
                        await db.bot_state.replace_one(
                            {"_id": symbol},
                            {"_id": symbol, "side": side, "size": lots, "entry": entry_ref,
                             "fill_price": fill_price,
                             "sl": sl_price, "tps": placed_tps, "rr": rr, "be_moved": False,
                             "votes": result["votes"], "agents": agent_ids,
                             "shadow_votes": result.get("shadow_votes") or {},
                             "risk_dollars": round(risk_dollars, 4),
                             "sl_method": sl_method, "tp_source": tp_source, "ai": ai_meta,
                             "leverage": lev, "big_trade": big, "capital_pct": cap_pct,
                             "opened_at": datetime.now(timezone.utc)},
                            upsert=True,
                        )
                        slip = fill_price - entry_ref
                        logger.info(f"Opened {side} {lots} lots @ {lev}x {'[BIG] ' if big else ''}| margin {fraction:.0f}% cap "
                                    f"(risk ${risk_dollars:.2f}) | mark-entry {entry_ref:.1f} fill {fill_price:.1f} "
                                    f"(slip {slip:+.1f}) SL {sl_price:.1f} ({sl_method}) "
                                    f"TPs {[t['price'] for t in placed_tps]} ({tp_source})")
            except _SkipEntry:
                pass  # order_status already set (e.g. concurrency cap)
            except Exception as e:
                logger.error(f"Order failed: {e}")
                order_status = f"error: {e}"

        # 5. Log to MongoDB
        log_doc = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": symbol,
            "action": action,
            "reason": reason,
            "price": price,
            "quantity": lots if action != "HOLD" else 0,
            "indicators": {
                "ema_fast": ind["ema_fast"],
                "ema_slow": ind["ema_slow"],
                "ema_signal": ind["ema_signal"],
                "rsi": ind["rsi"],
                "rsi_signal": ind["rsi_signal"],
                "breakout_signal": ind["breakout_signal"],
                "breakout_level": ind["breakout_level"],
                "macd": ind.get("macd"),
                "macd_signal_line": ind.get("macd_signal_line"),
                "macd_hist": ind.get("macd_hist"),
                "macd_signal": ind.get("macd_signal"),
            },
            "votes": result["votes"],
            "supertrend": st["latest"] if st else None,
            "trendline": tn["latest"] if tn else None,
            "fvg": fvg["latest"] if fvg else None,
            "ifvg": ifvg["latest"] if ifvg else None,
            "smc": {
                "entry_trend": (smc_e or {}).get("trend"),
                "trend_trend": (smc_t or {}).get("trend"),
                "structure_break": (smc_e or {}).get("structure_break"),
                "recent_sweep": (smc_e or {}).get("recent_sweep"),
                "zone": ((smc_e or {}).get("premium_discount") or {}).get("zone"),
            } if smc_e else None,
            "timeframes": {
                "entry_tf": settings.entry_timeframe,
                "trend_tf": settings.trend_timeframe,
                "trend_bias": bias_txt,
                "entry_action": entry_action,
                "trend_supertrend": st_t["latest"] if st_t else None,
            },
            "sizing": {"risk_pct": round(fraction, 2), "leverage": settings.leverage,
                       "reward_risk": round(rr, 1),
                       "stop_loss": round(sl_price, 2) if sl_price else None,
                       "take_profit": round(tp_price, 2) if tp_price else None},
            "ai": ai_meta,
            "agents": agent_meta,
            "liquidity": liq_meta,
            "expectancy_gate": expectancy_meta,
            "order_id": order_id,
            "order_status": order_status,
            "paper_trade": True,
        }
        await db.trade_logs.insert_one(log_doc)

        # 6. Arm/disarm the fast execution loop. Only the DEEP pass arms — letting
        # the fast pass re-arm would let one analysis fire repeatedly.
        if not fast:
            try:
                held = await _delta.get_position_size(symbol)
            except Exception:
                held = 0
            _update_watch(symbol, ai_plan, price, c_ltf,
                          allow_entry=allow_entry, in_position=bool(held))

    except Exception as e:
        logger.exception(f"[{symbol}] process error: {e}")


async def bot_tick():
    """
    One scheduler iteration. Processes the ACTIVE chart symbol (new entries allowed)
    plus EVERY symbol that still has an open position or lingering trade state, so
    switching coins never orphans a live trade — old positions keep being managed
    (breakeven, opposite-close, cleanup) while the active symbol trades fresh.
    """
    trade_syms = [s.strip().upper() for s in settings.trade_symbols.split(",") if s.strip()]
    tracked = set(trade_syms)
    try:
        for p in await _delta.get_positions():
            if p.get("size") and p.get("product_symbol"):
                tracked.add(p["product_symbol"])
    except Exception as e:
        logger.error(f"tracked-positions fetch failed: {e}")
    try:
        async for s in db.bot_state.find({}, {"_id": 1}):
            tracked.add(s["_id"])
    except Exception:
        pass

    weights = await autotune.get_weights()  # live-tuned vote weights (cached ~30s)
    # All configured trade symbols can open NEW trades simultaneously; any other
    # symbol with a stray open position is managed (closed) but not re-entered.
    # Held for the whole pass so the fast loop cannot place an order mid-tick.
    async with _tick_lock:
        for sym in tracked:
            await _process_symbol(sym, allow_entry=(sym in trade_syms), weights=weights)


async def fast_tick():
    """Latency loop between deep ticks.

    Two jobs, both cheap:
      1. Manage every open position on every pass, so a breakeven/cleanup move is
         not up to 2 minutes late. Purely mechanical — no AI call.
      2. For symbols the deep tick ARMED, compare the live price to the stored
         trigger. Only when it is crossed does this spend an AI call, and then on
         FAST_PROVIDERS (Groq ~1-2s) via the normal `_process_symbol` path, so
         every guardrail — position cap, news blackout, loss limit, SL/TP sizing —
         still applies to the entry.
    """
    if not settings.fast_check_enabled or not _bot_running:
        return
    # The deep tick is mid-flight; it owns execution right now.
    if _tick_lock.locked():
        return
    async with _tick_lock:
        try:
            open_syms = {p["product_symbol"] for p in await _delta.get_positions()
                         if p.get("size") and p.get("product_symbol")}
        except Exception as e:
            logger.error(f"fast_tick position fetch failed: {e}")
            open_syms = set()
        for sym in open_syms:
            try:
                await _manage_open_position(sym)
            except Exception as e:
                logger.error(f"[{sym}] fast manage failed: {e}")

        now = datetime.now(timezone.utc)
        weights = None
        for sym, w in list(_watch.items()):
            if now >= w["expires"]:
                _watch.pop(sym, None)
                logger.info(f"[{sym}] fast-watch expired without triggering")
                continue
            if sym in open_syms:          # filled in the meantime
                _watch.pop(sym, None)
                continue
            try:
                px = float((await _delta.get_ticker(sym)).get("mark_price") or 0)
            except Exception as e:
                logger.error(f"[{sym}] fast ticker fetch failed: {e}")
                continue
            if not px or not _trigger_hit(w, px):
                continue
            # Consume the watch BEFORE acting so a slow entry cannot double-fire.
            _watch.pop(sym, None)
            logger.info(f"[{sym}] fast-watch TRIGGERED {w['side']} @ {w['trigger']} "
                        f"(mark {px}) — confirming on fast providers")
            if weights is None:
                weights = await autotune.get_weights()
            await _process_symbol(sym, allow_entry=True, weights=weights, fast=True)


def start_bot():
    global _bot_running
    if _bot_running:
        return {"status": "already_running"}
    # Sub-minute cadence wins when set, so the loop can run every 30s.
    secs = settings.check_interval_seconds
    # A sub-minute loop is ~2,880 ticks/day. That is only affordable while the free
    # first rung (Ox Alpha via OpenRouter) is answering; without it every tick walks
    # down to metered/subscription providers. Warn loudly rather than silently bill.
    if 0 < secs < 60 and not ai_brain._has_openrouter_key():
        logger.warning(
            f"Sub-minute cadence ({secs}s = ~{86400 // secs} ticks/day) with NO "
            "OPENROUTER_API_KEY set — the free Ox Alpha rung is being skipped, so "
            "every tick falls through to rate-limited, metered and subscription "
            "providers. Set the key, or raise CHECK_INTERVAL_SECONDS."
        )
    trigger = (IntervalTrigger(seconds=secs) if secs > 0
               else IntervalTrigger(minutes=settings.check_interval_minutes))
    scheduler.add_job(
        bot_tick,
        trigger=trigger,
        id="bot_tick",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc),  # run immediately on start
        # A tick can outlast a 30s interval (ai_timeout_sec is 150). Never stack
        # overlapping ticks — they would double-read the book and can double-enter.
        # Skip the backlog and run once when the previous tick finishes.
        max_instances=1,
        coalesce=True,
        misfire_grace_time=None,
    )
    if settings.fast_check_enabled:
        scheduler.add_job(
            fast_tick,
            trigger=IntervalTrigger(seconds=settings.fast_check_seconds),
            id="fast_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=None,
        )
    # Start the scheduler only once; subsequent start/stop just add/remove the job.
    if not scheduler.running:
        scheduler.start()
    _bot_running = True
    cadence = f"{secs} seconds" if secs > 0 else f"{settings.check_interval_minutes} minutes"
    logger.info(f"Bot started — checking every {cadence}.")
    return {"status": "started"}


def stop_bot():
    global _bot_running
    if not _bot_running:
        return {"status": "not_running"}
    for job_id in ("bot_tick", "fast_tick"):
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass
    # Drop any armed watch — it must never survive a stop and fire on restart.
    _watch.clear()
    # Leave the scheduler running (just without the job) so it can be restarted.
    _bot_running = False
    logger.info("Bot stopped.")
    return {"status": "stopped"}


def set_symbol(symbol: str) -> dict:
    """Change the symbol the bot trades, at runtime. Takes effect on the next tick."""
    settings.trading_symbol = symbol
    logger.info(f"Bot trading symbol changed to {symbol}.")
    return {"symbol": symbol, "running": _bot_running}


def bot_status() -> dict:
    syms = [s.strip().upper() for s in settings.trade_symbols.split(",") if s.strip()]
    return {
        "running": _bot_running,
        "interval_minutes": settings.check_interval_minutes,
        # Sub-minute cadence, when configured, is what actually drives the loop.
        "interval_seconds": settings.check_interval_seconds or None,
        "fast_check_seconds": settings.fast_check_seconds if settings.fast_check_enabled else None,
        "armed": {s: {"side": w["side"], "trigger": w["trigger"]} for s, w in _watch.items()},
        "symbol": settings.trading_symbol,
        "symbols": syms,
        "next_run": (
            str(scheduler.get_job("bot_tick").next_run_time)
            if _bot_running and scheduler.get_job("bot_tick")
            else None
        ),
    }
