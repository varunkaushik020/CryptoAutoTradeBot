"""
AI trading brain.

Turns a multi-timeframe market snapshot into a validated trade plan by asking
Claude (via the local Claude Code CLI, authenticated with the user's subscription
— NO Anthropic API key required). The CLI is invoked in headless JSON mode:

    claude -p --output-format json --model <model>   (prompt piped on stdin)

The model returns a structured plan (direction, structure-based stop-loss, TP1/2/3,
confidence, reasoning). All numeric guardrails (risk sizing, min 1:2 R:R, SL clamps)
are enforced by the caller in scheduler.py — the AI proposes, code disposes.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger("bot.ai_brain")

_CLI_CACHE: Optional[str] = None


def _ver_key(p: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", p)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def find_claude_cli() -> Optional[str]:
    """Locate the Claude Code binary (subscription auth). Cached after first hit."""
    global _CLI_CACHE
    if _CLI_CACHE and Path(_CLI_CACHE).exists():
        return _CLI_CACHE
    if settings.claude_cli_path and Path(settings.claude_cli_path).exists():
        _CLI_CACHE = settings.claude_cli_path
        return _CLI_CACHE

    home = Path.home()
    patterns = [
        # Claude Desktop app (Windows Store package) bundles claude-code/<ver>/claude.exe
        str(home / "AppData/Local/Packages/*/LocalCache/Roaming/Claude/claude-code/*/claude.exe"),
        str(home / "AppData/Roaming/Claude/claude-code/*/claude.exe"),
        # npm global install
        str(home / "AppData/Roaming/npm/claude.cmd"),
        # posix installs
        str(home / ".local/bin/claude"),
        "/usr/local/bin/claude",
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates += glob.glob(pat)
    w = which("claude")
    if w:
        candidates.append(w)
    candidates = [c for c in candidates if Path(c).exists()]
    if not candidates:
        return None
    candidates.sort(key=_ver_key)  # newest version last
    _CLI_CACHE = candidates[-1]
    logger.info(f"AI brain using Claude CLI: {_CLI_CACHE}")
    return _CLI_CACHE


def _has_api_key() -> bool:
    k = (settings.anthropic_api_key or "").strip()
    return bool(k) and k != "paste-your-key-here"


def _has_gemini_key() -> bool:
    k = (settings.gemini_api_key or "").strip()
    return bool(k) and k != "paste-your-key-here"


def _has_groq_key() -> bool:
    k = (settings.groq_api_key or "").strip()
    return bool(k) and k != "paste-your-key-here"


def _has_openrouter_key() -> bool:
    k = (settings.openrouter_api_key or "").strip()
    return bool(k) and k != "paste-your-key-here"


def claude_available() -> bool:
    """True if Claude can be reached — direct API key, or the local Claude Code CLI."""
    return _has_api_key() or find_claude_cli() is not None


def available() -> bool:
    return settings.ai_enabled and (_has_gemini_key() or claude_available())


# --------------------------------------------------------------------------- #
#  Snapshot: compress the raw market state into a compact, model-friendly dict
# --------------------------------------------------------------------------- #
def _swings(candles: list[dict], highs: bool, span: int = 2) -> list[dict]:
    """Local swing highs/lows with a `span`-bar window each side."""
    out = []
    n = len(candles)
    for i in range(span, n - span):
        c = candles[i]
        if highs:
            v = c["high"]
            if all(v >= candles[j]["high"] for j in range(i - span, i + span + 1) if j != i) and \
               v > candles[i - 1]["high"]:
                out.append({"time": int(c["time"]), "price": round(v, 2)})
        else:
            v = c["low"]
            if all(v <= candles[j]["low"] for j in range(i - span, i + span + 1) if j != i) and \
               v < candles[i - 1]["low"]:
                out.append({"time": int(c["time"]), "price": round(v, 2)})
    return out


def _fmt_candles(candles: list[dict], n: int) -> list[list]:
    """Last n candles as compact [o,h,l,c] rows (2dp)."""
    return [[round(c["open"], 2), round(c["high"], 2), round(c["low"], 2), round(c["close"], 2)]
            for c in candles[-n:]]


def _near_zones(zones, price, pct=3.0, limit=6):
    """FVG/IFVG zones within `pct`% of price, nearest first."""
    if not zones:
        return []
    out = []
    for z in zones:
        top, bot = z.get("top"), z.get("bottom")
        if top is None or bot is None:
            continue
        mid = (top + bot) / 2
        if price > 0 and abs(mid - price) / price * 100 <= pct:
            out.append({"top": round(top, 2), "bottom": round(bot, 2),
                        "bull": bool(z.get("isbull", z.get("dir", 0) == 1))})
    out.sort(key=lambda z: abs((z["top"] + z["bottom"]) / 2 - price))
    return out[:limit]


def build_snapshot(symbol, price, c_entry, c_trend, ind, st, tn, fvg, ifvg,
                   ind_t, st_t, tn_t, bias_txt, votes, weights, account,
                   smc_entry=None, smc_trend=None,
                   c_ltf=None, ind_l=None, st_l=None, tn_l=None, smc_ltf=None,
                   historical_edge=None, news=None,
                   divergence=None, funding=None, orderbook=None) -> dict:
    """Assemble everything the model needs to reason about the trade, including the
    deterministic Smart Money Concepts read for each timeframe (1h bias, 15m decision,
    5m timing)."""
    et = settings.entry_timeframe
    tt = settings.trend_timeframe
    lt = settings.ltf_timeframe

    def tf_block(candles, ind_, st_, tn_):
        return {
            "ema_fast": _r(ind_.get("ema_fast")),
            "ema_slow": _r(ind_.get("ema_slow")),
            "rsi": _r(ind_.get("rsi")),
            "macd": {"line": _r(ind_.get("macd"), 4), "signal": _r(ind_.get("macd_signal_line"), 4),
                     "hist": _r(ind_.get("macd_hist"), 4), "state": ind_.get("macd_signal")},
            "supertrend": (st_.get("latest") if st_ else None),
            "trendline": (tn_.get("latest") if tn_ else None),
            "recent_swing_highs": _swings(candles, True)[-6:],
            "recent_swing_lows": _swings(candles, False)[-6:],
            "bb": {"mid": _r(ind_.get("bb_mid")), "upper": _r(ind_.get("bb_upper")), "lower": _r(ind_.get("bb_lower"))},
            "kc": {"mid": _r(ind_.get("kc_mid")), "upper": _r(ind_.get("kc_upper")), "lower": _r(ind_.get("kc_lower"))},
            "squeeze": ind_.get("squeeze_signal"),
            "adx": _r(ind_.get("adx"), 1),
            "vwap": _r(ind_.get("vwap")),
        }

    snap = {
        "symbol": symbol,
        "current_price": round(price, 2),
        "decision_timeframe_min": et,
        "trend_timeframe_min": tt,
        "timing_timeframe_min": lt,
        "trend_bias_1h": bias_txt,
        "entry_tf": {
            **tf_block(c_entry, ind, st, tn),
            "last_20_ohlc": _fmt_candles(c_entry, 20),
            "fvg_zones_near": _near_zones((fvg or {}).get("unmitigated"), price),
            "ifvg_zones_near": _near_zones((ifvg or {}).get("zones"), price),
            "smc": smc_entry,
            "divergence": (divergence or {}).get("latest"),
        },
        "trend_tf": {**tf_block(c_trend, ind_t, st_t, tn_t), "smc": smc_trend},
        "strategy_votes": votes,
        "strategy_weights": {k: round(v, 2) for k, v in (weights or {}).items()},
        # backtested edge on THIS symbol (clean mark data): {strat: {pf, exp, n}} — use it
        # to weight the votes; trust high-PF signals, discount PF~1.0 as noise.
        "historical_edge": historical_edge or {},
        "account": account,
        "risk_rules": {
            "min_reward_risk": settings.risk_reward,
            "risk_pct_range": [settings.risk_min_pct, settings.risk_max_pct],
            "sl_distance_pct_bounds": [settings.min_sl_pct, settings.max_sl_pct],
            "leverage": settings.leverage,
        },
    }
    # 5m timing timeframe (analyzed for entry timing/confirmation; not the decision TF)
    if ind_l is not None:
        snap["timing_tf"] = {**tf_block(c_ltf or [], ind_l, st_l, tn_l), "smc": smc_ltf}
    # Real-time news + upcoming high-impact economic events (see `news` guidance in system prompt).
    if news:
        snap["news"] = news
    # Crypto-native confluence, both optional (see `funding`/`orderbook` guidance in
    # system prompt): perpetual funding-rate bias, and short-horizon L2 book skew.
    if funding:
        snap["funding"] = funding
    if orderbook:
        snap["orderbook"] = orderbook
    return snap


def _r(v, nd=2):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
#  Prompt + call
# --------------------------------------------------------------------------- #
_SYSTEM = """You are an elite crypto-futures trader who trades Smart Money Concepts (SMC / ICT) on a live Delta \
Exchange account. You get a compact multi-timeframe snapshot with THREE timeframes: `trend_tf` (1h, sets bias), `entry_tf` \
(15m — the DECISION timeframe you trade on), and `timing_tf` (5m — used ONLY to refine entry timing/confirmation, \
never to override the 15m decision). Each carries EMAs, RSI, MACD (12/26/9 line/signal/histogram + state, for \
momentum confirmation), SuperTrend AI, trendline-break state, fair-value gaps, and — most importantly — a \
precomputed SMC read under `smc`: market structure (swings labelled HH/HL/LH/LL + trend), the latest BOS/CHoCH, liquidity \
(equal highs/lows, prev-day high/low, buy/sell-side pools), any recent liquidity sweep, order blocks (with a \
`mitigated` flag), and premium/discount (equilibrium + discount OTE zone). These SMC levels are computed from real \
candles — trust them over eyeballing the OHLC.

Trade the SMC playbook. Only take A+ setups:

BUY (mirror for SELL):
- 1h (trend_tf.smc) structure is bullish OR just printed a bullish CHoCH (reversal).
- Price has swept sell-side liquidity (recent_sweep dir bullish / took equal-lows or prev-day-low) then reclaimed.
- 15m confirms with a bullish BOS or CHoCH in your direction.
- Entry is into a DISCOUNT array: an unmitigated bullish order block and/or a fair-value gap, ideally at/below \
equilibrium (premium_discount.zone == "discount", inside discount_ote is best).
- Target the opposing liquidity: buy-side pools / equal-highs / prev-day-high / the range high.

Hard rules you MUST follow:
- Decide on the 15m (entry_tf). Use the 5m (timing_tf) only to CONFIRM the trigger (e.g. a 5m CHoCH/BOS or \
momentum turn in your direction) and sharpen entry — never take a trade the 15m doesn't support, and don't let \
5m noise flip a clean 15m read.
- `historical_edge` is each strategy's BACKTESTED profit-factor (pf) and expectancy on THIS symbol \
(clean data, no execution noise). Weight the `strategy_votes` by it: strongly trust a signal with pf >= 1.3, \
treat pf around 1.0 as noise, and be skeptical of pf < 1.0. Favor the setups that actually have edge here.
- Never fight the 1h trend bias. If bias is bullish, only BUY or HOLD; if bearish, only SELL or HOLD.
- Prefer discount entries for longs and premium entries for shorts. Do NOT buy into premium or sell into discount \
unless a fresh CHoCH + sweep justifies a reversal.
- Put the stop where structure is INVALIDATED — beyond the order block / swept swing that gave the entry (not a \
fixed distance). Stop distance must stay within the given sl_distance_pct_bounds (% of price).
- Provide AT MOST TWO take-profits (one is fine). The FIRST must be at least 2R at the next real liquidity pool; \
an optional TP2 sits further at the next pool / range extreme. size_pct must sum to ~1.0 (e.g. 0.5, 0.5 or a \
single 1.0), nearest-first. Never return three take-profits.
- Avoid chasing: if price already made a large impulsive move into premium/discount with no fresh sweep+OB, HOLD.
- A `news` block may be present: `upcoming_high_impact` lists economic releases with `in_min` (minutes \
until — negative means just released), currency, forecast vs previous; `latest_headlines` carry a rough \
tone; `headline_tone` is the net read. If a High-impact release for a relevant currency is imminent \
(small positive `in_min`, roughly < 30), treat it as elevated whipsaw risk — prefer HOLD or demand a \
cleaner setup and a tighter structural stop. Let `headline_tone` GENTLY tilt conviction, but never let a \
headline override a clean SMC read or invent a trade the structure doesn't support.
- If there is no clean SMC setup with a reachable 2R to real liquidity, return HOLD. Be selective — no forced trades.
- Each timeframe also carries `bb`/`kc`/`squeeze` (Bollinger/Keltner squeeze state — "SQUEEZE_ON" means volatility \
is compressed and building, "RELEASE_UP"/"RELEASE_DOWN" means it just let go in that direction), `adx` (trend \
strength, 0-100), and `vwap` (session volume-weighted average price, when available). Treat `adx` on trend_tf \
below ~20 as a warning that the 1h market is ranging — trend-following reads (EMA, SuperTrend, trendline) are \
less reliable there, so demand a cleaner SMC setup before trusting a breakout. A squeeze release in your \
direction is supportive confluence, never a standalone reason to trade.
- `entry_tf.divergence`, when present, is the latest RSI/price divergence at a swing point ("regular" = reversal, \
"hidden" = trend continuation). Treat it as one more piece of confluence for or against the setup — not an \
override of the SMC read.
- An optional `funding` block (perpetual funding rate + open interest) may be present: `extreme` is "high" \
(crowded longs paying heavily — mild contrarian lean against more upside), "low" (crowded shorts — mild \
contrarian lean against more downside), or null (not extreme / not enough history). Let it gently tilt \
conviction at most — never let it override a clean SMC setup or invent a trade the structure doesn't support.
- An optional `orderbook` block (top-of-book bid/ask volume imbalance) may be present: it is a SECONDS-scale \
signal, useful only the way `timing_tf` is — to sharpen entry timing — never to override the 15m decision.

Respond with ONLY minified JSON (no markdown, no prose) matching exactly:
{"action":"BUY|SELL|HOLD","confidence":0.0-1.0,"entry":<number>,"stop_loss":<number>,\
"take_profits":[{"price":<number>,"size_pct":<0..1>},...],"reasoning":"<2-3 sentences citing the SMC elements>",\
"invalidation":"<1 sentence>"}"""


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    # strip code fences if present, then grab the outermost {...}
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _call_openrouter(snapshot_json: str, system: str = None, user_prefix: str = "MARKET SNAPSHOT:\n",
                     model: str = None) -> Optional[str]:
    """OpenRouter (OpenAI-compatible) call — Ox Alpha by default.

    FIRST rung of every chain: free and 1M-context, so it carries the routine work
    before any rate-limited, metered or subscription-backed provider is reached.

    No `response_format` is sent: Ox Alpha is a stealth model whose JSON-mode
    support is undocumented, and a 400 here would waste the free rung. The system
    prompt already demands JSON and `_extract_json` tolerates fences/prose.
    """
    if not _has_openrouter_key():
        return None
    model = (model or settings.openrouter_model or "stealth/ox-alpha").strip()
    url = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": settings.openrouter_max_output_tokens,
        "messages": [
            {"role": "system", "content": system or _SYSTEM},
            {"role": "user", "content": user_prefix + snapshot_json},
        ],
    }
    try:
        r = httpx.post(url, json=body, timeout=settings.ai_timeout_sec, headers={
            "Authorization": f"Bearer {settings.openrouter_api_key.strip()}",
            "HTTP-Referer": settings.openrouter_referer,
            "X-Title": settings.openrouter_title,
        })
    except Exception as e:
        logger.error(f"OpenRouter request error: {e} — trying next provider.")
        return None
    if r.status_code != 200:
        # A stealth model can be withdrawn without notice; a 404 here means Ox Alpha
        # is gone and OPENROUTER_MODEL needs repointing.
        logger.error(f"OpenRouter {model} {r.status_code}: {r.text[:160]} — trying next provider.")
        return None
    try:
        choices = r.json().get("choices") or []
        if not choices:
            logger.error(f"OpenRouter returned no choices: {r.text[:200]}")
            return None
        return choices[0].get("message", {}).get("content") or None
    except Exception as e:
        logger.error(f"OpenRouter parse error: {e}")
        return None


def _call_groq(snapshot_json: str, system: str = None, user_prefix: str = "MARKET SNAPSHOT:\n",
               model: str = None) -> Optional[str]:
    """Groq (OpenAI-compatible) chat call. Returns raw text, or None.

    FIRST rung of every chain: fastest and cheapest, so routine ticks are absorbed
    here before any metered or subscription-backed provider is reached.
    """
    if not _has_groq_key():
        return None
    model = (model or settings.groq_model or "openai/gpt-oss-120b").strip()
    url = settings.groq_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": settings.groq_max_output_tokens,
        # Same contract as the Gemini rung: JSON out, so the plan parses cleanly.
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system or _SYSTEM},
            {"role": "user", "content": user_prefix + snapshot_json},
        ],
    }
    try:
        r = httpx.post(url, json=body, timeout=settings.ai_timeout_sec,
                       headers={"Authorization": f"Bearer {settings.groq_api_key.strip()}"})
    except Exception as e:
        logger.error(f"Groq request error: {e} — trying next provider.")
        return None
    if r.status_code != 200:
        # A decommissioned model 404s here — that is how this provider silently died
        # before; name the model so the log says which one to replace.
        logger.error(f"Groq {model} {r.status_code}: {r.text[:160]} — trying next provider.")
        return None
    try:
        choices = r.json().get("choices") or []
        if not choices:
            logger.error(f"Groq returned no choices: {r.text[:200]}")
            return None
        return choices[0].get("message", {}).get("content") or None
    except Exception as e:
        logger.error(f"Groq parse error: {e}")
        return None


def _call_gemini(snapshot_json: str, system: str = None, user_prefix: str = "MARKET SNAPSHOT:\n",
                 model: str = None) -> Optional[str]:
    """Google Gemini generateContent call (preferred provider). Returns raw text, or None.
    Forces JSON output via responseMimeType so the plan parses cleanly.
    `model` overrides the default tier (used to run the brief on Pro)."""
    if not _has_gemini_key():
        return None
    key = settings.gemini_api_key.strip()
    model = (model or settings.gemini_model or "gemini-flash-latest").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    gen: dict = {
        "temperature": 0.2,
        # Thinking tokens are charged as output and consume this same budget, so the
        # cap is the answer allowance PLUS whatever reasoning we've allowed.
        "maxOutputTokens": settings.gemini_max_output_tokens + max(0, settings.gemini_thinking_budget),
        "responseMimeType": "application/json",
    }
    if settings.gemini_thinking_budget != 0:
        gen["thinkingConfig"] = {"thinkingBudget": settings.gemini_thinking_budget}
    body = {
        "systemInstruction": {"parts": [{"text": system or _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user_prefix + snapshot_json}]}],
        "generationConfig": gen,
    }
    try:
        r = httpx.post(url, params={"key": key}, json=body, timeout=settings.ai_timeout_sec)
    except Exception as e:
        logger.error(f"Gemini request error: {e} — trying next provider.")
        return None
    if r.status_code != 200:
        # 429 on a Pro model usually means the key has no paid quota for that tier —
        # name the model so the log says which tier fell through.
        logger.error(f"Gemini {model} {r.status_code}: {r.text[:160]} — trying next provider.")
        return None
    try:
        cands = r.json().get("candidates") or []
        if not cands:
            logger.error(f"Gemini returned no candidates: {r.text[:200]}")
            return None
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)
    except Exception as e:
        logger.error(f"Gemini parse error: {e}")
        return None


_api_client = None
_api_client_key = None


def _get_api_client():
    """Lazily build (and cache) the Anthropic client for the current key."""
    global _api_client, _api_client_key
    if not _has_api_key():
        return None
    key = settings.anthropic_api_key.strip()
    if _api_client is None or _api_client_key != key:
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic SDK not installed — falling back to Claude CLI.")
            return None
        _api_client = anthropic.Anthropic(api_key=key)
        _api_client_key = key
    return _api_client


def _call_api(snapshot_json: str, system: str = None, user_prefix: str = "MARKET SNAPSHOT:\n") -> Optional[str]:
    """Direct Anthropic Messages API call (preferred). Returns the raw text, or None."""
    client = _get_api_client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=settings.ai_model,
            max_tokens=1500,
            system=system or _SYSTEM,
            messages=[{"role": "user", "content": user_prefix + snapshot_json}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    except Exception as e:
        logger.error(f"Anthropic API error ({type(e).__name__}): {str(e)[:200]} — falling back to CLI.")
        return None


#: Anthropic credentials the CLI honours. Scrubbed before every subprocess so a
#: stray global export can never silently redirect (or un-redirect) a call.
_ANTHROPIC_ENV_VARS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")


#: (UTC date, calls made) — resets itself when the date rolls over.
_router_spend: list = [None, 0]
#: Same, for the personal-subscription CLI rung.
_cli_spend: list = [None, 0]


def _day_budget_left(counter: list, cap: int) -> bool:
    """Shared UTC-day budget check; rolls the counter over on a new date."""
    if cap <= 0:
        return True
    today = datetime.now(timezone.utc).date()
    if counter[0] != today:
        counter[0], counter[1] = today, 0
    return counter[1] < cap


def _cli_budget_left() -> bool:
    """False once today's subscription-CLI cap is reached."""
    return _day_budget_left(_cli_spend, settings.subscription_cli_daily_call_cap)


def _cli_spend_record() -> None:
    _cli_spend[1] += 1
    cap = settings.subscription_cli_daily_call_cap
    if cap > 0 and _cli_spend[1] == cap:
        logger.warning(f"Subscription CLI daily cap reached ({cap} calls) — not spending "
                       f"more of it today; falling back to the mechanical engine.")


def _router_budget_left() -> bool:
    """False once today's AgentRouter call cap is reached.

    Every router call costs ~$0.28. On a 30-second loop that is 2,880 potential
    calls/day, so an outage of the free rungs would drain the balance in minutes.
    This cap turns that cliff into a soft landing: the chain simply moves on.
    """
    return _day_budget_left(_router_spend, settings.agentrouter_daily_call_cap)


def _router_spend_record() -> None:
    _router_spend[1] += 1
    cap = settings.agentrouter_daily_call_cap
    if cap > 0 and _router_spend[1] == cap:
        logger.warning(f"AgentRouter daily cap reached ({cap} calls) — skipping that rung "
                       f"until UTC midnight to protect the balance.")


def agentrouter_available() -> bool:
    """True if an AgentRouter token is configured, the CLI exists, and budget remains."""
    k = (settings.agentrouter_api_key or "").strip()
    return (bool(k) and k != "paste-your-key-here"
            and find_claude_cli() is not None and _router_budget_left())


def _claude_env(via_router: bool) -> dict:
    """Build the environment for a `claude -p` subprocess.

    AgentRouter is an Anthropic-compatible reseller that authenticates only
    Claude-Code-style clients — the Python SDK is rejected outright with
    `unauthorized_client_error`, so the CLI is the ONLY way to reach it.
    Injecting the credentials per-subprocess (rather than globally) keeps them
    out of the user's interactive Claude Code sessions, which stay on the
    subscription and remain the last-resort fallback when the router sinks.
    """
    env = os.environ.copy()
    for var in _ANTHROPIC_ENV_VARS:
        env.pop(var, None)
    if via_router:
        key = settings.agentrouter_api_key.strip()
        env["ANTHROPIC_BASE_URL"] = settings.agentrouter_base_url
        env["ANTHROPIC_AUTH_TOKEN"] = key
        env["ANTHROPIC_API_KEY"] = key
    return env


def _call_claude(prompt: str, via_router: bool = False) -> Optional[str]:
    """Run the headless Claude Code CLI.

    via_router=True bills AgentRouter credit; False uses the local subscription.
    """
    cli = find_claude_cli()
    if not cli:
        logger.warning("Claude CLI not found — AI brain unavailable, falling back to mechanical engine.")
        return None
    if via_router and not agentrouter_available():
        return None
    if not via_router and not settings.ai_allow_subscription_cli:
        logger.warning("Subscription CLI is disabled (AI_ALLOW_SUBSCRIPTION_CLI=false) — "
                       "not spending it; falling back to the mechanical engine.")
        return None
    if not via_router and not _cli_budget_left():
        return None
    label = "agentrouter" if via_router else "cli"
    model = settings.agentrouter_model if via_router else settings.ai_model
    if via_router:
        _router_spend_record()
    else:
        _cli_spend_record()
    try:
        proc = subprocess.run(
            [cli, "-p", "--output-format", "json", "--model", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=settings.ai_timeout_sec,
            env=_claude_env(via_router),
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Claude CLI [{label}] timed out after {settings.ai_timeout_sec}s")
        return None
    except Exception as e:
        logger.error(f"Claude CLI [{label}] invocation error: {e}")
        return None
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "ignore")[:300]
        # stderr is often just a benign CLI warning (e.g. "connectors disabled")
        # unrelated to the real failure — the actual reason usually lands in the
        # JSON envelope on stdout instead (e.g. {"is_error":true,"result":"..."}),
        # which was previously discarded entirely on a non-zero exit.
        try:
            stdout_raw = json.loads(proc.stdout.decode("utf-8", "ignore"))
            if stdout_raw.get("is_error") and stdout_raw.get("result"):
                detail = f"{stdout_raw['result']}" + (f" (stderr: {detail})" if detail else "")
        except (json.JSONDecodeError, AttributeError):
            pass
        logger.error(f"Claude CLI [{label}] exit {proc.returncode}: {detail}")
        return None
    try:
        raw = json.loads(proc.stdout.decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        logger.error(f"Claude CLI [{label}] returned non-JSON envelope")
        return None
    if raw.get("is_error"):
        logger.error(f"Claude CLI [{label}] reported error: {raw.get('result')}")
        return None
    return raw.get("result")


#: Everything except the news brief (trading loop, analysis, ad-hoc prompts):
#: Gemini Flash is the first port of call — fast and cheap enough to run every
#: 5 minutes. Claude via AgentRouter picks up when Gemini errors or hits its
#: daily quota; the subscription CLI is the last resort if the router sinks too.
#: The personal Claude subscription (`cli`) is deliberately ABSENT from every chain
#: below: the bot runs only on keys the user supplied (OpenRouter, Groq, Gemini,
#: AgentRouter). `cli` remains implemented for manual/one-off use but nothing in
#: the trading path routes to it, and AI_ALLOW_SUBSCRIPTION_CLI defaults to false.
DEFAULT_PROVIDERS = ("openrouter", "groq", "gemini", "agentrouter")
#: News brief: Ox Alpha first, then Claude via AgentRouter for a second opinion.
BRIEF_PROVIDERS = ("openrouter", "agentrouter", "groq", "gemini-pro")
#: Claude-family only — AgentRouter carries it, never the subscription.
CLAUDE_PROVIDERS = ("agentrouter",)
#: Latency-critical rung used by the fast execution loop. Ox Alpha is excluded on
#: purpose — it is a reasoning model (~55s/call) and would defeat the whole point.
#: Groq answers in ~1-2s; Gemini Flash is the backstop.
FAST_PROVIDERS = ("groq", "gemini")


def complete_json(system: str, payload: dict, user_prefix: str = "INPUT:\n",
                  providers: tuple[str, ...] = DEFAULT_PROVIDERS) -> Optional[dict]:
    """Run an arbitrary JSON-out prompt through a chosen provider chain.

    `providers` lets a caller pin which models may answer — the brief passes
    CLAUDE_PROVIDERS so it never silently degrades to a different model.
    """
    body = json.dumps(payload, separators=(",", ":"))
    for via in providers:
        try:
            if via == "openrouter":
                raw = _call_openrouter(body, system=system, user_prefix=user_prefix)
            elif via == "groq":
                raw = _call_groq(body, system=system, user_prefix=user_prefix)
            elif via == "gemini":
                raw = _call_gemini(body, system=system, user_prefix=user_prefix)
            elif via == "gemini-pro":
                raw = _call_gemini(body, system=system, user_prefix=user_prefix,
                                   model=settings.gemini_pro_model)
            elif via == "anthropic":
                raw = _call_api(body, system=system, user_prefix=user_prefix)
            elif via == "agentrouter":
                raw = _call_claude(system + "\n\n" + user_prefix + body, via_router=True)
            elif via == "cli":
                raw = _call_claude(system + "\n\n" + user_prefix + body)
            else:
                logger.warning(f"unknown provider '{via}' — skipping")
                continue
        except Exception as e:  # noqa: BLE001 — try the next provider
            logger.error(f"{via} call failed: {type(e).__name__}")
            raw = None
        if raw:
            out = _extract_json(raw)
            if out is not None:
                out["_via"] = via
                return out
    return None


def analyze(snapshot: dict, providers: tuple[str, ...] = DEFAULT_PROVIDERS) -> Optional[dict]:
    """
    Ask the AI for a trade plan. Returns a sanitized dict, or None if unavailable.
    Guardrail math (R:R, sizing, clamps) is enforced by the caller.

    `providers` pins the chain: the deep loop uses DEFAULT_PROVIDERS (Ox Alpha
    first), the fast execution loop passes FAST_PROVIDERS so a ripening setup is
    confirmed in seconds rather than waiting on a reasoning model.
    """
    snap_json = json.dumps(snapshot, separators=(",", ":"))
    # Walk the given chain so ordering has ONE definition (see the constants).
    # The fast execution loop passes FAST_PROVIDERS to skip slow reasoning models.
    prompt = _SYSTEM + "\n\nMARKET SNAPSHOT:\n" + snap_json
    result, via = None, None
    for provider in providers:
        if provider == "openrouter":
            result = _call_openrouter(snap_json)
        elif provider == "groq":
            result = _call_groq(snap_json)
        elif provider == "gemini":
            result = _call_gemini(snap_json)
        elif provider == "gemini-pro":
            result = _call_gemini(snap_json, model=settings.gemini_pro_model)
        elif provider == "anthropic":
            result = _call_api(snap_json)
        elif provider == "agentrouter":
            result = _call_claude(prompt, via_router=True)
        elif provider == "cli":
            result = _call_claude(prompt)
        else:
            logger.warning(f"unknown provider '{provider}' — skipping")
            continue
        if result is not None:
            via = provider
            break
    if result is None:
        return None
    plan = _extract_json(result)
    if not plan:
        logger.error(f"Could not parse AI plan ({via}) from: {result[:200]}")
        return None
    out = sanitize(plan)
    if out is not None:
        out["via"] = via
    return out


def sanitize(plan: dict) -> Optional[dict]:
    """Shape/type validation only. Returns None if the plan is structurally unusable."""
    if not isinstance(plan, dict):
        return None
    action = str(plan.get("action", "HOLD")).upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    try:
        conf = float(plan.get("confidence"))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    out = {
        "action": action,
        "confidence": conf,
        "entry": _r(plan.get("entry")),
        "stop_loss": _r(plan.get("stop_loss")),
        "take_profits": [],
        "reasoning": str(plan.get("reasoning", ""))[:500],
        "invalidation": str(plan.get("invalidation", ""))[:300],
    }
    tps = plan.get("take_profits") or []
    for t in tps:
        if isinstance(t, dict):
            p, sz = _r(t.get("price")), t.get("size_pct")
        else:
            p, sz = _r(t), None
        if p is not None:
            try:
                sz = float(sz)
            except (TypeError, ValueError):
                sz = None
            out["take_profits"].append({"price": p, "size_pct": sz})
    return out
