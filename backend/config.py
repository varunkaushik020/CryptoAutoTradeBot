from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root .env (one level above /backend) so it loads no matter the CWD.
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    delta_api_key: str = ""
    delta_api_secret: str = ""
    delta_base_url: str = "https://cdn-ind.testnet.deltaex.org"

    mongo_uri: str = "mongodb://localhost:27017/forexbot"

    trading_symbol: str = "BTCUSD"   # default/primary symbol (chart default)
    # Chart + analysis candles use the index-derived MARK price, not the thin
    # last-traded price. On the demo/testnet the traded feed prints fake wicks to
    # stale levels; mark-price candles are smooth and match the real market.
    use_mark_candles: bool = True
    # Symbols the bot actively trades SIMULTANEOUSLY (each opens + manages its own).
    trade_symbols: str = "BTCUSD,ETHUSD"
    max_concurrent_positions: int = 2  # cap total open positions across all coins
    trade_quantity: int = 1
    check_interval_minutes: int = 5
    # Deep-analysis cadence. When > 0 this WINS over check_interval_minutes.
    # 120s matches how long a full 2-symbol Ox Alpha tick actually takes, so the
    # scheduler stops firing ticks that only get dropped as overlapping.
    check_interval_seconds: int = 120

    # --- Fast execution loop -------------------------------------------------- #
    # The deep loop decides WHAT to trade; this loop decides WHEN, so a setup that
    # ripens between deep ticks is not entered (or exited) up to 2 minutes late.
    # It runs only for symbols the deep tick ARMED, uses FAST_PROVIDERS (Groq /
    # Gemini — never Ox Alpha, never the subscription), and never analyses from
    # scratch: it re-checks the armed trigger against a live price.
    fast_check_enabled: bool = True
    fast_check_seconds: int = 15
    # Arm when the distance from price to the trigger is within this multiple of
    # the move the market is expected to make before the next deep tick (derived
    # from 5m ATR). 1.0 = "reachable at current volatility"; raise to arm sooner.
    fast_arm_atr_mult: float = 1.2
    # An armed watch expires after this many seconds if the deep loop never renews
    # it, so a stale trigger can never fire on old analysis.
    fast_arm_ttl_sec: int = 300
    # Minimum strategies that must agree (on the entry timeframe) to trade.
    min_signals: int = 2
    # Multi-timeframe: 1h sets bias, 15m is the DECISION timeframe (votes + entry),
    # 5m is analyzed for entry timing/confirmation only (never the decision TF).
    trend_timeframe: int = 60    # 1h — sets allowed direction (bias)
    entry_timeframe: int = 15    # 15m — the timeframe trades are decided on
    ltf_timeframe: int = 5       # 5m — lower timeframe analyzed for entry timing
    # Auto-start the trading bot when the backend boots (survives restarts).
    auto_start_bot: bool = True
    # Comma-separated strategy ids to vote on each trade. Add new ones here.
    # Options: EMA_CROSS, RSI, BREAKOUT, SUPERTREND_AI, TRENDLINE_NAV
    strategies: str = "EMA_CROSS,RSI,BREAKOUT,SUPERTREND_AI,TRENDLINE_NAV,FVG,IFVG,SMC,MACD"
    # Trendline Navigator swing term: Long / Medium / Short (Short = most responsive).
    trendline_term: str = "Medium"
    # Candles fetched per tick (needs enough history for long-swing indicators).
    candle_limit: int = 500

    # --- Risk / sizing (CAPITAL-BASED: each trade deploys a fixed % of capital as margin) ---
    leverage: int = 50                 # margin leverage (max; auto-lowered so SL stays inside liquidation)
    position_capital_pct: float = 50.0 # normal trade: use this % of balance as margin
    big_trade_capital_pct: float = 20.0# "big" trade (strong confluence): smaller margin -> fewer lots
    big_trade_min_agree: float = 0.7   # fraction of enabled strategies that must agree for a big trade
    liq_buffer_pct: float = 0.5        # SL must sit at least this % of price INSIDE the liquidation price
    risk_min_pct: float = 0.5          # (legacy risk-based knobs, kept for fallback symbols)
    risk_max_pct: float = 1.5
    margin_cap_pct: float = 0.5        # never use more than 50% of *available* balance as margin on ONE trade
    stop_loss_pct: float = 1.0         # fallback stop distance (%) if no SL candidate
    risk_reward: float = 2.0           # FLOOR reward:risk (never below 1:2)
    # --- Per-symbol POINT limits (ETH). Normal trades are tight; "big" trades widen SL/TP. ---
    eth_sl_min_pts: float = 5.0
    eth_sl_max_pts: float = 20.0       # normal ETH stop: at most 20 points
    eth_tp_min_pts: float = 50.0       # normal ETH target band: 50–60 points
    eth_tp_max_pts: float = 60.0
    eth_big_sl_max_pts: float = 60.0   # big ETH trade: wider stop
    eth_big_tp_min_pts: float = 120.0  # big ETH trade: larger target
    eth_big_tp_max_pts: float = 220.0
    # Circuit breaker: stop opening NEW trades once today's realized loss reaches this
    # % of account (existing positions keep their exchange SL/TP). 0 disables.
    daily_loss_limit_pct: float = 10.0
    # --- Stop-loss placement (combined ATR + SuperTrend + structure) ---
    atr_period: int = 14
    atr_k: float = 1.5                 # ATR-based stop = entry +/- k*ATR
    sl_lookback: int = 20              # bars for structure swing (entry TF)
    min_sl_pct: float = 0.3            # clamp stop distance to >= this % of price
    max_sl_pct: float = 1.5            # clamp stop distance to <= this % of price (no big stops)
    target_lookback: int = 30          # 1h bars used for the 1:2 feasibility check
    # --- Partial take-profits + breakeven ---
    # Never more than 2 TPs. Set max_tps=1 for a single target (tp_splits="1.0").
    max_tps: int = 2
    tp_splits: str = "0.5,0.5"         # close 50% at TP1, 50% at TP2 (2 TPs)
    move_be_after_tp: int = 1          # move SL to breakeven after TP{n} fills (1 = after TP1)
    # Only show FVG / IFVG zones on the chart whose height is >= this % of price.
    fvg_min_pct: float = 0.6
    # --- Auto-tune: weight each strategy's vote by its live performance ---
    autotune_enabled: bool = True
    autotune_min_trades: int = 8     # need this many attributed trades before weighting a strategy
    autotune_gain: float = 0.6       # how strongly expectancy (R) shifts the weight
    weight_min: float = 0.3          # a strategy's vote can shrink to this
    weight_max: float = 2.0          # ...or grow to this

    # --- Trading agents (book-derived) + meta-labeling learning ensemble ---------- #
    # See .claude/skills/trading-books. The agents are the PRIMARY decision-maker; the
    # LLM brain below runs only as a FALLBACK when the agents abstain. Each agent learns
    # its own reliability from live trade outcomes (Beta posterior in Mongo `agent_perf`).
    agents_enabled: bool = True
    # Enabled agents (comma-separated ids). Options:
    #   MOMENTUM, MEAN_REVERSION, SMC_AGENT, CONFLUENCE
    agents: str = "MOMENTUM,MEAN_REVERSION,SMC_AGENT,CONFLUENCE"
    # Meta-label take/skip threshold: the ensemble only trades when its predicted
    # win-probability >= this. Below it, it abstains → the LLM fallback gets a turn.
    agents_min_confidence: float = 0.55
    # Pooled closed-trade count (across the winning agents) before half-Kelly sizing kicks in.
    agents_min_trades: int = 12
    agents_agreement_bonus: float = 0.03   # + per extra agreeing agent (capped by clamp)
    agents_opposition_penalty: float = 0.10 # − scaled by opposing pooled mass / winning mass
    # Position-size multiplier band applied to the trade's capital %. bet-size-from-prob
    # (AFML) maps into [min, max]; half-Kelly can only pull it DOWN, never above max.
    agents_size_min_mult: float = 0.5
    agents_size_max_mult: float = 1.5
    # Raw agent confidence band (before learning). Every agent maps its 0..1 signal
    # strength into [min, max]; the learning layer then shifts it by reliability.
    agent_conf_min: float = 0.55
    agent_conf_max: float = 0.95
    # Learning hyperparameters (symbol-agnostic).
    agents_beta_prior: float = 1.0     # Beta(prior,prior) prior on each agent's win-rate (0.5 start)
    agents_r_history: int = 50         # rolling R-multiples kept per agent for the Kelly estimate
    # --- Mean-reversion agent (Chan Ch.2–5) ---
    agent_hurst_mr_max: float = 0.5    # only fade when Hurst H < this (mean-reverting regime)
    agent_mr_z_entry: float = 1.5      # |z-score| beyond which price is "stretched"
    agent_mr_min_lookback: int = 10    # z-score window clamps (half-life sets it in between)
    agent_mr_max_lookback: int = 60
    # --- Momentum agent (Chan Ch.6–7) ---
    agent_hurst_trend_min: float = 0.5 # only trend-follow when Hurst H >= this (or H unknown)
    agent_mom_lookback: int = 20       # N-bar return + Donchian channel window
    agent_mom_full_return: float = 0.02 # |N-bar return| that counts as full momentum strength
    agent_mr_min_bars: int = 40        # min candles before the mean-reversion agent will act

    # --- AI brain: an LLM analyzes the chart each tick and sets entry/SL/TP ---
    # Trading provider priority: Gemini (GEMINI_API_KEY) → Anthropic API
    # (ANTHROPIC_API_KEY) → local Claude Code CLI → mechanical engine.
    # The news brief is Claude-only (Anthropic API → CLI) — see news.get_brief().
    # Groq was removed deliberately; leftover GROQ_* env vars are ignored.
    # --- OpenRouter / Ox Alpha: FIRST rung for all AI work ------------------- #
    # oxalpha.site is only a tracker page; the model itself is served by OpenRouter
    # as `stealth/ox-alpha` — free ($0 in/out), 1M context, OpenAI-compatible.
    # It is a STEALTH model: the provider is anonymous, there is no SLA, and it can
    # be withdrawn without notice — hence it leads a chain rather than replacing it.
    openrouter_api_key: str = ""
    openrouter_model: str = "stealth/ox-alpha"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_max_output_tokens: int = 4096
    # Sent as OpenRouter attribution headers (optional, but good manners).
    openrouter_referer: str = "https://github.com/local/forex-bot"
    openrouter_title: str = "forex-bot"

    # --- Groq: SECOND rung. Fastest and cheapest, so it absorbs the routine
    # 5-minute ticks before anything metered or subscription-backed is touched.
    # NOTE: llama-3.3-70b-versatile was decommissioned by Groq (404) — that is why
    # this provider went dark. Use a model that is live on the account.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_max_output_tokens: int = 4096
    gemini_api_key: str = ""
    # Two-tier Gemini. FLASH runs the 5-minute trading loop (288 calls/day — Pro there
    # would be slow and expensive). PRO runs the news brief, which is manual and
    # low-frequency, so it's where deeper reasoning actually pays for itself.
    # Pro falls through to Flash automatically if it errors or is out of quota.
    # Versioned names (gemini-2.5-flash) 404 for newer keys — use the `-latest` aliases.
    gemini_model: str = "gemini-flash-latest"
    gemini_pro_model: str = "gemini-pro-latest"
    # Thinking tokens for 2.5-class models: 0 = off, -1 = dynamic, or a fixed budget.
    # ~8k is the middle ground — enough for multi-step reasoning without paying for
    # pro-tier deliberation on every 5-minute tick.
    gemini_thinking_budget: int = 8192
    # Thinking tokens are billed as output and count toward maxOutputTokens, so the
    # cap must leave room for the answer ON TOP of the budget.
    gemini_max_output_tokens: int = 4096
    anthropic_api_key: str = ""
    # --- AgentRouter (Claude reseller, https://agentrouter.org) --------------- #
    # Anthropic-compatible, but it authenticates ONLY Claude-Code-style clients:
    # the Python SDK is rejected with `unauthorized_client_error`, so this is
    # reachable exclusively through the CLI path. ai_brain injects these into the
    # `claude -p` subprocess, which keeps them out of interactive Claude Code
    # sessions — those stay on the subscription and act as the final fallback.
    agentrouter_api_key: str = ""
    agentrouter_base_url: str = "https://agentrouter.org"
    # This token can only reach claude-opus-5 and claude-opus-4-8.
    agentrouter_model: str = "claude-opus-4-8"
    # Spend guard. A headless CLI call costs ~$0.28, so on a 30-SECOND loop
    # (2,880 ticks/day) an outage of the free rungs would burn ~$800/day and empty
    # the balance in minutes. Once this many router calls have been made in a UTC
    # day the rung is skipped and the chain moves on. 0 disables the cap.
    agentrouter_daily_call_cap: int = 120
    # Hard switch for the LAST rung. The Docker override mounts ~/.claude into the
    # container, so the `cli` provider spends the personal Claude subscription.
    # Set false to make the bot fail over to the mechanical engine instead of ever
    # touching it — the chain then ends at AgentRouter.
    # OFF by default: the trading bot runs only on user-supplied keys. No chain in
    # ai_brain routes to `cli`, and this is the belt that keeps it that way even if
    # one is added back by accident.
    ai_allow_subscription_cli: bool = False
    # Second belt on the same rung. Ox Alpha's daily ceiling is undocumented, so if
    # it 429s mid-day a 30s loop (5,760 calls/day across 2 symbols) could pour
    # thousands of calls into the personal subscription. Stop at this many per UTC
    # day and fall to the mechanical engine instead. 0 disables the cap.
    subscription_cli_daily_call_cap: int = 200
    ai_enabled: bool = True
    ai_model: str = "claude-sonnet-5"   # claude-sonnet-5 / claude-sonnet-5-6 / claude-haiku-5-20251001
    # decide  = AI picks direction + SL + TP (guardrails enforce risk); this is the default
    # refine  = strategy votes decide direction, AI only sets smarter SL/TP
    # advisory = AI analysis is logged/shown but the mechanical engine still trades
    ai_mode: str = "decide"
    ai_min_confidence: float = 0.55     # below this the AI's trade is skipped (HOLD)
    ai_timeout_sec: int = 150           # max seconds to wait for a Claude response
    # LLM fallback cooldown (seconds) PER SYMBOL. The agents decide every tick; the slow
    # LLM fallback is consulted at most this often. Critical for sub-minute cadence: a
    # single LLM call can take 15–150s, so calling it every 30s tick would overlap ticks
    # and hammer rate-limited providers (the 429 storm). 0 = no cooldown (call every tick).
    ai_min_interval_sec: int = 300
    # Leave blank to auto-discover the Claude Code binary; set to override.
    claude_cli_path: str = ""
    ai_respect_trend_filter: bool = True  # still block trades that fight the 1h trend

    # --- Market news + economic calendar ------------------------------------ #
    # ForexFactory has no public news API, so headlines are aggregated from forex/
    # crypto RSS feeds; the "forecast" table uses FF's official weekly calendar JSON.
    news_enabled: bool = True
    news_calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    # Comma-separated `Name|url` RSS feeds (name optional). Fetched concurrently; any
    # that fail are skipped. NOTE: The Block (theblock.co) is deliberately absent — it
    # answers 403 to all server-side requests regardless of user-agent.
    news_feeds: str = (
        "ForexLive|https://www.forexlive.com/feed/news,"
        "FXStreet|https://www.fxstreet.com/rss/news,"
        "Investing|https://www.investing.com/rss/news_1.rss,"
        "Cointelegraph|https://cointelegraph.com/rss,"
        "CoinDesk|https://www.coindesk.com/arc/outboundfeeds/rss/,"
        "The Defiant|https://thedefiant.io/api/feed,"
        "Decrypt|https://decrypt.co/feed,"
        "Yahoo Finance|https://finance.yahoo.com/news/rssindex,"
        "CryptoSlate|https://cryptoslate.com/feed/,"
        "Bitcoin.com|https://news.bitcoin.com/feed/"
    )
    # Currencies the bot *reacts* to: the AI snapshot and the trading blackout only
    # consider events for these (always keep USD — it drives DXY & crypto). The
    # dashboard calendar shows the full ForexFactory week regardless of this list.
    news_currencies: str = "USD,EUR,GBP,JPY,CNY"
    news_refresh_sec: int = 180      # cache TTL for headline fetches
    # Calendar changes weekly, and its host rate-limits frequent polling — cache it long.
    news_calendar_refresh_sec: int = 900   # 15 min
    # Public Telegram channels, comma-separated `Name|handle` (name optional).
    # Read via each channel's public web preview (t.me/s/<handle>) — no bot token and
    # no user session. A Telegram BOT cannot read channels it doesn't administer, so
    # the preview is the only credential-free way to follow third-party channels.
    telegram_channels: str = "LMWM News|lmwmnews"
    telegram_max_posts: int = 60     # cap the merged channel timeline
    news_max_stories: int = 90       # cap the merged headline list (10 feeds now)
    news_brief_ttl_sec: int = 900    # cached AI brief lifetime (manual refresh overrides)
    news_ai_context: bool = True     # inject news + upcoming events into the AI snapshot
    news_lookahead_hours: float = 24.0  # how far ahead a High-impact event counts as "upcoming"
    # Safety blackout: skip opening NEW trades within this many minutes (before OR after)
    # of a High-impact event for a relevant currency. 0 disables. Open positions keep SL/TP.
    news_blackout_min: int = 15

    # Manual close guard: refuse (pending confirmation) a market close whose real fill
    # is further than this from the mark price. The testnet book is often one-sided,
    # so an unguarded market close can turn a mark-profit into a real loss.
    close_max_slippage_pct: float = 0.5
    # Liquidity gates checked before EVERY entry. Never enter a market you cannot exit:
    # a wide book eats the edge on the way in, and an empty one traps the position.
    max_entry_spread_pct: float = 0.15    # quoted bid/ask spread vs mark
    max_exit_slippage_pct: float = 0.40   # cost of closing the intended size
    # Master switch + mode for the liquidity gate. "shadow" runs the check and logs
    # what it WOULD have done without ever blocking a trade — use this first to see
    # real block-rate data (ETH's testnet book is sometimes empty) before "enforce".
    liquidity_gate_enabled: bool = True
    liquidity_gate_mode: str = "shadow"   # "off" | "shadow" | "enforce"

    # Pre-trade expectancy/profitability gate: require the strategies backing the
    # proposed action to have REAL backtested edge on this symbol (from bot/edge.py)
    # before committing capital — not just "enough strategies agree".
    expectancy_gate_enabled: bool = True
    # The original +0.05 (demand a margin ABOVE breakeven) blocked ~100% of
    # BTCUSD/ETHUSD entries for 24h+ live — the persistent blocked case measured
    # -0.056R, which is only mildly negative, not a clearly-bad setup. -0.1 blocks
    # setups with a real demonstrated negative edge while letting near-breakeven
    # ones (which is most real setups here; PFs cluster ~0.96-1.19) through.
    expectancy_gate_min_R: float = -0.1        # min blended backtested expectancy (R) required
    expectancy_gate_min_strategy_n: int = 15   # backtested trades needed before a strategy's edge counts
    expectancy_gate_fail_open: bool = True     # allow the trade when there's no qualifying evidence yet
    # Train the strategy tuner on mark→mark P/L (the DECISION) rather than on fills
    # (the VENUE). Set false only on a venue whose fills you trust completely.
    autotune_use_mark_pnl: bool = True

    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # --- New indicators: Bollinger/Keltner squeeze, ADX, VWAP -------------------- #
    bb_period: int = 20
    bb_mult: float = 2.0
    kc_period: int = 20
    kc_atr_mult: float = 1.5
    kc_atr_len: int = 10
    adx_period: int = 14
    vwap_enabled: bool = True

    # ADX trend-strength GATE (global filter, not a vote): ranging markets (low ADX)
    # get no trend-following edge, so block entries there. Off by default until
    # backtest-validated per symbol (see GET /bot/backtest COMBINED_ADX_GATED).
    adx_gate_enabled: bool = False
    adx_min_trend: float = 20.0   # standard Wilder no-trend threshold

    # Divergence swing detection (RSI vs price at swing points).
    divergence_swing_left: int = 2
    divergence_swing_right: int = 2

    # --- Crypto-native strategies, shadow-mode validated (never influence real
    # trades until proven positive-expectancy live — see GET /bot/performance/shadow) --- #
    shadow_strategies: str = "FUNDING_BIAS,ORDERBOOK_IMBALANCE"
    funding_history_window_days: int = 30
    funding_min_history_samples: int = 20
    funding_extreme_percentile: float = 0.90   # >=90th / <=10th percentile = "extreme"
    ob_imbalance_levels: int = 10
    ob_imbalance_threshold: float = 0.35       # |bid-ask skew| beyond this casts a vote

    # --- Portfolio-level risk: correlation-aware sizing + volatility-adjusted sizing --- #
    correlation_check_enabled: bool = True
    correlation_lookback_bars: int = 200
    correlation_timeframe_min: int = 60
    correlation_high_threshold: float = 0.7
    correlation_dampen_factor: float = 0.5     # cap_pct multiplier when highly correlated & same direction

    vol_sizing_enabled: bool = False           # off by default — changes real sizing math
    vol_ref_atr_pct: float = 0.5               # "normal" ATR% of price this sizing is calibrated to
    vol_scalar_min: float = 0.4
    vol_scalar_max: float = 1.5

    model_config = SettingsConfigDict(env_file=str(_ROOT_ENV), extra="ignore")

settings = Settings()
