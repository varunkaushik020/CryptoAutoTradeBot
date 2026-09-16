"use client";
import { useEffect, useState } from "react";
import { getTraining } from "../lib/api";
import { GraduationCap, ShieldCheck, Ban, Info, ChevronDown, ChevronUp, AlertTriangle } from "lucide-react";
import clsx from "clsx";

interface Trade {
  _id?: string;
  symbol: string;
  side: string;
  closed_at: string;
  entry_mark: number;
  exit_mark: number;
  strategy_pnl: number;
  execution_pnl: number;
  slippage_cost: number;
  strategy_r: number;
  execution_r: number;
}

const usd = (n: number) => `${n > 0 ? "+" : ""}$${n.toFixed(2)}`;
const when = (iso: string) =>
  iso ? new Date(iso).toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "—";

const PAGE = 8;      // rows appended each time the list is scrolled to the bottom
const PREVIEW = 3;   // rows shown while collapsed
// Past this, "no recent trades" stops being a plausible quiet market and starts
// looking like the outcome recorder has stopped writing.
const STALE_AFTER_DAYS = 3;

export default function TrainingMonitor() {
  const [d, setD] = useState<any>(null);
  const [open, setOpen] = useState(false);
  const [limit, setLimit] = useState(PREVIEW);   // collapsed: the last few trades

  useEffect(() => {
    const fetchData = async () => {
      try { setD((await getTraining()).data); } catch { /* keep last */ }
    };
    fetchData();
    const iv = setInterval(fetchData, 30_000);
    return () => clearInterval(iv);
  }, []);

  // Expanding starts a page; collapsing drops back to the preview rows.
  useEffect(() => setLimit(open ? PAGE : PREVIEW), [open]);

  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (!open) return;
    const el = e.currentTarget;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 60) {
      setLimit((n) => n + PAGE);          // capped by slice() below
    }
  };

  const cfg = d?.config;
  const s = d?.summary;
  const trades: Trade[] = d?.recent_trades ?? [];
  const shown = trades.slice(0, Math.min(limit, trades.length));
  const skipped: { reason: string; count: number }[] = d?.skipped ?? [];
  const maxSkip = Math.max(1, ...skipped.map((x) => x.count));

  return (
    <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-4 space-y-4">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <GraduationCap size={16} className="text-[#00C896]" />
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Training Monitor</h3>
        </div>
        {cfg && (
          <div className="flex items-center gap-2 text-[11px] flex-wrap">
            <span className="px-2 py-0.5 rounded bg-[#00C896]/10 text-[#00C896] border border-[#00C896]/30">
              learns from {cfg.trains_on === "mark" ? "mark P/L (decisions)" : "fills (venue)"}
            </span>
            <span className="px-2 py-0.5 rounded bg-[#0d1117] text-gray-400 border border-[#30363d]">
              {cfg.symbols?.join(", ")}
            </span>
            <span className="px-2 py-0.5 rounded bg-[#0d1117] text-gray-400 border border-[#30363d]">
              spread ≤ {cfg.max_entry_spread_pct}% · exit ≤ {cfg.max_exit_slippage_pct}%
            </span>
          </div>
        )}
      </div>

      {/* The figures below are only as fresh as the recorder that writes them. Age is
          shown so a stopped writer reads as "stale", not as a calm month. */}
      {s?.stale_days != null && s.stale_days > STALE_AFTER_DAYS && (
        <div className="flex items-start gap-2 text-[11px] bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2 text-amber-300">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          <span>
            Newest scored trade is <b>{s.stale_days} days old</b>. If positions have closed
            since then, they are not being recorded — the figures below describe an older
            period, not the present one.
          </span>
        </div>
      )}

      {(s?.unverified_excluded ?? 0) > 0 && (
        <div className="flex items-start gap-2 text-[11px] bg-[#0d1117] border border-[#30363d] rounded-lg px-3 py-2 text-gray-500">
          <AlertTriangle size={12} className="mt-0.5 shrink-0 text-gray-600" />
          <span>
            <b className="text-gray-400">{s.unverified_excluded}</b> closed trade(s) had no
            matching fills, so they could not be scored and are excluded from every figure
            here — and from training.
          </span>
        </div>
      )}

      {/* Decision quality vs venue cost — the whole point of the split */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: "Strategy P/L", value: s ? usd(s.strategy_pnl) : "—", hint: "were the decisions right?",
            color: (s?.strategy_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400" },
          { label: "Execution P/L", value: s ? usd(s.execution_pnl) : "—", hint: "what the venue paid",
            color: (s?.execution_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400" },
          { label: "Slippage cost", value: s ? usd(s.slippage_cost) : "—", hint: "lost to the order book",
            color: (s?.slippage_cost ?? 0) >= 0 ? "text-gray-300" : "text-amber-400" },
          { label: "Decision win rate", value: s ? `${s.strategy_win_rate}%` : "—",
            hint: s?.stale_days != null ? `${s.trades} trades · newest ${s.stale_days}d ago`
                                        : `${s?.trades ?? 0} closed trades`,
            color: "text-gray-200" },
        ].map((c) => (
          <div key={c.label} className="bg-[#0d1117] border border-[#30363d] rounded-lg p-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider">{c.label}</p>
            <p className={clsx("text-lg font-bold tabular-nums mt-0.5", c.color)}>{c.value}</p>
            <p className="text-[10px] text-gray-600 mt-0.5">{c.hint}</p>
          </div>
        ))}
      </div>

      <div className="flex items-start gap-2 text-[11px] text-gray-500 bg-[#0d1117] border border-[#30363d] rounded-lg px-3 py-2">
        <Info size={12} className="mt-0.5 shrink-0 text-gray-600" />
        <span>
          <b className="text-gray-400">Strategy P/L</b> is measured mark-price entry → mark-price exit, so it scores the
          bot&apos;s decision. <b className="text-gray-400">Execution P/L</b> is the real fills. The gap between them is
          what the order book cost you — it trains nothing, because a thin book would otherwise punish correct calls.
        </span>
      </div>

      {/* Why entries were skipped */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <ShieldCheck size={13} className="text-gray-500" />
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Trades not taken · last {d?.window_days ?? 30} days
          </h4>
          <span className="text-[11px] text-gray-600">{d?.skipped_total ?? 0} blocked by guard rails</span>
        </div>
        {skipped.length === 0 ? (
          <p className="text-xs text-gray-600">No entries blocked yet.</p>
        ) : (
          <div className="space-y-1.5">
            {skipped.map((x) => (
              <div key={x.reason} className="flex items-center gap-2 text-[11px]">
                <Ban size={11} className="text-gray-600 shrink-0" />
                <span className="text-gray-400 w-48 shrink-0">{x.reason}</span>
                <div className="flex-1 h-1.5 rounded-full bg-[#0d1117] overflow-hidden">
                  <div className="h-full rounded-full bg-amber-500/50" style={{ width: `${(x.count / maxSkip) * 100}%` }} />
                </div>
                <span className="text-gray-500 tabular-nums w-10 text-right">{x.count}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Per-trade: decision vs execution. Collapsed shows only the latest trade;
          expanded scrolls and pages more rows in as you reach the bottom. */}
      <div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center gap-2 mb-2 group"
          title={open ? "Collapse trade history" : "Expand trade history"}
        >
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider group-hover:text-gray-200 transition">
            Recent closed trades
          </h4>
          <span className="text-[11px] text-gray-600">{trades.length}</span>
          <span className="flex-1 h-px bg-[#30363d]" />
          <span className="flex items-center gap-1 text-[10px] text-gray-500 group-hover:text-gray-300 transition">
            {open ? "collapse" : "expand"}
            {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </span>
        </button>

        {trades.length === 0 ? (
          <p className="text-xs text-gray-600">
            No trades closed since decision-scoring was enabled — this fills in as the bot trades.
          </p>
        ) : (
          <div
            onScroll={onScroll}
            className={clsx("overflow-x-auto", open && "overflow-y-auto max-h-[320px]")}
          >
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[#161b22] z-10">
                <tr className="text-gray-500 border-b border-[#30363d]">
                  <th className="px-2 py-2 text-left">Closed</th>
                  <th className="px-2 py-2 text-left">Side</th>
                  <th className="px-2 py-2 text-right">Mark in → out</th>
                  <th className="px-2 py-2 text-right">Strategy</th>
                  <th className="px-2 py-2 text-right">Execution</th>
                  <th className="px-2 py-2 text-right">Slippage</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((t, i) => (
                  <tr key={t._id || i} className="border-b border-[#21262d]">
                    <td className="px-2 py-2 text-gray-400 whitespace-nowrap">{when(t.closed_at)}</td>
                    <td className={clsx("px-2 py-2 font-semibold", t.side === "BUY" ? "text-green-400" : "text-red-400")}>
                      {t.side === "BUY" ? "LONG" : "SHORT"}
                    </td>
                    <td className="px-2 py-2 text-right font-mono text-gray-400 whitespace-nowrap">
                      {t.entry_mark?.toLocaleString()} → {t.exit_mark?.toLocaleString()}
                    </td>
                    <td className={clsx("px-2 py-2 text-right font-mono font-semibold",
                      t.strategy_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                      {usd(t.strategy_pnl)} <span className="text-gray-600">({t.strategy_r}R)</span>
                    </td>
                    <td className={clsx("px-2 py-2 text-right font-mono",
                      t.execution_pnl >= 0 ? "text-green-400" : "text-red-400")}>
                      {usd(t.execution_pnl)}
                    </td>
                    <td className={clsx("px-2 py-2 text-right font-mono",
                      t.slippage_cost < 0 ? "text-amber-400" : "text-gray-500")}>
                      {usd(t.slippage_cost)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {open && shown.length < trades.length && (
              <p className="py-2 text-center text-[10px] text-gray-600">
                scroll for more · {shown.length} of {trades.length}
              </p>
            )}
            {!open && trades.length > PREVIEW && (
              <p className="py-1.5 text-center text-[10px] text-gray-600">
                {trades.length - PREVIEW} more — click the header to expand
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
