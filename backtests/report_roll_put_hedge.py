"""report_roll_put_hedge.py — per-cycle HTML report for run_roll_put_hedge.py (any symbol).

Reads <results>/{cycles,windows,paths}.csv written by ``backtest-roll-put-hedge`` (+ the
F1/F2 panel from ArcticDB for the full final-week spread path) and writes one
self-contained HTML file — Plotly inlined, light/dark aware.

    report-roll-put-hedge                                     # IREDA 5minute
    report-roll-put-hedge --symbol SBICARD --interval 15minute
    report-roll-put-hedge --results results/ireda_roll_put_hedge_otm --html plots/otm.html
    (or: python backtests/report_roll_put_hedge.py ...)

Defaults: --results results/{symbol}_roll_put_hedge, --html plots/{symbol}_roll_put_hedge.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import ALL_TRADING_HOLIDAYS
from data.storage.arctic_store import ArcticStore

BANDS = ["tight", "medium", "relaxed"]
COMP = ["pnl_spread", "pnl_gamma", "pnl_vega", "pnl_theta", "pnl_other"]


def _ts(s: pd.Series) -> list[str]:
    return [t.strftime("%Y-%m-%d %H:%M") for t in pd.to_datetime(s)]


def build_payload(res: Path, symbol: str, interval: str) -> dict:
    cyc = pd.read_csv(res / "cycles.csv", parse_dates=["entry_ts", "exit_ts"])
    win = pd.read_csv(res / "windows.csv", parse_dates=["entry_ts", "exit_ts"])
    paths = pd.read_csv(res / "paths.csv", parse_dates=["ts"])
    cyc["f1_expiry"] = cyc["f1_expiry"].astype(str)
    win["f1_expiry"] = win["f1_expiry"].astype(str)
    paths["f1_expiry"] = paths["f1_expiry"].astype(str)
    cyc["pnl_other"] = cyc["pnl_mark"] - cyc[COMP[:-1]].sum(axis=1)   # hedge slack + higher order
    cyc["premium"] = cyc["put_entry"] * cyc["qty"]
    cyc["hold_h"] = (cyc["exit_ts"] - cyc["entry_ts"]).dt.total_seconds() / 3600

    gross = cyc[cyc.costs == "gross"].set_index(["f1_expiry", "band"])
    net = cyc[cyc.costs == "net"].set_index(["f1_expiry", "band"])
    expiries = sorted(cyc["f1_expiry"].unique())

    # ── Band summary ──────────────────────────────────────────────────────────
    bands = []
    for b in BANDS:
        g = gross.xs(b, level="band")["engine_realized"]
        n = net.xs(b, level="band")
        prem = gross.xs(b, level="band")["premium"].sum()
        bands.append({
            "band": b, "band_delta": float(n["band_delta"].iloc[0]),
            "gross": g.sum(), "net": n["engine_net"].sum(),
            "commission": n["engine_commission"].sum(),
            "slippage": g.sum() - n["engine_net"].sum() - n["engine_commission"].sum(),
            "rehedges": int(n["rehedges"].sum()),
            "turnover": int(n["hedge_turnover_shares"].sum()),
            "net_mean": n["engine_net"].mean(), "net_std": n["engine_net"].std(),
            "net_wins": int((n["engine_net"] > 0).sum()), "gross_wins": int((g > 0).sum()),
            "net_worst": n["engine_net"].min(), "net_best": n["engine_net"].max(),
            "premium": prem, "net_on_premium": n["engine_net"].sum() / prem,
        })

    # ── Per-cycle rows ────────────────────────────────────────────────────────
    rows = []
    for e in expiries:
        m = gross.loc[(e, "medium")]
        w = win.set_index("f1_expiry").loc[e]
        r = {
            "f1": e, "f2": str(m["f2_expiry"]),
            "entry": m["entry_ts"].strftime("%m-%d %H:%M"),
            "exit": m["exit_ts"].strftime("%m-%d %H:%M"),
            "entry_delayed": m["entry_ts"] != w["entry_ts"],
            "oracle_entry": w["entry_ts"].strftime("%m-%d %H:%M"),
            "hold_h": m["hold_h"], "strike": m["strike"], "lot": int(m["lot_size"]),
            "F1_entry": m["F1_entry"], "F2_entry": m["F2_entry"],
            "moneyness": m["strike"] / m["F2_entry"],
            "s_entry": m["spread_entry"], "s_exit": m["spread_exit"],
            "ds": m["spread_exit"] - m["spread_entry"],
            "dF1": m["F1_exit"] - m["F1_entry"], "dF2": m["F2_exit"] - m["F2_entry"],
            "put_entry": m["put_entry"], "put_exit": m["put_exit"],
            "iv_entry": m["iv_entry"], "iv_exit": m["iv_exit"],
            "d_entry": m["delta_entry"], "d_exit": m["delta_exit"],
            "opt_pnl": (m["put_exit"] - m["put_entry"]) * m["qty"],
            "premium": m["premium"],
            **{f"attr_{c}": m[c] for c in COMP},
        }
        for b in BANDS:
            r[f"gross_{b}"] = gross.loc[(e, b), "engine_realized"]
            r[f"net_{b}"] = net.loc[(e, b), "engine_net"]
            r[f"cost_{b}"] = r[f"gross_{b}"] - r[f"net_{b}"]
            r[f"reh_{b}"] = int(gross.loc[(e, b), "rehedges"])
            r[f"hedge_pnl_{b}"] = r[f"gross_{b}"] - r["opt_pnl"]
        rows.append(r)

    # ── Paths: full final-week spread (panel) + in-trade P&L/hedge (strategy) ─
    store = ArcticStore()
    tz = "Asia/Kolkata"
    detail = {}
    for e in expiries:
        w = win.set_index("f1_expiry").loc[e]
        lo = pd.Timestamp(str(w["week_start"]), tz=tz)
        hi = pd.Timestamp(e, tz=tz) + pd.Timedelta(days=1)
        pnl = store.read_futures(symbol, interval, with_spot=False, date_range=(lo, hi),
                                 columns=["F1_close", "F2_close", "F1_expiry"])
        pnl = pnl[(pnl["F1_expiry"].astype(str).str[:10] == e)].dropna(subset=["F1_close", "F2_close"])
        d = {"week": {"ts": _ts(pnl.index.to_series()),
                      "spread": (pnl["F1_close"] - pnl["F2_close"]).round(3).tolist()},
             "entry": gross.loc[(e, "medium"), "entry_ts"].strftime("%Y-%m-%d %H:%M"),
             "exit": gross.loc[(e, "medium"), "exit_ts"].strftime("%Y-%m-%d %H:%M"),
             "bands": {}}
        for b in BANDS:
            p = paths[(paths.f1_expiry == e) & (paths.band == b)]
            d["bands"][b] = {"ts": _ts(p["ts"]), "cum_pnl": p["cum_pnl"].round(1).tolist(),
                             "hedge": p["hedge_shares"].astype(int).tolist(),
                             "target": (-p["delta"] * gross.loc[(e, b), "qty"]).round(0).tolist()}
        detail[e] = d

    tot_g = gross.xs("medium", level="band")
    return {
        "bands": bands, "rows": rows, "detail": detail, "expiries": expiries,
        "attr_total": {c: float(tot_g[c].sum()) for c in COMP},
        "gross_medium": float(tot_g["engine_realized"].sum()),
        "holidays": [str(h)[:10] for h in ALL_TRADING_HOLIDAYS],
    }


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SYMBOL__ Roll-Spread Put Hedge</title>
<style>
:root{color-scheme:light;
 --surface-0:#f6f5f2;--surface-1:#fcfcfb;--border:#e4e2dc;--grid:#ecebe7;
 --text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7974;
 --band-1:#2a78d6;--band-2:#eb6834;--band-3:#1baf7a;
 --c-1:#eda100;--c-2:#e87ba4;--c-3:#008300;--c-4:#4a3aa7;--c-5:#e34948;
 --pos:#008300;--neg:#c23b3a;--shade:rgba(42,120,214,.10);}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--border:#33332f;--grid:#2a2a27;
 --text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
 --band-1:#3987e5;--band-2:#d95926;--band-3:#199e70;
 --c-1:#c98500;--c-2:#d55181;--c-3:#008300;--c-4:#9085e9;--c-5:#e66767;
 --pos:#3fae3f;--neg:#e66767;--shade:rgba(57,135,229,.16);}}
:root[data-theme="dark"]{color-scheme:dark;
 --surface-0:#121211;--surface-1:#1a1a19;--border:#33332f;--grid:#2a2a27;
 --text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
 --band-1:#3987e5;--band-2:#d95926;--band-3:#199e70;
 --c-1:#c98500;--c-2:#d55181;--c-3:#008300;--c-4:#9085e9;--c-5:#e66767;
 --pos:#3fae3f;--neg:#e66767;--shade:rgba(57,135,229,.16);}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
 font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:17px;margin:36px 0 6px}
p.sub,.note{color:var(--text-secondary);margin:0 0 14px}
.note{font-size:13px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:16px;margin-top:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:16px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--text-secondary);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:24px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .s{color:var(--text-muted);font-size:12px}
.chart{width:100%;height:360px}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th{color:var(--text-secondary);font-weight:600;position:sticky;top:0;background:var(--surface-1)}
td:first-child,th:first-child{text-align:left}
tr.sel td{background:var(--shade)}
tbody tr{cursor:pointer}
.pos{color:var(--pos)} .neg{color:var(--neg)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
select{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--border);
 background:var(--surface-1);color:var(--text-primary)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px 16px;margin-top:12px;font-size:13px}
.kv div span{color:var(--text-secondary);display:block;font-size:12px}
ul.caveats{color:var(--text-secondary);padding-left:18px} ul.caveats li{margin:4px 0}
code{font-size:12px}
</style></head>
<body><main>
<h1>__SYMBOL__ roll-spread put hedge</h1>
<p class="sub">Long 1 lot F2 ATM put, delta-hedged with the F1 future, held in the final expiry week
from the F1&minus;F2 spread trough to the week's spread peak. __NCYC__ monthly cycles, __SPAN__, __INTERVAL__ bars.
<strong>Entry and exit use look-ahead by design</strong>, so treat this as an upper bound, not a tradeable result.</p>

<div class="tiles" id="tiles"></div>

<h2>Net P&amp;L per cycle by hedge band</h2>
<p class="note">Engine realised P&amp;L after engine-default costs. Hover for gross, costs and rehedge count.</p>
<div class="card"><div id="c-net" class="chart"></div></div>

<h2>Where the P&amp;L comes from</h2>
<p class="note">Gross P&amp;L of the medium band, split per bar using the previous bar's Greeks. Spread = |&Delta;|&times;qty&times;&Delta;(F1&minus;F2).
"Hedge slack &amp; other" = residual delta left inside the band plus higher-order terms. Totals across all cycles are in the table further down.</p>
<div class="card"><div id="c-attr" class="chart"></div></div>

<h2>Cycle detail</h2>
<div class="card">
 <div class="controls"><label for="sel">Cycle (F1 expiry)</label><select id="sel"></select>
 <span class="note" id="sel-note" style="margin:0"></span></div>
 <div class="kv" id="kv"></div>
 <div id="c-spread" class="chart" style="height:280px"></div>
 <div id="c-cum" class="chart" style="height:260px"></div>
 <div id="c-hedge" class="chart" style="height:240px"></div>
</div>

<h2>Band summary</h2>
<div class="card tbl"><table id="t-bands"></table></div>

<h2>All cycles</h2>
<p class="note">Click a row to open its detail above. Option and spread columns are band-independent; the P&amp;L columns are per band.</p>
<div class="card tbl"><table id="t-cycles"></table></div>

<h2>Gross P&amp;L attribution (medium band)</h2>
<div class="card tbl"><table id="t-attr"></table></div>

<h2>Method and caveats</h2>
<div class="card"><ul class="caveats">
<li><b>Look-ahead.</b> Final week = the last 5 trading days up to and including F1 expiry. Exit = the bar with the week's maximum F1&minus;F2 (wide-panel closes).
Entry = the spread minimum before that bar. Both are known only with hindsight.</li>
<li><b>Engine.</b> Run on <code>BacktestEngine</code> with <code>RollSpreadPutHedgeStrategy</code> (<code>options/strategies/roll_spread_put_hedge.py</code>).
Snapshots come from <code>LocalSnapshotBuilder(source="arctic")</code>. Quantities are in shares (<code>lot_size=1</code>).</li>
<li><b>Hedge bands.</b> Rehedge to delta-flat when |net delta| &gt; band &times; put shares: tight 0.02, medium 0.05, relaxed 0.10.
The hedge trades in whole shares, not futures lots. With 1 put lot (&Delta;&asymp;&minus;0.5) a real hedge would need a fraction of a futures lot, so this is a per-share idealisation.</li>
<li><b>Option marks.</b> The F2 chain is thin (a few strikes print per __INTERVAL__ bar). The held strike is marked with Black-76 at its last observed IV on the current F2 future and fractional DTE (sticky strike).
Entry and exit fill at that mark. The strategy's own P&amp;L matches the engine's realised P&amp;L to within &plusmn;&#8377;25 per cycle (the engine rounds fill prices to 2 dp).</li>
<li><b>Costs ("net").</b> Engine defaults: &#8377;20 per order, STT 0.0625% on sells (also applied to futures, where actual STT is 0.02%), 0.05% exchange charges, 18% GST, stamp duty, and 0.1% slippage on every fill.
Option bid/ask beyond that 0.1% is not modelled.</li>
<li><b>Strike choice.</b> "ATM" = the put strike nearest F2 among strikes with a valid IV in the chain at entry. On 2025-10-28 the nearest available strike was 145 against F2 152.8 (&Delta; &minus;0.29), so that cycle held a smaller, OTM put.</li>
<li><b>Entry slip.</b> On 2025-05-29 and 2026-03-30 the F2 chain had no priced put at the hindsight bar, so entry happened at the next bar that had one (shown in the table).</li>
</ul></div>
</main>
<script>__PLOTLY__</script>
<script>
const D = __DATA__;
const BANDS = ["tight","medium","relaxed"], BANDV = {tight:0.02,medium:0.05,relaxed:0.10};
const COMP = [["pnl_spread","Spread capture"],["pnl_gamma","Gamma"],["pnl_vega","Vega"],["pnl_theta","Theta"],["pnl_other","Hedge slack & other"]];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const inr = v => (v<0?"−":"")+"₹"+Math.abs(Math.round(v)).toLocaleString("en-IN");
const f = (v,d=2) => (v<0?"−":"")+Math.abs(v).toFixed(d);
const cls = v => v>0?"pos":(v<0?"neg":"");
const bandColor = b => css("--band-"+(BANDS.indexOf(b)+1));
let sel = D.expiries[D.expiries.length-1];

function baseLayout(extra={}){
  return Object.assign({
    paper_bgcolor:css("--surface-1"), plot_bgcolor:css("--surface-1"),
    font:{family:"system-ui,-apple-system,Segoe UI,Roboto,sans-serif",size:12,color:css("--text-secondary")},
    margin:{l:64,r:16,t:28,b:44}, hovermode:"closest",
    hoverlabel:{bgcolor:css("--surface-1"),bordercolor:css("--border"),font:{color:css("--text-primary")}},
    legend:{orientation:"h",x:0,y:1.12,font:{color:css("--text-primary")}},
    xaxis:{gridcolor:css("--grid"),linecolor:css("--border"),zeroline:false,tickfont:{color:css("--text-secondary")}},
    yaxis:{gridcolor:css("--grid"),linecolor:css("--border"),zerolinecolor:css("--text-muted"),tickfont:{color:css("--text-secondary")}},
  }, extra);
}
const CFG = {displayModeBar:false, responsive:true};

function tiles(){
  const best = D.bands.reduce((a,b)=>b.net>a.net?b:a);
  const spreadShare = D.attr_total.pnl_spread / D.gross_medium;
  const t = [
    ...D.bands.map(b=>({k:`Net · ${b.band} (${b.band_delta})`, v:inr(b.net), s:`${b.net_wins}/15 cycles positive · ${b.rehedges} rehedges`, band:b.band})),
    {k:"Gross · medium", v:inr(D.bands[1].gross), s:`${D.bands[1].gross_wins}/15 cycles positive before costs`},
    {k:"Spread share of gross", v:Math.round(spreadShare*100)+"%", s:`theta ${inr(D.attr_total.pnl_theta)} · gamma+vega ${inr(D.attr_total.pnl_gamma+D.attr_total.pnl_vega)}`},
    {k:"Net / premium paid", v:(best.net_on_premium*100).toFixed(0)+"%", s:`${best.band}: ${inr(best.net)} on ${inr(best.premium)} premium`},
  ];
  document.getElementById("tiles").innerHTML = t.map(x=>`<div class="tile"><div class="k">${x.band?`<span class="sw" style="background:${bandColor(x.band)}"></span>`:""}${x.k}</div><div class="v">${x.v}</div><div class="s">${x.s}</div></div>`).join("");
}

function chartNet(){
  const x = D.rows.map(r=>r.f1);
  const tr = BANDS.map(b=>({type:"bar", name:`${b} (${BANDV[b]})`, x, y:D.rows.map(r=>r["net_"+b]),
    marker:{color:bandColor(b), line:{color:css("--surface-1"),width:2}},
    customdata:D.rows.map(r=>[r["gross_"+b],r["cost_"+b],r["reh_"+b]]),
    hovertemplate:`<b>%{x}</b> · ${b}<br>net ₹%{y:,.0f}<br>gross ₹%{customdata[0]:,.0f}<br>costs ₹%{customdata[1]:,.0f}<br>rehedges %{customdata[2]}<extra></extra>`}));
  Plotly.react("c-net", tr, baseLayout({barmode:"group", bargap:0.25, bargroupgap:0.08, margin:{l:64,r:16,t:28,b:84},
    yaxis:Object.assign(baseLayout().yaxis,{title:{text:"Net P&L (₹)"},tickformat:",.0f"}),
    xaxis:Object.assign(baseLayout().xaxis,{type:"category",title:{text:"F1 expiry"}})}), CFG);
  document.getElementById("c-net").on("plotly_click", e=>{ select(e.points[0].x); });
}

function chartAttr(){
  const x = D.rows.map(r=>r.f1);
  const tr = COMP.map(([k,lab],i)=>({type:"bar", name:lab, x, y:D.rows.map(r=>r["attr_"+k]),
    marker:{color:css("--c-"+(i+1)), line:{color:css("--surface-1"),width:2}},
    hovertemplate:`<b>%{x}</b><br>${lab} ₹%{y:,.0f}<extra></extra>`}));
  tr.push({type:"scatter", mode:"markers", name:"Gross total", x, y:D.rows.map(r=>r.gross_medium),
    marker:{symbol:"diamond", size:9, color:css("--text-primary"), line:{color:css("--surface-1"),width:2}},
    hovertemplate:"<b>%{x}</b><br>Gross ₹%{y:,.0f}<extra></extra>"});
  Plotly.react("c-attr", tr, baseLayout({barmode:"relative", bargap:0.3, margin:{l:64,r:16,t:28,b:84},
    yaxis:Object.assign(baseLayout().yaxis,{title:{text:"₹ (medium band, gross)"},tickformat:",.0f"}),
    xaxis:Object.assign(baseLayout().xaxis,{type:"category",title:{text:"F1 expiry"}})}), CFG);
  document.getElementById("c-attr").on("plotly_click", e=>{ select(e.points[0].x); });
}

function chartDetail(){
  const d = D.detail[sel], r = D.rows.find(x=>x.f1===sel);
  document.getElementById("sel-note").textContent = r.entry_delayed ? `Entry slipped from the hindsight bar ${r.oracle_entry} to ${r.entry} (no priced F2 put at that bar).` : "";
  const kv = [["F2 put", `${r.strike} PE ${r.f2}`],["Lot", r.lot.toLocaleString("en-IN")],
    ["Hold", `${r.entry} → ${r.exit} (${r.hold_h.toFixed(1)} h)`],["Spread", `${f(r.s_entry)} → ${f(r.s_exit)} (Δ ${f(r.ds)})`],
    ["ΔF1 / ΔF2", `${f(r.dF1)} / ${f(r.dF2)}`],["Put", `${f(r.put_entry)} → ${f(r.put_exit)}`],
    ["IV", `${(r.iv_entry*100).toFixed(1)}% → ${(r.iv_exit*100).toFixed(1)}%`],["Delta", `${f(r.d_entry)} → ${f(r.d_exit)}`],
    ...BANDS.map(b=>[`Net · ${b}`, `<span class="${cls(r["net_"+b])}">${inr(r["net_"+b])}</span> (${r["reh_"+b]} rehedges)`])];
  document.getElementById("kv").innerHTML = kv.map(([k,v])=>`<div><span>${k}</span>${v}</div>`).join("");
  const shade = [{type:"rect", xref:"x", yref:"paper", x0:d.entry, x1:d.exit, y0:0, y1:1, fillcolor:css("--shade"), line:{width:0}, layer:"below"}];
  const xa = Object.assign(baseLayout().xaxis,{type:"date", rangebreaks:[{bounds:["sat","mon"]},{bounds:[15.5,9.25],pattern:"hour"}]});
  const i0 = d.week.ts.indexOf(d.entry), i1 = d.week.ts.indexOf(d.exit);
  Plotly.react("c-spread", [
    {type:"scatter", mode:"lines", name:"F1 − F2", x:gap(d.week.ts,d.week.spread)[0], y:gap(d.week.ts,d.week.spread)[1], line:{color:css("--band-1"),width:2}, hovertemplate:"%{x}<br>spread %{y:.2f}<extra></extra>"},
    {type:"scatter", mode:"markers+text", name:"entry / exit", x:[d.week.ts[i0]??d.entry, d.week.ts[i1]??d.exit], y:[d.week.spread[i0], d.week.spread[i1]],
     text:["trough","peak"], textposition:["bottom center","top center"], textfont:{color:css("--text-primary")},
     marker:{size:10, color:css("--text-primary"), line:{color:css("--surface-1"),width:2}}, hovertemplate:"%{text} %{x}<br>spread %{y:.2f}<extra></extra>"}],
    baseLayout({title:{text:"F1 − F2 spread over the final week (₹) · shaded = position held", font:{size:13,color:css("--text-primary")}, x:0, xanchor:"left", y:0.98, yanchor:"top"}, margin:{l:64,r:16,t:62,b:36}, legend:{orientation:"h",x:0,y:1.0,yanchor:"bottom",font:{color:css("--text-primary")}},
      shapes:shade, showlegend:false, xaxis:xa, yaxis:Object.assign(baseLayout().yaxis,{title:{text:"₹"}})}), CFG);
  Plotly.react("c-cum", BANDS.map(b=>({type:"scatter", mode:"lines", name:b, x:gap(d.bands[b].ts,d.bands[b].cum_pnl)[0], y:gap(d.bands[b].ts,d.bands[b].cum_pnl)[1],
      line:{color:bandColor(b),width:2}, hovertemplate:`${b} %{x}<br>₹%{y:,.0f}<extra></extra>`})),
    baseLayout({title:{text:"Cumulative gross P&L while held (₹, mark-to-model)", font:{size:13,color:css("--text-primary")}, x:0, xanchor:"left", y:0.98, yanchor:"top"}, margin:{l:64,r:16,t:62,b:36}, legend:{orientation:"h",x:0,y:1.0,yanchor:"bottom",font:{color:css("--text-primary")}},
      hovermode:"x unified", xaxis:Object.assign({},xa), yaxis:Object.assign(baseLayout().yaxis,{tickformat:",.0f"})}), CFG);
  const hz = BANDS.map(b=>({type:"scatter", mode:"lines", name:`hedge · ${b}`, x:gap(d.bands[b].ts,d.bands[b].hedge)[0], y:gap(d.bands[b].ts,d.bands[b].hedge)[1], line:{color:bandColor(b),width:2,shape:"hv"},
      hovertemplate:`${b} %{x}<br>%{y:,} F1 shares<extra></extra>`}));
  hz.unshift({type:"scatter", mode:"lines", name:"full hedge (−Δ × qty)", x:gap(d.bands.medium.ts,d.bands.medium.target)[0], y:gap(d.bands.medium.ts,d.bands.medium.target)[1],
      line:{color:css("--text-muted"),width:1.5,dash:"dot"}, hovertemplate:"%{x}<br>target %{y:,}<extra></extra>"});
  Plotly.react("c-hedge", hz, baseLayout({title:{text:"F1 hedge held (shares) vs full-delta target", font:{size:13,color:css("--text-primary")}, x:0, xanchor:"left", y:0.98, yanchor:"top"}, margin:{l:64,r:16,t:62,b:36}, legend:{orientation:"h",x:0,y:1.0,yanchor:"bottom",font:{color:css("--text-primary")}},
      hovermode:"x unified", xaxis:Object.assign({},xa), yaxis:Object.assign(baseLayout().yaxis,{tickformat:","})}), CFG);
  document.querySelectorAll("#t-cycles tbody tr").forEach(tr=>tr.classList.toggle("sel", tr.dataset.f1===sel));
}

function tables(){
  const sw = b=>`<span class="sw" style="background:${bandColor(b)}"></span>`;
  document.getElementById("t-bands").innerHTML = `<thead><tr><th>Band</th><th>|Δ| band</th><th>Gross</th><th>Commission</th><th>Slippage</th><th>Net</th><th>Net mean / cycle</th><th>Net std</th><th>Net wins</th><th>Worst</th><th>Best</th><th>Rehedges</th><th>Hedge turnover (sh)</th><th>Net / premium</th></tr></thead><tbody>`+
    D.bands.map(b=>`<tr><td>${sw(b.band)}${b.band}</td><td>${b.band_delta}</td><td>${inr(b.gross)}</td><td>${inr(b.commission)}</td><td>${inr(b.slippage)}</td><td class="${cls(b.net)}"><b>${inr(b.net)}</b></td><td>${inr(b.net_mean)}</td><td>${inr(b.net_std)}</td><td>${b.net_wins}/15</td><td class="${cls(b.net_worst)}">${inr(b.net_worst)}</td><td>${inr(b.net_best)}</td><td>${b.rehedges}</td><td>${b.turnover.toLocaleString("en-IN")}</td><td>${(b.net_on_premium*100).toFixed(0)}%</td></tr>`).join("")+"</tbody>";
  const H = ["F1 expiry","Entry","Exit","Hold h","Strike","F2 @entry","K/F2","Spread in→out","ΔF1","ΔF2","Put in→out","IV in→out","Δ in→out","Option P&L",
    ...BANDS.flatMap(b=>[`Gross ${b}`,`Net ${b}`,`Reh ${b}`])];
  document.getElementById("t-cycles").innerHTML = `<thead><tr>${H.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>`+
    D.rows.map(r=>`<tr data-f1="${r.f1}"><td>${r.f1}</td><td>${r.entry}${r.entry_delayed?"*":""}</td><td>${r.exit}</td><td>${r.hold_h.toFixed(1)}</td><td>${r.strike}</td><td>${f(r.F2_entry)}</td><td>${r.moneyness.toFixed(3)}</td>
      <td>${f(r.s_entry)} → ${f(r.s_exit)}</td><td class="${cls(r.dF1)}">${f(r.dF1)}</td><td class="${cls(r.dF2)}">${f(r.dF2)}</td><td>${f(r.put_entry)} → ${f(r.put_exit)}</td>
      <td>${(r.iv_entry*100).toFixed(1)} → ${(r.iv_exit*100).toFixed(1)}</td><td>${f(r.d_entry)} → ${f(r.d_exit)}</td><td class="${cls(r.opt_pnl)}">${inr(r.opt_pnl)}</td>
      ${BANDS.map(b=>`<td class="${cls(r["gross_"+b])}">${inr(r["gross_"+b])}</td><td class="${cls(r["net_"+b])}"><b>${inr(r["net_"+b])}</b></td><td>${r["reh_"+b]}</td>`).join("")}</tr>`).join("")+
    `</tbody><tfoot><tr><td colspan="13"><b>Total</b></td><td>${inr(D.rows.reduce((a,r)=>a+r.opt_pnl,0))}</td>${BANDS.map(b=>{const g=D.rows.reduce((a,r)=>a+r["gross_"+b],0),n=D.rows.reduce((a,r)=>a+r["net_"+b],0),h=D.rows.reduce((a,r)=>a+r["reh_"+b],0);return `<td><b>${inr(g)}</b></td><td class="${cls(n)}"><b>${inr(n)}</b></td><td>${h}</td>`}).join("")}</tr></tfoot>`;
  document.querySelectorAll("#t-cycles tbody tr").forEach(tr=>tr.addEventListener("click",()=>{select(tr.dataset.f1); document.getElementById("sel").scrollIntoView({behavior:"smooth",block:"center"});}));
  document.getElementById("t-attr").innerHTML = `<thead><tr><th>F1 expiry</th>${COMP.map(([k,l],i)=>`<th><span class="sw" style="background:${css("--c-"+(i+1))}"></span>${l}</th>`).join("")}<th>Gross (engine)</th></tr></thead><tbody>`+
    D.rows.map(r=>`<tr data-f1="${r.f1}"><td>${r.f1}</td>${COMP.map(([k])=>`<td class="${cls(r["attr_"+k])}">${inr(r["attr_"+k])}</td>`).join("")}<td><b>${inr(r.gross_medium)}</b></td></tr>`).join("")+
    `</tbody><tfoot><tr><td><b>Total</b></td>${COMP.map(([k])=>`<td><b>${inr(D.attr_total[k])}</b></td>`).join("")}<td><b>${inr(D.gross_medium)}</b></td></tr></tfoot>`;
}

function gap(ts, ...ys){   // insert a null point across a holiday so lines don't bridge it
  const X=[], Y=ys.map(()=>[]);
  for(let i=0;i<ts.length;i++){
    if(i>0){const a=ts[i-1].slice(0,10), b=ts[i].slice(0,10);
      if(D.holidays.some(h=>h>a && h<b)){X.push(a+" 15:30"); Y.forEach(y=>y.push(null));}}
    X.push(ts[i]); ys.forEach((y,k)=>Y[k].push(y[i]));
  }
  return [X, ...Y];
}
function select(e){ sel = e; document.getElementById("sel").value = e; chartDetail(); }
function renderAll(){ tiles(); tables(); chartNet(); chartAttr(); chartDetail(); }
document.getElementById("sel").innerHTML = D.expiries.map(e=>`<option value="${e}">${e}</option>`).join("");
document.getElementById("sel").value = sel;
document.getElementById("sel").addEventListener("change", e=>select(e.target.value));
renderAll();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
</script></body></html>
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="IREDA", help="Underlying (default IREDA).")
    ap.add_argument("--interval", default="5minute", help="Bar interval of the run (default 5minute).")
    ap.add_argument("--results", default=None,
                    help="Folder written by backtest-roll-put-hedge (default results/{symbol}_roll_put_hedge).")
    ap.add_argument("--html", default=None,
                    help="Output HTML (default plots/{symbol}_roll_put_hedge.html).")
    args = ap.parse_args(argv)
    symbol = args.symbol.upper()
    res = Path(args.results) if args.results else ROOT / "results" / f"{symbol.lower()}_roll_put_hedge"
    out = Path(args.html) if args.html else ROOT / "plots" / f"{symbol.lower()}_roll_put_hedge.html"

    payload = build_payload(res, symbol, args.interval)
    exp = payload["expiries"]
    span = (f"{pd.Timestamp(exp[0]):%b %Y} &ndash; {pd.Timestamp(exp[-1]):%b %Y}" if exp else "no cycles")
    html = (HTML.replace("__SYMBOL__", symbol)
                .replace("__NCYC__", str(len(exp)))
                .replace("__SPAN__", span)
                .replace("__INTERVAL__", args.interval)
                .replace("__PLOTLY__", get_plotlyjs())
                .replace("__DATA__", json.dumps(payload, default=float)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
