# -*- coding: utf-8 -*-
"""Build a STATIC, client-side Phase 2 dashboard page for GitHub Pages.

GitHub Pages cannot run the Python Dash server, so this renders the same four
Phase 2 views (uplift ranking, policy comparison, profit simulation, SHAP + CQR)
as a self-contained HTML page using Plotly.js with the REAL result data embedded
inline (read from phase2/results/*.csv). Matches docs/style.css.

Output: docs/phase2.html   (link: <pages-url>/phase2.html)
Run:    python dashboard/build_docs_phase2.py
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P2R = ROOT / "phase2" / "results"
OUT = ROOT / "docs" / "phase2.html"

# ---- read REAL data ----
um = pd.read_csv(P2R / "x5_uplift_models.csv")
pol = pd.read_csv(P2R / "x5_policy_comparison.csv")
pol["K"] = pol["K"].str.rstrip("%").astype(int)
ov = pd.read_csv(P2R / "x5_policy_overlap.csv")


def shap_top(ctx):
    d = pd.read_csv(P2R / f"{ctx}_shap_stage2.csv").nlargest(10, "mean_abs_shap_norm")
    return {"feature": d["feature"].tolist(), "value": [round(v, 4) for v in d["mean_abs_shap_norm"]]}


def cqr(ctx):
    d = pd.read_csv(P2R / f"{ctx}_cqr_coverage.csv")
    return [{"nominal": float(r.nominal_coverage), "emp": float(r.empirical_coverage),
             "width": float(r.mean_interval_width)} for _, r in d.iterrows()]


# KPI numbers (real)
best = um.sort_values("Qini_AUC", ascending=False).iloc[0]
resp_q = float(um.set_index("model").loc["ResponseModel", "Qini_AUC"])
ov10 = ov[(ov.K == "10%") & ov.policy_A.str.contains(r"Value\(") & ov.policy_B.str.contains("Uplift")]
overlap = float(ov10["overlap"].iloc[0]) if len(ov10) else float("nan")
va = pol[pol.policy == "Value-adjusted(uplift x value)"]
va_best = va.loc[va["profit"].idxmax()]

POLICY_COLORS = {
    "Random": "#c0392b", "RFM-only": "#e67e22", "Value-only(CLV proxy)": "#f1c40f",
    "Uplift-only(S-Learner)": "#27ae60", "Value-adjusted(uplift x value)": "#2f7ed8"}
MODEL_COLORS = {"S-Learner": "#2f7ed8", "X-Learner": "#e05a1f", "T-Learner": "#27ae60",
                "ClassTransform": "#8e44ad", "Random": "#e0a458", "ResponseModel": "#c0392b"}

DATA = {
    "uplift": {m: [float(um.set_index("model").loc[m, f"uplift@{k}"]) for k in (10, 20, 30)]
               for m in um["model"]},
    "modelColors": MODEL_COLORS,
    "policies": list(POLICY_COLORS.keys()),
    "policyColors": POLICY_COLORS,
    "K": [10, 20, 30],
    "incrValue": {p: [float(pol[(pol.policy == p) & (pol.K == k)]["incremental_revenue"].iloc[0]) / 1e6
                      for k in (10, 20, 30)] for p in POLICY_COLORS},
    "profit": {p: [float(pol[(pol.policy == p) & (pol.K == k)]["profit"].iloc[0]) / 1e6
                   for k in (10, 20, 30)] for p in POLICY_COLORS},
    "shap": {"online_retail": shap_top("online_retail"), "dunnhumby": shap_top("dunnhumby")},
    "cqr": {"online_retail": cqr("online_retail"), "dunnhumby": cqr("dunnhumby")},
}
KPI = {"best": best["model"], "bestQ": round(float(best["Qini_AUC"]), 4),
       "resp": round(resp_q, 4), "overlap": round(overlap, 3),
       "vaProfit": round(float(va_best["profit"]) / 1e6, 2), "vaK": f"{int(va_best['K'])}%"}

HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Phase 2 — Uplift-Aware Targeting Dashboard | Group 11</title>
<meta name="description" content="Phase 2 live dashboard: CLV ranking, uplift learners, value-adjusted targeting policy, SHAP and conformal uncertainty across three retail datasets (Online Retail II, Dunnhumby, X5 RetailHero)."/>
<link rel="stylesheet" href="style.css"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  .p2grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:820px){.p2grid{grid-template-columns:1fr}}
  .p2card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 14px 6px;box-shadow:0 1px 3px rgba(20,35,59,.05)}
  .p2card h3{margin:.1rem 0 .2rem;color:var(--navy2);font-size:1.02rem}
  .p2card .cap{color:var(--muted);font-size:.82rem;margin:0 0 6px}
  .kpirow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 26px}
  @media(max-width:820px){.kpirow{grid-template-columns:repeat(2,1fr)}}
  .k2{background:linear-gradient(180deg,#1f3a5f,#14233b);border-radius:14px;padding:16px 16px 14px;color:#fff}
  .k2 .v{font-size:1.8rem;font-weight:800}
  .k2 .v.blue{color:#7fb3ee}.k2 .v.orange{color:#f08a3c}.k2 .v.green{color:#7ee0a3}.k2 .v.red{color:#f39b8f}
  .k2 .l{font-size:.68rem;letter-spacing:1px;text-transform:uppercase;color:#9fb0c6;margin-top:6px}
  .k2 .s{font-size:.74rem;color:#c7d3e2;margin-top:3px}
  .ctl{margin:6px 0 16px}.ctl select{padding:8px 12px;border-radius:8px;border:1px solid var(--line);font-size:.95rem}
  .cqr{font-size:.82rem;color:var(--muted);padding:2px 6px 8px}
  .plot{width:100%;height:340px}
</style>
</head><body>

<nav class="nav"><div class="container">
  <div class="brand">CLV<span>·</span>VIP Targeting</div>
  <a class="link" href="index.html">&larr; Main site</a>
  <a class="link" href="index.html#live">Phase 1 Dashboard</a>
  <a class="link" href="#views">Phase 2 Views</a>
</div></nav>

<header class="hero"><div class="container">
  <div class="eyebrow">Phase 2 &middot; Live Dashboard</div>
  <h1>Uplift-Aware Targeting &mdash; Decision Report</h1>
  <p class="sub">CLV ranks <b>who is valuable</b>, uplift estimates <b>who is persuadable</b>,
  and value-adjusted uplift targets <b>who is worth targeting</b>. All numbers are the real
  held-out results from the X5&nbsp;RetailHero RCT and the walk-forward CLV contexts.</p>
  <div class="divider"></div>
  <div class="kpirow">
    <div class="k2"><div class="v blue" id="k-best"></div><div class="l">Best uplift model</div><div class="s" id="k-best-s"></div></div>
    <div class="k2"><div class="v red" id="k-resp"></div><div class="l">Response model (Qini)</div><div class="s">below random &rarr; value &ne; persuadability</div></div>
    <div class="k2"><div class="v orange" id="k-ov"></div><div class="l">Value vs Uplift overlap</div><div class="s">top-10% &mdash; nearly disjoint</div></div>
    <div class="k2"><div class="v green" id="k-va"></div><div class="l">Value-adjusted profit</div><div class="s" id="k-va-s"></div></div>
  </div>
</div></header>

<section id="views"><div class="container">
  <div class="tag">Interactive &middot; Plotly</div>
  <h2 class="h">Four decision views</h2>
  <p class="lead">The same four views as the Python Dash app (route <code>/uplift</code>), rendered
  client-side so the link works directly on GitHub Pages.</p>

  <div class="ctl"><label>Explainability context (View 4):&nbsp;</label>
    <select id="ctx"><option value="online_retail">Online Retail II (e-commerce)</option>
    <option value="dunnhumby">Dunnhumby (grocery)</option></select></div>

  <div class="p2grid">
    <div class="p2card"><h3>1 &middot; Customer ranking (CLV + uplift)</h3>
      <p class="cap">uplift@K by learner &mdash; response model ranks below random</p>
      <div id="v1" class="plot"></div></div>
    <div class="p2card"><h3>2 &middot; Policy comparison</h3>
      <p class="cap">incremental revenue by policy (million rubles)</p>
      <div id="v2" class="plot"></div></div>
    <div class="p2card"><h3>3 &middot; Profit simulation</h3>
      <p class="cap">held-out profit by policy (million rubles)</p>
      <div id="v3" class="plot"></div></div>
    <div class="p2card"><h3>4 &middot; Explainability &amp; uncertainty</h3>
      <p class="cap">Stage-2 SHAP drivers &mdash; who, why, and with what risk</p>
      <div id="v4" class="plot"></div><div class="cqr" id="cqr"></div></div>
  </div>
  <p class="lead" style="margin-top:22px">Source: <code>phase2/results/*.csv</code> in the repository.
  The interactive Python version (with the K / margin / cost sliders) runs locally via
  <code>python dashboard/app.py</code> &rarr; <code>http://127.0.0.1:8050/uplift</code>.</p>
</div></section>

<footer style="background:#14233b;color:#9fb0c6;padding:26px 0;text-align:center;font-size:.85rem">
  <div class="container">ADY201m &middot; Group 11 &middot; Phase 2: Uplift-aware targeting across Online Retail II, Dunnhumby &amp; X5 RetailHero</div>
</footer>

<script>
const D = __DATA__;
const K = __KPI__;
const L = {plot_bgcolor:'#fff', paper_bgcolor:'#fff', font:{family:'Segoe UI,Roboto,Arial', size:12, color:'#1b2733'},
  margin:{l:52,r:14,t:10,b:70}, legend:{orientation:'h', y:-0.25, font:{size:10}}, xaxis:{gridcolor:'#eef2f7'}, yaxis:{gridcolor:'#eef2f7'}};
const CFG = {displaylogo:false, responsive:true};

// KPI
document.getElementById('k-best').textContent = K.best.split('-')[0];
document.getElementById('k-best-s').textContent = 'Qini ' + K.bestQ.toFixed(4) + ' (held-out RCT)';
document.getElementById('k-resp').textContent = K.resp.toFixed(4);
document.getElementById('k-ov').textContent = K.overlap.toFixed(3);
document.getElementById('k-va').textContent = K.vaProfit.toFixed(2) + ' M';
document.getElementById('k-va-s').textContent = 'best at K=' + K.vaK + ' (rubles)';

// View 1 — uplift@K by model
Plotly.newPlot('v1', Object.keys(D.uplift).map(m => ({
  x:['10%','20%','30%'], y:D.uplift[m], name:m, mode:'lines+markers',
  line:{color:D.modelColors[m]}, marker:{color:D.modelColors[m]}})),
  Object.assign({}, L, {yaxis:{title:'uplift@K (incremental)', gridcolor:'#eef2f7'}, xaxis:{title:'Top-K% targeted', gridcolor:'#eef2f7'}}), CFG);

function policyTraces(key){ return D.policies.map(p => ({
  x:D.K, y:D[key][p], name:p, mode:'lines+markers',
  line:{color:D.policyColors[p]}, marker:{color:D.policyColors[p]}})); }

// View 2 — incremental revenue
Plotly.newPlot('v2', policyTraces('incrValue'),
  Object.assign({}, L, {yaxis:{title:'Incremental revenue (M)', gridcolor:'#eef2f7'}, xaxis:{title:'Top-K% targeted', tickvals:[10,20,30], gridcolor:'#eef2f7'}}), CFG);

// View 3 — profit (with zero line)
Plotly.newPlot('v3', policyTraces('profit'),
  Object.assign({}, L, {yaxis:{title:'Held-out profit (M)', gridcolor:'#eef2f7', zeroline:true, zerolinecolor:'#aaa'},
  xaxis:{title:'Top-K% targeted', tickvals:[10,20,30], gridcolor:'#eef2f7'},
  shapes:[{type:'line', x0:10, x1:30, y0:0, y1:0, line:{color:'#888', dash:'dot'}}]}), CFG);

// View 4 — SHAP + CQR (context toggle)
function drawShap(ctx){
  const s = D.shap[ctx];
  Plotly.newPlot('v4', [{type:'bar', orientation:'h', x:s.value.slice().reverse(), y:s.feature.slice().reverse(),
    marker:{color:'#2f7ed8'}}], Object.assign({}, L, {margin:{l:150,r:14,t:10,b:44},
    xaxis:{title:'mean |SHAP| (normalised)', gridcolor:'#eef2f7'}, yaxis:{automargin:true}}), CFG);
  const c = D.cqr[ctx].map(r => 'CQR nominal '+Math.round(r.nominal*100)+'% &rarr; empirical '+(r.emp*100).toFixed(1)+'% (mean width '+Math.round(r.width).toLocaleString()+')').join(' &nbsp;|&nbsp; ');
  document.getElementById('cqr').innerHTML = '<b>Uncertainty:</b> '+c+' &mdash; coverage close to nominal, so each customer\\'s interval is trustworthy for risk-aware targeting.';
}
drawShap('online_retail');
document.getElementById('ctx').addEventListener('change', e => drawShap(e.target.value));
</script>
</body></html>
"""

html = HTML.replace("__DATA__", json.dumps(DATA)).replace("__KPI__", json.dumps(KPI))
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, "| bytes", len(html))
print("KPI:", KPI)
