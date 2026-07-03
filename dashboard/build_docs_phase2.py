# -*- coding: utf-8 -*-
"""Build the STATIC + INTERACTIVE Phase 2 dashboard page for GitHub Pages.

Three datasets, three roles:
  * X5 RetailHero (treatment/control) -> interactive targeting SIMULATOR with client-side
    upload (raw uplift+CI, policy comparison, profit with live cost/margin sliders, overlap).
  * Online Retail II & Dunnhumby (CLV, no treatment) -> CROSS-CONTEXT CLV section driven by
    one dataset selector: walk-forward model benchmark, LOO feature ablation, Stage-2 SHAP,
    and conformal (CQR) coverage.

All numbers are the real Phase 2 results (phase2/results/*.csv). Output: docs/phase2.html
Run:  python dashboard/build_docs_phase2.py
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P2R = ROOT / "phase2" / "results"
OUT = ROOT / "docs" / "phase2.html"

CTX = {"online_retail": "Online Retail II (e-commerce)", "dunnhumby": "Dunnhumby (grocery)"}

# ---- walk-forward CLV benchmark (both contexts) ----
wf = pd.read_csv(P2R / "clv_walkforward_summary.csv")
wmap = {"Online Retail II (W1-W3)": "online_retail", "Dunnhumby (D1-D4)": "dunnhumby"}
WALK = {}
for raw, key in wmap.items():
    d = wf[(wf.context == raw) & (wf.model != "Mean")].sort_values("Norm_Gini_mean", ascending=False)
    WALK[key] = [{"model": m, "ng": round(float(ng), 3), "rc": round(float(rc), 1)}
                 for m, ng, rc in zip(d.model, d.Norm_Gini_mean, d.RevCapture_10_mean)]

# ---- LOO feature ablation (both contexts) ----
abl = pd.read_csv(P2R / "ablation_loo_summary.csv")
relabel = {"ALL-behavioural": "drop behavioural", "ALL-semantic": "drop semantic",
           "ALL-sequence": "drop sequence", "RFM-only": "RFM only"}
amap = {"Online Retail II": "online_retail", "Dunnhumby": "dunnhumby"}
ABL = {}
for raw, key in amap.items():
    d = abl[(abl.context == raw) & (abl.variant != "ALL")]
    ABL[key] = [{"variant": relabel.get(v, v), "dng": round(float(dn), 4),
                 "p": (None if pd.isna(p) else round(float(p), 4))}
                for v, dn, p in zip(d.variant, d.dNG_vs_ALL, d.paired_p)]

# ---- SHAP + CQR (both contexts) ----
def shap_top(ctx):
    d = pd.read_csv(P2R / f"{ctx}_shap_stage2.csv").nlargest(10, "mean_abs_shap_norm")
    return {"feature": d["feature"].tolist(), "value": [round(v, 4) for v in d["mean_abs_shap_norm"]]}


def cqr(ctx):
    d = pd.read_csv(P2R / f"{ctx}_cqr_coverage.csv")
    return [{"nominal": float(r.nominal_coverage), "emp": float(r.empirical_coverage),
             "width": float(r.mean_interval_width)} for _, r in d.iterrows()]


SHAP = {k: shap_top(k) for k in CTX}
CQR = {k: cqr(k) for k in CTX}

# ---- cross-context contrast (context_comparison.csv, real) ----
cc = pd.read_csv(P2R.parent / "tables" / "context_comparison.csv").set_index("Unnamed: 0")
def col(key):  # dataset column
    return "Online Retail II" if key == "online_retail" else "Dunnhumby"
CONTRAST = {k: {"zero": round(float(cc.loc["zero_clv_rate", col(k)]) * 100),
                "skew": round(float(cc.loc["clv_skew_nonzero", col(k)]), 1),
                "freq": int(cc.loc["median_frequency", col(k)]),
                "rec": int(round(float(cc.loc["mean_recency_days", col(k)])))} for k in CTX}

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Phase 2 — Uplift-Aware Targeting & Cross-Context CLV | Group 11</title>
<meta name="description" content="Phase 2 dashboard: interactive X5 targeting simulator (client-side upload) plus cross-context CLV benchmark, feature ablation, SHAP and conformal uncertainty for Online Retail II and Dunnhumby."/>
<link rel="stylesheet" href="style.css"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
<script src="https://cdn.sheetjs.com/xlsx-0.20.2/package/dist/xlsx.full.min.js"></script>
<style>
  .p2grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:820px){.p2grid{grid-template-columns:1fr}}
  .p2card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 14px 6px;box-shadow:0 1px 3px rgba(20,35,59,.05)}
  .p2card h3{margin:.1rem 0 .2rem;color:var(--navy2);font-size:1.02rem}
  .p2card .cap{color:var(--muted);font-size:.82rem;margin:0 0 6px}
  .kpirow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 6px}
  @media(max-width:820px){.kpirow{grid-template-columns:repeat(2,1fr)}}
  .k2{background:linear-gradient(180deg,#1f3a5f,#14233b);border-radius:14px;padding:16px 16px 14px;color:#fff}
  .k2 .v{font-size:1.7rem;font-weight:800}
  .k2 .v.blue{color:#7fb3ee}.k2 .v.orange{color:#f08a3c}.k2 .v.green{color:#7ee0a3}.k2 .v.red{color:#f39b8f}
  .k2 .l{font-size:.66rem;letter-spacing:1px;text-transform:uppercase;color:#9fb0c6;margin-top:6px}
  .k2 .s{font-size:.72rem;color:#c7d3e2;margin-top:3px}
  .ctl{margin:6px 0 12px}.ctl select,.ctl input{padding:7px 10px;border-radius:8px;border:1px solid var(--line);font-size:.92rem}
  .cqr{font-size:.82rem;color:var(--muted);padding:2px 6px 8px}
  .plot{width:100%;height:330px}
  .uploadbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
  .btn{display:inline-block;padding:9px 15px;border-radius:9px;font-weight:600;font-size:.9rem;cursor:pointer;border:1px solid transparent}
  .btn.orange{background:var(--accent);color:#fff}.btn.ghost{background:#fff;color:var(--navy2);border-color:var(--line)}
  .status{color:var(--muted);font-size:.85rem}
  .dropzone{margin-top:10px;border:2px dashed #c7d3e2;border-radius:12px;padding:16px;text-align:center;color:var(--muted);font-size:.88rem}
  .sliders{display:flex;flex-wrap:wrap;gap:22px;margin:14px 0 4px;align-items:center}
  .sliders label{font-size:.85rem;color:var(--navy2);font-weight:600}
  .note{background:#eef4fb;border-left:4px solid var(--blue);border-radius:8px;padding:10px 14px;color:var(--navy2);font-size:.88rem;margin:0 0 16px}
  .hint{color:var(--muted);font-size:.82rem;margin-top:8px}
  code{background:#eef2f7;padding:1px 5px;border-radius:5px;font-size:.85em}
</style>
</head><body>

<nav class="nav"><div class="container">
  <div class="brand">CLV<span>·</span>VIP Targeting</div>
  <a class="link" href="index.html">&larr; Main site</a>
  <a class="link" href="index.html#live">Phase 1 Dashboard</a>
  <a class="link" href="#sim">X5 Simulator</a>
  <a class="link" href="#clv">CLV Contexts</a>
</div></nav>

<header class="hero"><div class="container">
  <div class="eyebrow">Phase 2 &middot; Interactive</div>
  <h1>Uplift-Aware Targeting &amp; Cross-Context CLV</h1>
  <p class="sub">Three datasets, three roles: <b>X5 RetailHero</b> (treatment/control) drives the
  targeting simulator; <b>Online Retail II</b> and <b>Dunnhumby</b> drive the cross-context CLV
  benchmark. CLV = who is valuable, uplift = who is persuadable, value-adjusted = who is worth targeting.</p>
  <div class="divider"></div>
  <div class="kpirow">
    <div class="k2"><div class="v green" id="k-raw"></div><div class="l">X5 raw uplift (treated&minus;control)</div><div class="s" id="k-raw-s"></div></div>
    <div class="k2"><div class="v orange" id="k-ov"></div><div class="l">Value vs Uplift overlap</div><div class="s">top-10% &mdash; nearly disjoint</div></div>
    <div class="k2"><div class="v blue" id="k-bp"></div><div class="l">Best policy @20%</div><div class="s">by held-out profit</div></div>
    <div class="k2"><div class="v" id="k-bpf"></div><div class="l">Best profit @20%</div><div class="s" id="k-bpf-s"></div></div>
  </div>
</div></header>

<section id="sim"><div class="container">
  <div class="tag">X5 RetailHero &middot; upload your own data</div>
  <h2 class="h">Targeting simulator</h2>
  <p class="lead">Upload a customer table with columns <code>treatment</code> (0/1),
  <code>converted</code> (0/1 outcome), <code>value</code> (monetary / CLV proxy) and, optionally,
  an <code>uplift_score</code>. Everything is computed in your browser. Policies are evaluated on the
  held-out treated&minus;control difference inside each selected top-K set.</p>

  <div class="card" style="margin-bottom:16px">
    <div class="uploadbar">
      <label class="btn orange" for="fileInput">&#11014; Upload table (CSV / XLSX)</label>
      <input id="fileInput" type="file" accept=".csv,.xlsx,.xls" hidden/>
      <button id="resetBtn" class="btn ghost" type="button">&#8635; Reset to project data</button>
      <span id="status" class="status">Loading project data&hellip;</span>
    </div>
    <div id="dropZone" class="dropzone">&hellip; or drag &amp; drop a CSV / XLSX file here &hellip;</div>
    <p class="hint">Columns matched flexibly (<code>treatment_flg</code>, <code>target</code>,
    <code>monetary</code> all work). &#9654; Try it:
    <a href="data/phase2_demo.csv" download><b>download the demo file</b></a>
    (full X5 held-out test set, 60k rows).</p>
    <div class="sliders">
      <label>Contact cost&nbsp;<output id="costO">100</output>
        <input id="cost" type="range" min="0" max="300" step="10" value="100"/></label>
      <label>Margin&nbsp;<output id="marginO">1.0</output>
        <input id="margin" type="range" min="0.1" max="1" step="0.1" value="1"/></label>
    </div>
  </div>

  <div class="p2grid">
    <div class="p2card"><h3>1 &middot; Uplift@K by policy</h3><p class="cap">incremental response at each budget</p><div id="v1" class="plot"></div></div>
    <div class="p2card"><h3>2 &middot; Policy comparison</h3><p class="cap">incremental value by policy and budget</p><div id="v2" class="plot"></div></div>
    <div class="p2card"><h3>3 &middot; Profit simulation</h3><p class="cap">profit = margin &times; value &minus; cost &times; targeted</p><div id="v3" class="plot"></div></div>
    <div class="p2card"><h3>4 &middot; Balance &amp; raw uplift</h3><p class="cap">conversion rate: treated vs control</p><div id="v4" class="plot"></div><div class="cqr" id="rawnote"></div></div>
  </div>
</div></section>

<section id="clv"><div class="container">
  <div class="tag">Online Retail II &amp; Dunnhumby &middot; CLV ranking (RQ1&ndash;RQ2)</div>
  <h2 class="h">Cross-context CLV</h2>
  <p class="lead">These two datasets have no treatment, so they drive the CLV-ranking side of Phase 2:
  a walk-forward model benchmark, a leave-one-group-out feature ablation, and the trained-model SHAP
  and conformal (CQR) coverage. Pick a dataset:</p>
  <div class="ctl"><label>Dataset:&nbsp;</label>
    <select id="ctx"><option value="online_retail">Online Retail II (e-commerce)</option>
    <option value="dunnhumby">Dunnhumby (grocery)</option></select></div>
  <div class="note" id="contrast"></div>

  <div class="p2grid">
    <div class="p2card"><h3>Walk-forward benchmark</h3><p class="cap">Normalized Gini per model (mean across windows)</p><div id="c1" class="plot"></div></div>
    <div class="p2card"><h3>Feature ablation (leave-one-group-out)</h3><p class="cap">&Delta;NG vs ALL when a group is dropped &mdash; green = significant (p&lt;0.05)</p><div id="c2" class="plot"></div></div>
    <div class="p2card"><h3>Stage-2 SHAP drivers</h3><p class="cap">who is selected, and why</p><div id="c3" class="plot" style="height:360px"></div></div>
    <div class="p2card"><h3>Conformal uncertainty (CQR)</h3><p class="cap">empirical vs nominal coverage</p><div id="c4" class="plot"></div><div class="cqr" id="cqr"></div></div>
  </div>
  <p class="lead" style="margin-top:22px">Source: <code>phase2/results/*.csv</code>. Full Python version
  (Dash, per-model uplift learners) runs locally via <code>python dashboard/app.py</code> &rarr; <code>/uplift</code>.</p>
</div></section>

<footer style="background:#14233b;color:#9fb0c6;padding:26px 0;text-align:center;font-size:.85rem">
  <div class="container">ADY201m &middot; Group 11 &middot; Phase 2: uplift-aware targeting across Online Retail II, Dunnhumby &amp; X5 RetailHero</div>
</footer>

<script>
const SHAP=__SHAP__, CQR=__CQR__, WALK=__WALK__, ABL=__ABL__, CONTRAST=__CONTRAST__, CTXL=__CTXL__;
const DEMO='data/phase2_demo.csv';
const PC={Random:'#c0392b',Value:'#f1c40f',Uplift:'#27ae60','Value-adjusted':'#2f7ed8'};
const KS=[0.10,0.20,0.30];
const L={plot_bgcolor:'#fff',paper_bgcolor:'#fff',font:{family:'Segoe UI,Arial',size:12,color:'#1b2733'},
  margin:{l:56,r:14,t:8,b:66},legend:{orientation:'h',y:-0.25,font:{size:10}},xaxis:{gridcolor:'#eef2f7'},yaxis:{gridcolor:'#eef2f7'}};
const CFG={displaylogo:false,responsive:true};
let RAW=null;

// ================= X5 SIMULATOR =================
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function detect(cols,cands){const lc=cols.map(c=>c.toLowerCase().trim());
  for(const c of cands){let i=lc.indexOf(c);if(i>=0)return cols[i];}
  for(const c of cands){let i=lc.findIndex(x=>x.includes(c));if(i>=0)return cols[i];}return null;}
function num(x){const v=parseFloat(String(x).replace(/[, ]/g,''));return isFinite(v)?v:0;}
function scaleOf(m){m=Math.abs(m);if(m>=1e6)return{d:1e6,s:' (M)'};if(m>=1e3)return{d:1e3,s:' (K)'};return{d:1,s:''};}
function parseRows(rows){
  if(!rows.length)throw 'empty file';
  const cols=Object.keys(rows[0]);
  const ct=detect(cols,['treatment_flg','treatment','treat','is_treated','group','w']);
  const cy=detect(cols,['converted','conversion','target','outcome','response','purchase','bought','y']);
  const cv=detect(cols,['value_proxy','value','monetary','clv','spend','revenue','amount','sales']);
  const cu=detect(cols,['uplift_score','uplift','score','tau','cate']);
  if(!ct||!cy)throw 'need a treatment column and a converted/target column';
  const t=[],y=[],v=[],u=cu?[]:null;
  for(const r of rows){const tv=num(r[ct]);if(!(tv===0||tv===1))continue;
    t.push(tv);y.push(num(r[cy])>0?1:0);v.push(cv?num(r[cv]):1);if(cu)u.push(num(r[cu]));}
  const n=t.length,rnd=[],rng=mulberry32(42);for(let i=0;i<n;i++)rnd.push(rng());
  return {t,y,v,u,rnd,n,cols:{ct,cy,cv,cu}};
}
function computeBase(R){
  const n=R.n,idxAll=Array.from({length:n},(_,i)=>i);
  const pol={Random:R.rnd,Value:R.v};
  if(R.u){pol['Uplift']=R.u;pol['Value-adjusted']=R.u.map((x,i)=>x*R.v[i]);}
  const order={};for(const p in pol)order[p]=idxAll.slice().sort((a,b)=>pol[p][b]-pol[p][a]);
  const base={};
  for(const p in pol){base[p]={};
    for(const k of KS){const ntop=Math.max(1,Math.floor(n*k)),idx=order[p].slice(0,ntop);
      let nt=0,nc=0,st=0,sc=0,vt=0,vc=0;
      for(const i of idx){if(R.t[i]===1){nt++;st+=R.y[i];vt+=R.y[i]*R.v[i];}else{nc++;sc+=R.y[i];vc+=R.y[i]*R.v[i];}}
      base[p][k]={ntop,upl:(nt&&nc)?(st/nt-sc/nc):0,incrVal:(nt&&nc)?(vt/nt-vc/nc)*ntop:0};}}
  let nt=0,nc=0,st=0,sc=0;for(let i=0;i<n;i++){if(R.t[i]===1){nt++;st+=R.y[i];}else{nc++;sc+=R.y[i];}}
  const pt=st/nt,pc=sc/nc,raw=pt-pc,se=Math.sqrt(pt*(1-pt)/nt+pc*(1-pc)/nc);
  let overlap=null;
  if(R.u){const k=Math.max(1,Math.floor(n*0.10)),tv=new Set(order['Value'].slice(0,k)),tu=order['Uplift'].slice(0,k);
    let inter=0;for(const i of tu)if(tv.has(i))inter++;overlap=inter/k;}
  R.base=base;R.policies=Object.keys(pol);R.balance={pt,pc,raw,lo:raw-1.96*se,hi:raw+1.96*se};R.overlap=overlap;
}
function render(){
  const R=RAW,cost=+cost_.value,margin=+margin_.value;
  costO.textContent=cost;marginO.textContent=margin.toFixed(1);
  const Kp=['10%','20%','30%'];
  let mx=0;for(const p of R.policies)for(const k of KS){mx=Math.max(mx,Math.abs(R.base[p][k].incrVal),Math.abs(margin*R.base[p][k].incrVal-cost*R.base[p][k].ntop));}
  const sc=scaleOf(mx),fin=x=>isFinite(x)?x:0;
  const tr=key=>R.policies.map(p=>({x:Kp,y:KS.map(k=>{const b=R.base[p][k];return fin(key==='upl'?b.upl:key==='val'?b.incrVal/sc.d:(margin*b.incrVal-cost*b.ntop)/sc.d);}),
    name:p,mode:'lines+markers',line:{color:PC[p]},marker:{color:PC[p]}}));
  Plotly.newPlot('v1',tr('upl'),Object.assign({},L,{yaxis:{title:'uplift@K',gridcolor:'#eef2f7'},xaxis:{title:'Top-K% targeted',gridcolor:'#eef2f7'}}),CFG);
  Plotly.newPlot('v2',tr('val'),Object.assign({},L,{yaxis:{title:'incremental value'+sc.s,gridcolor:'#eef2f7'},xaxis:{title:'Top-K% targeted',gridcolor:'#eef2f7'}}),CFG);
  Plotly.newPlot('v3',tr('profit'),Object.assign({},L,{yaxis:{title:'profit'+sc.s,gridcolor:'#eef2f7'},xaxis:{title:'Top-K% targeted',gridcolor:'#eef2f7'},
    shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,line:{color:'#888',dash:'dot'}}]}),CFG);
  const b=R.balance;
  Plotly.newPlot('v4',[{type:'bar',x:['Control','Treated'],y:[b.pc,b.pt],marker:{color:['#9aa6b2','#2f7ed8']},text:[b.pc.toFixed(3),b.pt.toFixed(3)],textposition:'outside'}],
    Object.assign({},L,{margin:{l:56,r:14,t:20,b:30},yaxis:{title:'conversion rate',gridcolor:'#eef2f7',rangemode:'tozero'},showlegend:false}),CFG);
  rawnote.innerHTML='<b>Raw uplift:</b> +'+b.raw.toFixed(4)+' &nbsp;95% CI ['+b.lo.toFixed(4)+', '+b.hi.toFixed(4)+']'+(b.lo>0?' &mdash; excludes 0, signal is significant.':'.');
  kraw.textContent='+'+b.raw.toFixed(4);kraw_s.textContent='95% CI ['+b.lo.toFixed(4)+', '+b.hi.toFixed(4)+']';
  kov.textContent=R.overlap==null?'n/a':R.overlap.toFixed(3);
  let bp='',bpf=-Infinity;for(const p of R.policies){const pr=margin*R.base[p][0.20].incrVal-cost*R.base[p][0.20].ntop;if(pr>bpf){bpf=pr;bp=p;}}
  kbp.textContent=bp;kbpf.textContent=(bpf/sc.d).toLocaleString(undefined,{maximumFractionDigits:2})+sc.s.trim().replace(/[()]/g,'');
  kbpf_s.textContent='cost '+cost+', margin '+margin.toFixed(1);
}
const cost_=document.getElementById('cost'),margin_=document.getElementById('margin');
const costO=document.getElementById('costO'),marginO=document.getElementById('marginO');
const kraw=document.getElementById('k-raw'),kraw_s=document.getElementById('k-raw-s'),kov=document.getElementById('k-ov');
const kbp=document.getElementById('k-bp'),kbpf=document.getElementById('k-bpf'),kbpf_s=document.getElementById('k-bpf-s');
function loadTable(rows,label){
  try{RAW=parseRows(rows);}catch(e){document.getElementById('status').innerHTML='<b style="color:#c0392b">Could not read file:</b> '+e;return;}
  computeBase(RAW);
  document.getElementById('status').innerHTML='Showing: <b>'+label+'</b> &mdash; '+RAW.n.toLocaleString()+' rows'+(RAW.u?'':' &middot; <i>no uplift_score</i>');
  render();
}

// ================= CROSS-CONTEXT CLV =================
function drawCLV(ctx){
  // benchmark NG
  const w=WALK[ctx];
  Plotly.newPlot('c1',[{type:'bar',orientation:'h',x:w.map(r=>r.ng).reverse(),y:w.map(r=>r.model).reverse(),
    marker:{color:w.map(r=>r.model==='Hurdle'?'#e05a1f':'#2f7ed8').reverse()},
    text:w.map(r=>r.ng.toFixed(3)).reverse(),textposition:'outside',
    hovertemplate:'%{y}: NG %{x:.3f}<extra></extra>'}],
    Object.assign({},L,{margin:{l:110,r:30,t:8,b:44},xaxis:{title:'Normalized Gini',gridcolor:'#eef2f7',range:[0,1]},yaxis:{automargin:true}}),CFG);
  // ablation dNG
  const a=ABL[ctx];
  Plotly.newPlot('c2',[{type:'bar',orientation:'h',x:a.map(r=>r.dng).reverse(),y:a.map(r=>r.variant).reverse(),
    marker:{color:a.map(r=>(r.p!=null&&r.p<0.05)?'#27ae60':'#9aa6b2').reverse()},
    text:a.map(r=>(r.dng>=0?'+':'')+r.dng.toFixed(4)+(r.p!=null&&r.p<0.05?' *':'')).reverse(),textposition:'outside',
    hovertemplate:'%{y}: dNG %{x:.4f}<extra></extra>'}],
    Object.assign({},L,{margin:{l:120,r:74,t:8,b:44},xaxis:{title:'ΔNG vs ALL (higher = group matters)',gridcolor:'#eef2f7',rangemode:'tozero'},yaxis:{automargin:true}}),CFG);
  // shap
  const s=SHAP[ctx];
  Plotly.newPlot('c3',[{type:'bar',orientation:'h',x:s.value.slice().reverse(),y:s.feature.slice().reverse(),marker:{color:'#2f7ed8'}}],
    Object.assign({},L,{margin:{l:150,r:14,t:8,b:44},xaxis:{title:'mean |SHAP| (normalised)',gridcolor:'#eef2f7'},yaxis:{automargin:true}}),CFG);
  // cqr bars + text
  const q=CQR[ctx];
  Plotly.newPlot('c4',[
    {type:'bar',name:'nominal',x:q.map(r=>Math.round(r.nominal*100)+'%'),y:q.map(r=>r.nominal*100),marker:{color:'#9aa6b2'}},
    {type:'bar',name:'empirical',x:q.map(r=>Math.round(r.nominal*100)+'%'),y:q.map(r=>r.emp*100),marker:{color:'#2f7ed8'}}],
    Object.assign({},L,{barmode:'group',margin:{l:44,r:14,t:8,b:40},yaxis:{title:'coverage (%)',gridcolor:'#eef2f7',range:[0,100]},xaxis:{title:'nominal level'}}),CFG);
  document.getElementById('cqr').innerHTML='<b>Coverage:</b> '+q.map(r=>Math.round(r.nominal*100)+'% &rarr; '+(r.emp*100).toFixed(1)+'% (width '+Math.round(r.width).toLocaleString()+')').join(' &nbsp;|&nbsp; ')+' &mdash; close to nominal.';
  const c=CONTRAST[ctx];
  document.getElementById('contrast').innerHTML='<b>'+CTXL[ctx]+':</b> ~'+c.zero+'% zero-CLV rate, non-zero skew '+c.skew+', median frequency '+c.freq+', mean recency '+c.rec+' days &mdash; '+(ctx==='online_retail'?'sparse &amp; incidence-dominated (Hurdle Stage-1 helps).':'dense &amp; magnitude/sequence-dominated (two-part structure adds little).');
}
document.getElementById('ctx').addEventListener('change',e=>drawCLV(e.target.value));
drawCLV('online_retail');

// ================= wiring =================
function handleFile(file){
  document.getElementById('status').textContent='Parsing '+file.name+'…';
  const ext=file.name.split('.').pop().toLowerCase();
  if(ext==='csv'){Papa.parse(file,{header:true,skipEmptyLines:true,complete:r=>loadTable(r.data,file.name)});}
  else{const rd=new FileReader();rd.onload=e=>{const wb=XLSX.read(e.target.result,{type:'array'});loadTable(XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]]),file.name);};rd.readAsArrayBuffer(file);}
}
document.getElementById('fileInput').addEventListener('change',e=>{if(e.target.files[0])handleFile(e.target.files[0]);});
const dz=document.getElementById('dropZone');
['dragover','dragenter'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.style.background='#eef4fb';}));
['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.style.background='';}));
dz.addEventListener('drop',e=>{if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0]);});
cost_.addEventListener('input',()=>RAW&&render());margin_.addEventListener('input',()=>RAW&&render());
function loadDemo(){document.getElementById('status').textContent='Loading project data…';
  Papa.parse(DEMO,{header:true,download:true,skipEmptyLines:true,complete:r=>loadTable(r.data,'project data (X5 RetailHero held-out test set)')});}
document.getElementById('resetBtn').addEventListener('click',loadDemo);
loadDemo();
</script>
</body></html>
"""

repl = {"__SHAP__": SHAP, "__CQR__": CQR, "__WALK__": WALK, "__ABL__": ABL,
        "__CONTRAST__": CONTRAST, "__CTXL__": CTX}
html = HTML
for k, v in repl.items():
    html = html.replace(k, json.dumps(v))
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, "| bytes", len(html))
