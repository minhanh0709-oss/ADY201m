/* Live interactive dashboard — 100% client-side.
   Renders Plotly charts from default project aggregates, and recomputes
   everything in-browser when a user uploads an Online Retail II CSV/XLSX. */
(function () {
  "use strict";

  const SEG_ORDER = ["Champion", "Loyal", "At-Risk", "Lost", "New"];
  const SEG_COLOR = {
    Champion: "#1e7e34", Loyal: "#2f7ed8", "At-Risk": "#e0a800",
    Lost: "#d62728", New: "#6c757d"
  };
  const PCONF = { responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  const baseLayout = (extra) => Object.assign({
    margin: { l: 60, r: 34, t: 50, b: 54 }, height: 360,
    paper_bgcolor: "white", plot_bgcolor: "white",
    font: { family: "Segoe UI, sans-serif", size: 12, color: "#1b2733" },
    legend: { orientation: "h", y: -0.22, x: 0.5, xanchor: "center" }
  }, extra || {});

  const $ = (id) => document.getElementById(id);
  const fmtMoney = (v) => "$" + Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });

  // ---------- KPI cards ----------
  function renderKpis(k) {
    $("liveKpis").innerHTML = [
      ["Transactions", Number(k.transactions).toLocaleString()],
      ["Customers", Number(k.customers).toLocaleString()],
      ["Countries", Number(k.countries).toLocaleString()],
      ["Total revenue", fmtMoney(k.revenue)]
    ].map(([l, v]) => `<div class="kpi-l"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  }

  // ---------- charts ----------
  function chartMonthly(m) {
    const x = m.map(d => d.ym);
    Plotly.newPlot("chMonthly", [
      { type: "bar", x, y: m.map(d => d.revenue), name: "Revenue ($)",
        marker: { color: "#9ec5e8" }, yaxis: "y2", opacity: 0.75 },
      { type: "scatter", mode: "lines+markers", x, y: m.map(d => d.active),
        name: "Active customers", line: { color: "#d62728", width: 2.5 } }
    ], baseLayout({
      title: "Monthly Active Customers & Revenue",
      margin: { l: 64, r: 64, t: 50, b: 50 },
      xaxis: { gridcolor: "#eee", automargin: true },
      yaxis: { title: { text: "Active customers", standoff: 8 },
               gridcolor: "#eee", automargin: true },
      // NB: no title on the right overlaying axis — Plotly's automargin does not
      // reserve space for an overlaying-axis title, so it collides with the ticks.
      // The "$" tickprefix + the "Revenue ($)" legend already label this series.
      yaxis2: { overlaying: "y", side: "right", showgrid: false,
                tickprefix: "$", tickformat: "~s", automargin: true }
    }), PCONF);
  }

  function chartSegments(segs) {
    const map = {}; segs.forEach(s => map[s.segment] = s.count);
    const traces = SEG_ORDER.filter(s => map[s] != null).map(s => ({
      type: "bar", x: [s], y: [map[s]], name: s,
      marker: { color: SEG_COLOR[s] }, text: [map[s]], textposition: "outside",
      cliponaxis: false
    }));
    Plotly.newPlot("chSeg", traces, baseLayout({
      title: "Customers by RFM Segment",
      xaxis: { categoryorder: "array", categoryarray: SEG_ORDER, automargin: true },
      yaxis: { title: "Customers", gridcolor: "#eee", automargin: true }
    }), PCONF);
  }

  function chartCountry(c) {
    const top = c.slice().sort((a, b) => a.revenue - b.revenue);
    Plotly.newPlot("chCountry", [{
      type: "bar", orientation: "h",
      x: top.map(d => d.revenue), y: top.map(d => d.country),
      marker: { color: "#2f7ed8" },
      hovertemplate: "%{y}<br>%{x:$,.0f}<br>%{customdata} customers<extra></extra>",
      customdata: top.map(d => d.customers)
    }], baseLayout({
      title: "Revenue by Country (Top 15)", showlegend: false,
      margin: { l: 130, r: 24, t: 50, b: 48 },
      xaxis: { title: "Revenue", tickprefix: "$", tickformat: "~s",
               gridcolor: "#eee", automargin: true },
      yaxis: { automargin: true }
    }), PCONF);
  }

  function chartScatter(pts) {
    const traces = SEG_ORDER.map(seg => {
      const p = pts.filter(d => d.seg === seg);
      return {
        type: "scattergl", mode: "markers", name: seg,
        x: p.map(d => d.r), y: p.map(d => d.m),
        marker: { color: SEG_COLOR[seg], size: 6, opacity: 0.65 },
        hovertemplate: "Recency %{x}d<br>Monetary %{y:$,.0f}<extra>" + seg + "</extra>"
      };
    }).filter(t => t.x.length);
    Plotly.newPlot("chScatter", traces, baseLayout({
      title: "RFM Scatter — click a legend to filter",
      xaxis: { title: "Recency (days)", gridcolor: "#eee", automargin: true },
      yaxis: { title: "Monetary", type: "log", tickprefix: "$", tickformat: "~s",
               gridcolor: "#eee", automargin: true }
    }), PCONF);
  }

  function chartHist(vals) {
    const v = vals.filter(x => x > 0).map(x => Math.log10(x));
    Plotly.newPlot("chHist", [{
      type: "histogram", x: v, marker: { color: "#6fc7bf" }, nbinsx: 40
    }], baseLayout({
      title: "Customer Monetary Distribution (log10 $)",
      xaxis: { title: "log10(Monetary $)", gridcolor: "#eee", automargin: true },
      yaxis: { title: { text: "Customers", standoff: 8 },
               gridcolor: "#eee", automargin: true }, showlegend: false
    }), PCONF);
  }

  function renderTopTable(rows) {
    const head = "<table><thead><tr><th>#</th><th>Customer</th><th>Segment</th>" +
      "<th>R</th><th>F</th><th>Monetary</th><th>Country</th></tr></thead><tbody>";
    const body = rows.map((d, i) =>
      `<tr><td>${i + 1}</td><td>${d.id}</td><td>${d.seg || ""}</td>` +
      `<td class="r">${d.r}</td><td class="r">${d.f}</td>` +
      `<td class="r">${fmtMoney(d.m)}</td><td>${d.country}</td></tr>`).join("");
    $("topTable").innerHTML = head + body + "</tbody></table>";
  }

  function renderAll(D, label) {
    renderKpis(D.kpis);
    chartMonthly(D.monthly);
    chartSegments(D.segments);
    chartCountry(D.countries);
    chartScatter(D.scatter);
    chartHist(D.monetary);
    renderTopTable(D.top);
    $("status").textContent = "Showing: " + (label || D.source || "project data");
  }

  // ---------- in-browser pipeline for uploaded data ----------
  function findKey(keys, names) {
    const low = keys.map(k => k.toLowerCase().trim());
    for (const n of names) { const i = low.indexOf(n); if (i >= 0) return keys[i]; }
    return null;
  }
  function toMs(v) {
    if (v instanceof Date) return v.getTime();
    const d = new Date(String(v)); const t = d.getTime();
    return isNaN(t) ? null : t;
  }
  function ntile(values) {
    const s = values.slice().sort((a, b) => a - b);
    const q = p => s[Math.max(0, Math.floor(p * (s.length - 1)))];
    const t = [q(0.2), q(0.4), q(0.6), q(0.8)];
    return v => v <= t[0] ? 1 : v <= t[1] ? 2 : v <= t[2] ? 3 : v <= t[3] ? 4 : 5;
  }

  function computeFromRows(rows) {
    if (!rows.length) throw new Error("Empty file");
    const keys = Object.keys(rows[0]);
    const kInv = findKey(keys, ["invoice", "invoiceno"]);
    const kQty = findKey(keys, ["quantity", "qty"]);
    const kDate = findKey(keys, ["invoicedate", "date"]);
    const kPrice = findKey(keys, ["price", "unitprice"]);
    const kCust = findKey(keys, ["customer id", "customerid", "custid", "customer"]);
    const kCtry = findKey(keys, ["country"]);
    if (!(kQty && kDate && kPrice && kCust)) {
      throw new Error("Missing expected columns (need Quantity, InvoiceDate, Price, Customer ID).");
    }
    let maxMs = 0;
    const cust = new Map();         // id -> {lastMs, inv:Set, mon, ctry}
    const monthly = new Map();      // ym -> {rev, cust:Set, inv:Set}
    const country = new Map();      // ctry -> {rev, cust:Set}
    let nClean = 0, totRev = 0;

    for (const row of rows) {
      const id = row[kCust]; if (id === null || id === undefined || id === "") continue;
      const qty = parseFloat(row[kQty]); const price = parseFloat(row[kPrice]);
      if (!(qty > 0) || !(price > 0)) continue;
      const ms = toMs(row[kDate]); if (ms === null) continue;
      const total = qty * price;
      const inv = kInv ? row[kInv] : (id + "|" + ms);
      const ctry = kCtry ? (row[kCtry] || "Unknown") : "Unknown";
      nClean++; totRev += total; if (ms > maxMs) maxMs = ms;

      let c = cust.get(id);
      if (!c) { c = { lastMs: 0, inv: new Set(), mon: 0, ctry }; cust.set(id, c); }
      if (ms > c.lastMs) c.lastMs = ms;
      c.inv.add(inv); c.mon += total; c.ctry = ctry;

      const ym = new Date(ms).toISOString().slice(0, 7);
      let mm = monthly.get(ym);
      if (!mm) { mm = { rev: 0, cust: new Set(), inv: new Set() }; monthly.set(ym, mm); }
      mm.rev += total; mm.cust.add(id); mm.inv.add(inv);

      let cc = country.get(ctry);
      if (!cc) { cc = { rev: 0, cust: new Set() }; country.set(ctry, cc); }
      cc.rev += total; cc.cust.add(id);
    }
    if (!cust.size) throw new Error("No valid rows after cleaning.");
    const snap = maxMs + 86400000;

    const custArr = [];
    cust.forEach((c, id) => custArr.push({
      id: String(id).replace(/\.0$/, ""),
      r: Math.floor((snap - c.lastMs) / 86400000),
      f: c.inv.size, m: c.mon, country: c.ctry
    }));
    const sR = ntile(custArr.map(c => c.r));
    const sF = ntile(custArr.map(c => c.f));
    const sM = ntile(custArr.map(c => c.m));
    custArr.forEach(c => {
      const score = (6 - sR(c.r)) + sF(c.f) + sM(c.m);
      c.seg = score >= 13 ? "Champion" : score >= 10 ? "Loyal"
        : score >= 7 ? "At-Risk" : score >= 5 ? "Lost" : "New";
    });

    const segCount = {};
    custArr.forEach(c => segCount[c.seg] = (segCount[c.seg] || 0) + 1);
    const segments = Object.keys(segCount).map(s => ({ segment: s, count: segCount[s] }));

    const monthlyArr = [...monthly.entries()].sort((a, b) => a[0] < b[0] ? -1 : 1)
      .map(([ym, v]) => ({ ym, revenue: v.rev, active: v.cust.size, invoices: v.inv.size }));
    const countries = [...country.entries()]
      .map(([k, v]) => ({ country: k, revenue: v.rev, customers: v.cust.size }))
      .sort((a, b) => b.revenue - a.revenue).slice(0, 15);

    const pos = custArr.filter(c => c.m > 0);
    const scatter = (pos.length > 1500 ? sample(pos, 1500) : pos)
      .map(c => ({ r: c.r, m: c.m, f: c.f, seg: c.seg }));
    const monetary = pos.map(c => c.m);
    const top = pos.slice().sort((a, b) => b.m - a.m).slice(0, 20)
      .map(c => ({ id: c.id, r: c.r, f: c.f, m: c.m, country: c.country, seg: c.seg }));

    return {
      source: "uploaded dataset",
      kpis: { transactions: nClean, customers: cust.size, countries: country.size, revenue: totRev },
      monthly: monthlyArr, segments, countries, scatter, monetary, top
    };
  }
  function sample(arr, n) {
    const out = arr.slice(); const step = out.length / n;
    const r = []; for (let i = 0; i < n; i++) r.push(out[Math.floor(i * step)]); return r;
  }

  // ---------- file handling ----------
  function setBusy(msg) { $("status").textContent = msg; }

  function handleFile(file) {
    const name = (file.name || "").toLowerCase();
    setBusy("Processing " + file.name + " …");
    const done = (rows) => {
      try {
        const D = computeFromRows(rows);
        renderAll(D, file.name + "  (" + D.kpis.transactions.toLocaleString() + " valid rows)");
      } catch (e) { setBusy("⚠ " + e.message); }
    };
    if (name.endsWith(".xlsx") || name.endsWith(".xls")) {
      const rd = new FileReader();
      rd.onload = (ev) => {
        try {
          const wb = XLSX.read(new Uint8Array(ev.target.result), { type: "array", cellDates: true });
          const ws = wb.Sheets[wb.SheetNames[0]];
          done(XLSX.utils.sheet_to_json(ws, { raw: false }));
        } catch (e) { setBusy("⚠ Could not read Excel file: " + e.message); }
      };
      rd.readAsArrayBuffer(file);
    } else {
      Papa.parse(file, {
        header: true, skipEmptyLines: true, worker: true,
        complete: (res) => done(res.data),
        error: (e) => setBusy("⚠ CSV parse error: " + e.message)
      });
    }
  }

  // ---------- wire up ----------
  function init() {
    if (!window.DASH_DEFAULT) return;
    renderAll(window.DASH_DEFAULT);

    $("fileInput").addEventListener("change", (e) => {
      if (e.target.files[0]) handleFile(e.target.files[0]);
    });
    $("resetBtn").addEventListener("click", () => renderAll(window.DASH_DEFAULT));

    const dz = $("dropZone");
    ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.remove("drag");
    }));
    dz.addEventListener("drop", (e) => {
      if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
