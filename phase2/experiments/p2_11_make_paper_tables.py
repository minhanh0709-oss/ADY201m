"""Phase 2 — generate LaTeX tables for paper2 from result CSVs (kept in sync).
Revision v2: CLV tables use the WALK-FORWARD summary (temporal, Phase 1-aligned), and all
cell text is sanitised so '<'/'>' do not render as inverted punctuation under OT1.
Output: paper2/tables_auto.tex  (\\input by sn-article.tex)
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

PAPER = C.PHASE2_ROOT / "paper2"
PAPER.mkdir(exist_ok=True)


def san(s):
    """Escape LaTeX specials OUTSIDE math spans; leave $...$ (e.g. $\\pm$, $<$) intact."""
    s = str(s)
    parts = s.split("$")
    for i in range(0, len(parts), 2):          # even indices are outside math
        p = parts[i].replace("<=", r"$\le$").replace(">=", r"$\ge$")
        p = p.replace("<", r"$<$").replace(">", r"$>$")
        p = p.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        parts[i] = p
    return "$".join(parts)


def tex_table(df, caption, label, cols=None):
    df = df[cols] if cols else df
    align = "l" + "r" * (df.shape[1] - 1)
    lines = [r"\begin{table}[t]\centering",
             rf"\caption{{{caption}}}\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\hline",
             " & ".join(san(c) for c in df.columns) + r" \\", r"\hline"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                if v != v:                       # NaN -> en-dash
                    cells.append("--")
                else:
                    cells.append(f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}")
            else:
                cells.append(san(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def fmt_pm(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def main():
    out = []

    # 2. WALK-FORWARD CLV benchmark (temporal; Phase 1-aligned) -- both contexts
    wf = pd.read_csv(C.RESULTS / "clv_walkforward_summary.csv")
    wf = wf[~wf["model"].isin(["Mean", "RFM_score"])]  # drop sanity/weak rows for length
    wf["NG"] = [fmt_pm(m, s) for m, s in zip(wf.Norm_Gini_mean, wf.Norm_Gini_std)]
    wf["RC@10"] = [fmt_pm(m, s) for m, s in zip(wf.RevCapture_10_mean, wf.RevCapture_10_std)]
    wf["Prec@10"] = wf.Precision_10_mean.round(3)
    wf_tab = wf[["context", "model", "NG", "RC@10", "Prec@10"]]
    out.append(tex_table(wf_tab,
               "Walk-forward CLV benchmark. NG and RC@10 are mean $\\pm$ std across temporal "
               "windows (Online Retail II on the Phase~1 windows W1--W3, Dunnhumby on D1--D4); "
               "Precision@10 is the across-window mean. Results are temporally valid and lie in the "
               "same range as the Phase~1 walk-forward benchmark (NG~$\\approx$~0.83); small "
               "differences follow from the feature set and the window subset.",
               "tab:clvwf"))

    # 4. X5 uplift models
    u = pd.read_csv(C.RESULTS / "x5_uplift_models.csv")
    out.append(tex_table(u, "X5 uplift model comparison (RQ3). The response model ranks below random.",
                         "tab:uplift", cols=["model", "Qini_AUC", "AUUC", "uplift@10", "uplift@20"]))

    # 5. policy comparison (K=10,20; incr. value shown so profit = value - cost is transparent)
    p = pd.read_csv(C.RESULTS / "x5_policy_comparison.csv")
    p = p[p["K"].isin(["10%", "20%"])].rename(columns={
        "incremental_conversions": "incr. conv.", "incremental_revenue": "incr. value (M)",
        "profit": "profit (M)"})
    p["incr. conv."] = p["incr. conv."].map(lambda v: f"{v:.1f}")            # 1 decimal
    p["incr. value (M)"] = p["incr. value (M)"].map(lambda v: f"{v/1e6:,.2f}")  # million rubles
    p["profit (M)"] = p["profit (M)"].map(lambda v: f"{v/1e6:,.2f}")            # million rubles
    p = p[["policy", "K", "incr. conv.", "incr. value (M)", "profit (M)"]]
    out.append(tex_table(p, "Targeting policy comparison on X5 (RQ4). The value proxy "
               "$\\hat V_i$ is pre-period historical monetary spend (a gross-value proxy, not "
               "future CLV). Incremental conversions and incremental value are held-out "
               "treated-minus-control estimates within each selected top-$K$ set (Eq.~\\ref{eq:profit}); "
               "the headline uses margin $m=1$, so profit $=$ (incr.\\ value) $-$ contact cost with "
               "$c=100$ rubles per contact (value and profit in millions of rubles). Across a "
               "sensitivity sweep ($m\\in\\{0.1,0.2,0.3\\}$, $c\\in\\{50,100,200\\}$) value-adjusted "
               "uplift has the highest point-estimate profit in all nine settings.", "tab:policy"))

    # 6. walk-forward / LOO significance of key claims  [T1.5 p<0.001; clean 3-col format]
    import re as _re
    r = pd.read_csv(C.RESULTS / "robustness_summary.csv")
    dvals, pvals = [], []
    for e in r["evidence"]:
        m = _re.search(r"dNG=([+-]?[0-9.]+)", e)
        dvals.append(f"${m.group(1)}$" if m else "--")
        if "0.000" in e:
            pvals.append("$<0.001$")
        else:
            m2 = _re.search(r"p=([0-9.]+)", e)
            pvals.append(f"${m2.group(1)}$" if m2 else "--")
    rtab = pd.DataFrame({"claim": r["claim"], "$\\Delta$NG": dvals, "$p$": pvals})
    out.append(tex_table(rtab, "Significance of key claims ($\\Delta$NG with walk-forward paired "
                         "$t$-tests for CLV; bootstrap for the uplift Qini comparison). With only "
                         "3--4 windows, the CLV tests are robustness diagnostics rather than "
                         "definitive inference. Only sequence-in-grocery and uplift-vs-response are "
                         "significant.", "tab:robust"))

    # 7. LOO feature ablation (RQ2)
    abl = pd.read_csv(C.RESULTS / "ablation_loo_summary.csv")
    out.append(tex_table(abl, "Leave-one-group-out ablation (Hurdle, walk-forward). "
                         "dNG\\_vs\\_ALL $>$ 0 means dropping the group hurts; paired\\_p is the "
                         "paired t-test across windows.", "tab:ablation"))

    # (X5 covariate-balance table dropped for length; max |SMD|<0.01 stated in text.)

    # 9. policy overlap (RQ4) -- K=10% only to save space  [T0.4 define overlap]
    ov = pd.read_csv(C.RESULTS / "x5_policy_overlap.csv")
    ov = ov[ov["K"] == "10%"][["policy_A", "policy_B", "overlap"]]
    out.append(tex_table(ov, "Top-10\\% targeting overlap between policies, defined as "
               "$|A_K\\cap B_K|/K$ where $A_K,B_K$ are the top-$K$ customer sets selected by each "
               "policy. Value and Uplift select almost disjoint customers (0.007), evidencing that "
               "value $\\neq$ persuadability.", "tab:overlap"))

    # 10. CQR coverage (main body, compact) -- SHAP reported inline in the text
    cq_rows = []
    for key, lab in [("online_retail", "Online Retail II"), ("dunnhumby", "Dunnhumby")]:
        c = pd.read_csv(C.RESULTS / f"{key}_cqr_coverage.csv")
        c.insert(0, "context", lab)
        cq_rows.append(c[["context", "nominal_coverage", "empirical_coverage", "mean_interval_width"]])
    cq = pd.concat(cq_rows, ignore_index=True)
    out.append(tex_table(cq, "Conformalized quantile regression: empirical coverage and mean "
               "interval width per context. Coverage is close to nominal; grocery intervals are "
               "tighter, consistent with its lighter tail.", "tab:cqr"))

    (PAPER / "tables_auto.tex").write_text("\n".join(out), encoding="utf-8")
    print(f"  -> {PAPER / 'tables_auto.tex'}  ({len(out)} tables)")


if __name__ == "__main__":
    main()
