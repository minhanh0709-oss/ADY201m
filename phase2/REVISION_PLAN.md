# Phase 2 — Detailed Revision Plan (post-review v2)

Two reviewers, one consensus. This plan maps every concern to a concrete action, with a
guiding principle the user made explicit.

## 0. Guiding principle (non-negotiable)
**CLV evaluation must reuse Phase 1's walk-forward temporal protocol, NOT a random 80/20 split.**
Random split on a temporal CLV problem leaks future structure and inflates ranking metrics:
Phase 2 currently reports Online Retail II Hurdle **NG = 0.9118 / RC@10 = 70.32** from a single
stratified random split, whereas Phase 1's temporally-valid walk-forward gives
**NG = 0.834 ± 0.057 / RC@10 = 60.96 ± 9.26** (results/MASTER_TABLE.csv, hurdle_walkforward.csv).
Phase 2's Online Retail II numbers must land back on the Phase 1 numbers. The X5 uplift study is a
cross-sectional randomised campaign (not a temporal CLV task), so its 70/30 split stays — but we
strengthen it with a covariate-balance check and CIs.

## 1. Consolidated issue → priority matrix
| # | Issue | Raised by | Severity | Fix tier |
|---|-------|-----------|----------|----------|
| I1 | CLV uses single random 80/20, not walk-forward | R1 (main), user | Blocker | P0 |
| I2 | OR-II results (0.91) diverge from Phase 1 (0.834); unexplained | R1 | Blocker | P0 |
| I3 | Methodology mixes Modules A/B/C/D | R2 (main) | Blocker | P0 |
| I4 | LaTeX encoding: `<`/`>` render as ¡/¿, equation `CLV[i`, table order | R1 | High | P0 |
| I5 | Semantic claim too strong vs ablation evidence | R1, R2 | High | P1 |
| I6 | Profit simulation under-specified (cost, margin, units, formula) | R1, R2 | High | P1 |
| I7 | X5 "randomised" unverified; no covariate-balance table | R1 | High | P1 |
| I8 | Value-adjusted uplift under-emphasised; no overlap analysis | R2 (main) | High | P1 |
| I9 | Report CIs (not just p) in main tables | R1 | High | P1 |
| I10 | Discussion shallow (why CLV/uplift disagree; mechanism) | R2 | High | P1 |
| I11 | Related Work too short; only 10 refs (need 25–40) | R1, R2 | High | P1 |
| I12 | Abstract over-claims; figure captions ("dominates") | R1 | Med | P2 |
| I13 | Limitations thin (Dunnhumby n=2,500; single campaign) | R2 | Med | P2 |
| I14 | Tables 3/4 cluttered; no literature-comparison table | R1, R2 | Med | P2 |

---

## 2. P0 — must-fix (blocks submission)

### P0.1 Walk-forward CLV protocol (the core fix) — addresses I1, I2
**Online Retail II (reuse Phase 1, READ-ONLY):**
- Reuse `data/processed/window_{1..5}_features.csv` + `semantic_v2_window_{1..5}.npz`
  (anchor 2009-12-01; W1–W5 observation/prediction windows from Phase 1).
- Run the Phase 2 model suite (Mean, Monetary, RFM, Ridge, RF, XGBoost, LightGBM, Hurdle) on
  **each window**, stratified by IsVIP within window (Phase 1 protocol).
- Report **mean ± std across windows** and a per-window appendix table.
- **Acceptance test:** Phase 2 Hurdle on OR-II must reproduce Phase 1 to within noise
  (NG ≈ 0.83–0.84, RC@10 ≈ 58–62). If it does, the divergence (I2) is resolved by construction.

**Dunnhumby (new week-level windows):**
- D1 obs 26w → pred 8w; D2 39w → 13w; D3 52w → 13w; D4 78w → 13w (grocery is higher-frequency).
- Build with the existing schema normaliser + a walk-forward splitter; report mean ± std.

**New script:** `experiments/p2_12_walkforward_clv.py`
**New results:** `results/{or2,dunnhumby}_clv_walkforward.csv` (per-window + summary).
**Headline tables (tab:bench_*) switch to walk-forward mean ± std.**

### P0.2 Protocol-alignment note/appendix — addresses I2
Add a short subsection "Protocol alignment with Phase 1": state windows, horizons, feature set,
and that all OR-II numbers are walk-forward means (not random split), so they are directly
comparable to Phase 1. Remove or clearly label any random-split numbers.

### P0.3 Methodology restructure into 4 modules — addresses I3
Rewrite Section 4 as explicit subsections:
- 4.1 Module A — Hurdle CLV ranking (OR-II + Dunnhumby)
- 4.2 Module B — Semantic product profiling
- 4.3 Module C — Uplift learners (X5 only)
- 4.4 Module D — **Value-adjusted uplift targeting** (flagged as the central contribution)
- 4.5 Evaluation protocol (walk-forward CLV; 70/30 RCT uplift; bootstrap; conformal)

### P0.4 LaTeX/encoding/table fixes — addresses I4
- In `p2_11_make_paper_tables.py`, sanitise cell text: `<` → `$<$`, `>` → `$>$`, `<=` → `$\le$`,
  `>=` → `$\ge$` (root cause of `Hurdle¿Monetary`, `p(diff¡=0)` under OT1 font encoding).
- Verify all inline equations (`$\widehat{CLV}_i$`) compile; fix the `CLV[i` artifact.
- Reorder tables so Table numbering follows first mention (datastats before bench, etc.); use
  `\ref` everywhere; consider `[H]`/`table*` placement to avoid float drift.

---

## 3. P1 — important (raises quality from 6.5 to 8+)

### P1.1 Leave-one-group-out ablation + reword — addresses I5
Current ablation only adds groups to RFM. Add the dropping direction:
ALL, ALL−semantic, ALL−sequence, ALL−behavioural, RFM-only. Run on **both contexts** under
walk-forward; report ΔNG with 95% CI. **New script:** `p2_13_ablation_loo.py`.
**Reword** the claim to: "additional behavioural, sequence, and semantic groups *collectively*
improve ranking beyond RFM; sequence features dominate grocery, while semantic profiles mainly aid
interpretation/segmentation." (Matches Phase 1's honest finding.)

### P1.2 X5 audit + covariate-balance table — addresses I7
Table: treatment/control counts, response rate treated/control, raw uplift + CI, n clients,
n purchases, missing rates; plus **pre-period covariate balance** (age, gender, monetary,
frequency, recency, category diversity) treated vs control with standardised mean differences.
Soften wording to "treatment/control labels provided by the campaign dataset (balanced design;
see Table)". **New script:** `p2_14_x5_balance.py`.

### P1.3 Profit-simulation formalisation — addresses I6
State explicitly in text + caption:
- value proxy V̂ = historical monetary over the **pre-communication** window (a proxy, not future
  CLV); units = rubles; revenue (not margin) unless margin assumed.
- Formula: Profit@K = Σ_{i∈TopK} (τ̂_i · V̂_i · m − c), with margin m and contact cost c.
- **Sensitivity grid:** sweep c ∈ {50,100,200} and m ∈ {0.1,0.2,0.3}; show the policy ranking is
  stable. Explain that negative incremental revenue (Random@30%, Value-only@10%) is estimator
  noise on RCT subsets (already visible in wide bootstrap CIs).
**Extend** `p2_09_policy_comparison.py`.

### P1.4 CIs in main tables — addresses I9
Report NG-diff, Qini-diff, and profit-diff as **mean + 95% CI** (not just p) in the headline
tables. Already computed in `p2_10`; surface them in `tables_auto.tex`.

### P1.5 Policy overlap analysis — addresses I8
Table + figure: Jaccard/overlap of top-10% sets across {CLV-only, Uplift-only, Value-adjusted}.
Directly demonstrates CLV and uplift select different customers (the paper's thesis).
**New script:** `p2_15_policy_overlap.py`; figure `x5_policy_overlap.png`.

### P1.6 Deepen Discussion — addresses I8, I10
Add: (a) a quantified "disagreement" analysis (overlap numbers); (b) mechanism — why e-commerce
(incidence-driven, zero-inflated) rewards the Hurdle stage-1 while grocery (habitual, magnitude-
driven) rewards sequence features; (c) a humbler "transferable insights" paragraph for emerging
markets (principles, not numbers).

### P1.7 Expand Related Work + references to 25–40 — addresses I11
Restructure into: (i) CLV modelling & explainability; (ii) uplift / heterogeneous treatment
effects; (iii) profit-driven & value-based targeting; (iv) cross-context / generalisation.
**Key papers to add (high relevance):**
- Ascarza (2018) *Retention futility: targeting high-risk customers might be ineffective* — directly
  supports "response ≠ persuadable".
- Devriendt et al. (2021) uplift evaluation; Diemert et al. (2018) Criteo uplift benchmark.
- Radcliffe & Surry (2011); Rzepakowski & Jaroszewicz (2012) uplift trees.
- Athey & Imbens (2016); Nie & Wager (2021, R-learner) HTE.
- Gubela & Lessmann (2020) profit-driven uplift; Verbeke et al. (2012) profit-driven analytics;
  Lemmens & Gupta (2020) profit-based churn.
- Gupta et al. (2006) CLV models survey; Vanderveld et al. (2016) engagement-based LTV;
  Chen et al. (2018) deep CLV; Hitsch & Misra (2018) targeting policy.
Target ~28–32 references.

---

## 4. P2 — polish
- P2.1 Trim Abstract claims (state results once, hedge profit).
- P2.2 Figure captions: replace "dominates" with "outperforms ... under this protocol".
- P2.3 Limitations: add Dunnhumby small sample (~2,500 households), single campaign/market,
  value-proxy approximation, modest absolute Qini.
- P2.4 Add a small literature-comparison table (method, dataset, metric, vs ours).

---

## 5. Script change list
| Script | Action |
|--------|--------|
| `p2_12_walkforward_clv.py` | NEW — walk-forward CLV both contexts (reuse Phase 1 OR-II windows) |
| `p2_13_ablation_loo.py` | NEW — leave-one-group-out ablation, both contexts |
| `p2_14_x5_balance.py` | NEW — X5 treatment + covariate balance table |
| `p2_15_policy_overlap.py` | NEW — top-K overlap CLV vs uplift vs value-adjusted |
| `p2_09_policy_comparison.py` | EXTEND — profit formula, margin/cost, sensitivity grid |
| `p2_10_robustness.py` | EXTEND — emit diffs as mean + 95% CI for tables |
| `p2_11_make_paper_tables.py` | EXTEND — sanitise `<`/`>`, add new tables, reorder |
| `sn-article.tex` | REWRITE — modules, deeper Discussion, expanded Related Work, refs |
| `references.bib` | EXTEND — to ~28–32 verified entries |

## 6. Revision sprint (suggested 3–4 sessions)
- **S1 (P0):** walk-forward CLV (OR-II reuse + Dunnhumby D1–D4) → confirm OR-II matches Phase 1;
  methodology restructure; LaTeX/encoding fixes. *Biggest score impact.*
- **S2 (P1 experiments):** LOO ablation; X5 balance; profit sensitivity; overlap; CIs in tables.
- **S3 (P1 writing):** Discussion depth; Related Work + references; reword semantic claim.
- **S4 (P2 + recompile):** polish, captions, limitations, literature table; recompile to 12 pages
  + rebuild Overleaf zip (new name `_v2`).

## 7. Expected score impact
| Criterion | Now | After |
|-----------|-----|-------|
| Methodology detail | 6.5 | 8.0 (modules + protocol) |
| Statistical reliability | 6.0 | 8.0 (walk-forward + CIs + balance) |
| Results | 7.5 | 8.5 (aligned, overlap, sensitivity) |
| Paper readiness | 6.5–7 | 8–8.5 |
| Q3/Q4 accept chance | ~70% | ~80–85% |

## 8. Naming discipline (user instruction)
All new outputs use new names; never overwrite Phase 1 files or the existing paper/proposal.
The revised paper zip will be `Group11_Phase2_Overleaf_v2.zip`; revised proposal artefacts keep
the `Group11_Phase2_*` prefix.
