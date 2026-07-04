# TỔNG HỢP KIẾN THỨC — PHASE 2
## CLV Ranking → Uplift-Aware Targeting across Three Retail Contexts
**ADY201m · Group 11 · FPT University**

> **Luận điểm trung tâm:** **CLV** cho biết *ai có giá trị (who is valuable)* · **Uplift** cho biết *ai bị campaign tác động (who is persuadable)* · **Value-adjusted uplift** cho biết *ai đáng target (who is worth targeting)*.

---

## MỤC LỤC
1. Bài toán & Research Questions
2. Dữ liệu (3 dataset, 3 vai trò)
3. Input / Output từng bài toán
4. SQL (DuckDB)
5. Pipeline Python
6. Features (4 nhóm)
7. Models (CLV + Uplift + Policy)
8. Metrics (đo cái gì, vì sao)
9. Evaluation protocol (walk-forward, RCT, bootstrap)
10. Kết quả chính (RQ1–RQ4)
11. Explainability & Uncertainty (SHAP, CQR)
12. Gaps / Limitations
13. Deliverables & Reproducibility

---

## 1. BÀI TOÁN & RESEARCH QUESTIONS

**Vấn đề.** Khách hàng giá trị cao (high-CLV) *không phải lúc nào cũng* là đối tượng target tốt nhất: nhiều khách high-value là *sure-buyers* — vẫn mua dù không nhận campaign → target họ lãng phí ngân sách. Cần nối **CLV ranking** với **uplift-aware targeting**.

| RQ | Câu hỏi | Trả lời bằng |
|----|---------|--------------|
| **RQ1** | Framework Hurdle–Semantic CLV có robust cross-context (e-com vs grocery)? | Walk-forward CLV benchmark |
| **RQ2** | Nhóm feature nào cải thiện ranking: RFM, sequence, hay semantic? | LOO feature ablation |
| **RQ3** | Uplift model có nhận diện persuadable tốt hơn response model? | X5 uplift benchmark (Qini/AUUC) |
| **RQ4** | Value-adjusted uplift có tăng held-out profit? | Policy simulation trên RCT |

---

## 2. DỮ LIỆU (3 DATASET, 3 VAI TRÒ — CAUSAL-CLEAN)

| Dataset | Vai trò | Quy mô | Đặc điểm CLV | Không được claim |
|---------|---------|--------|--------------|------------------|
| **Online Retail II** (UCI) | Baseline CLV (Phase 1, read-only) | ~5,038 khách / 805,549 giao dịch (2009–2011, UK e-commerce) | zero-CLV **~50%**, skew **16.7** (thưa, đuôi nặng) | Không có treatment |
| **Dunnhumby** Complete Journey | External CLV validation | 2,500 households / 102 tuần / 2.6M dòng (US grocery) | zero-CLV **~7%**, skew **2.5** (dày, đuôi nhẹ) | **Coupon redemption KHÔNG phải treatment sạch** |
| **X5 RetailHero** | Main uplift dataset | 200,039 clients / 45.7M dòng purchase; test 60,012 | RCT: treatment ratio **0.500**; raw uplift **+3.32pp** CI [2.90, 3.75] | Không phải dataset CLV chính |

**Quyết định causal quan trọng (Human Delta):** Coupon redemption của Dunnhumby là *post-campaign outcome* (selection bias — người redeem vốn sẵn muốn mua), **không phải** treatment assignment ngẫu nhiên → loại khỏi uplift. Uplift chỉ ước lượng trên X5 (RCT thật). Dunnhumby chỉ dùng để validate CLV.

**Hai context CLV đối nghịch** (làm RQ1 thành robustness test thật):
- E-commerce (OR-II): zero-inflation nặng ⇒ **incidence-dominated** → Hurdle Stage-1 (“có quay lại không?”) tạo lợi thế.
- Grocery (Dunnhumby): hầu như ai cũng quay lại ⇒ **magnitude/sequence-dominated** → Stage-1 gần trivial, quan trọng là “mua bao nhiêu”.

---

## 3. INPUT / OUTPUT TỪNG BÀI TOÁN

| Bài toán | Input | Output |
|----------|-------|--------|
| **CLV ranking** | Feature từ cửa sổ quá khứ (RFM + behavioral + sequence + semantic) | Predicted future 6-month CLV → **xếp hạng khách (VIP)** |
| **Uplift (RQ3)** | Feature pre-communication của X5 (RFM/basket/points/demographics) | **Uplift score τ̂** mỗi khách (mức tác động của campaign) |
| **Value-adjusted policy (RQ4)** | Uplift score τ̂ + value proxy V̂ (historical monetary) | **Danh sách target top-K** + **profit ước lượng** |

**Nhãn (label):**
- CLV: `ActualCLV` = tổng revenue trong horizon dự báo (6 tháng).
- Uplift: `target` (X5) = nhãn nhị phân purchase-response do RetailHero benchmark cung cấp.
- Value proxy V̂ = historical monetary spend (pre-period) — **là proxy, KHÔNG phải future CLV thật** (ghi rõ trong paper).

---

## 4. SQL (DuckDB — in-process, chạy trên CSV/Parquet)

**Vì sao DuckDB:** nhẹ, không cần server, chạy tốt trên Colab/Windows, feed thẳng vào Python + Plotly Dash.

**SQL dùng để dựng bảng phân tích minh bạch TRƯỚC khi modeling:**
- Clean & validate: `Quantity > 0` & `Price > 0`, bỏ missing `CustomerID`, loại cancellations (`InvoiceNo` bắt đầu `C`) / duplicates.
- Bảng RFM per customer + value proxy V̂ (pre-period spend) mỗi dataset.
- Walk-forward observation/prediction windows (past-only vs future-period).
- Cohort treatment/control của X5 (align communication flag với conversion).
- Product/category aggregation; bảng output cho dashboard.

**3 truy vấn tiêu biểu (link tới RQ):**

| Query → RQ | SQL (rút gọn) | Insight |
|------------|---------------|---------|
| **Q1** · RFM + value proxy → RQ1/RQ4 | `SELECT CustomerID, MAX(date), COUNT(*), SUM(spend) FROM tx GROUP BY CustomerID` | Recency/Frequency/Monetary + V̂ per khách |
| **Q2** · Walk-forward windows → RQ1/RQ2 | `WITH past AS (…< cutoff), fut AS (…>= cutoff) SELECT … JOIN` | Feature quá khứ vs nhãn CLV tương lai, không leakage |
| **Q3** · Treatment cohorts → RQ3/RQ4 | `SELECT client, treatment_flag, MAX(converted) FROM x5 GROUP BY client` | Treated vs control để ước lượng uplift |

---

## 5. PIPELINE PYTHON (config-driven, cô lập trong `phase2/`)

**Guard cô lập:** `assert_phase2_path()` trong `src/config.py` từ chối mọi ghi ra ngoài `phase2/`; Phase 1 chỉ được ĐỌC.

**5 bước chính (script `p2_01` → `p2_19`):**
1. **Clean & Build tables** — loaders + SQL feature tables (RFM, V̂, walk-forward windows) cho cả 3 dataset.
2. **CLV ranking (RQ1–RQ2)** — `clv_walkforward` · `hurdle_model` · `loo_ablation`.
3. **Uplift learners (RQ3)** — `uplift_models` (S/T/X-Learner, ClassTransform vs Response).
4. **Value-adjusted policy (RQ4)** — `policy_comparison` (τ̂·V̂ vs value/uplift/RFM/random) + profit sweep.
5. **Explain + Calibrate** — `shap_hurdle` · `conformal_cqr` + bootstrap robustness.

**Thư viện:** pandas, numpy, scikit-learn, **LightGBM**, XGBoost, **scikit-uplift (sklift)**, shap, matplotlib, duckdb.

---

## 6. FEATURES (4 NHÓM)

| Nhóm | Feature | Ý nghĩa |
|------|---------|---------|
| **RFM** | Recency, Frequency, Monetary, AvgOrderValue, M_per_T | Hành vi mua cơ bản |
| **Behavioral** | ActiveMonths, spend_per_active_day, regularity… | Cường độ & đều đặn |
| **Sequence** | recent-purchase order, seq_spend_recent_half, sequence trend (18 tháng) | Thói quen mua theo thời gian (mạnh ở grocery) |
| **Semantic** | PPMI co-occurrence → Truncated SVD **32-d** product embeddings → revenue-weighted taste vector (full + 90 ngày gần) | Sở thích/“gu” sản phẩm (interpretation) |

- **Value proxy V̂** = historical monetary spend (dùng cho value-adjusted policy).
- Semantic = **PPMI + SVD**: xây ma trận đồng xuất hiện sản phẩm theo basket → PMI dương → SVD giảm chiều 32-d → vector taste của khách = trung bình embedding có trọng số revenue.

---

## 7. MODELS

### 7.1 CLV models (so ≥ 5 model)
Mean · Monetary · RFM_score · Ridge · RandomForest · XGBoost · LightGBM · **Hurdle (proposed)**.

**Hurdle two-stage** (vì CLV zero-inflated + heavy-tailed):
$$\widehat{CLV}_i = p_i \cdot m_i,\quad p_i = P(Y_i>0\mid X_i),\quad m_i = \mathbb{E}[Y_i\mid Y_i>0, X_i]$$
- Stage 1: gradient-boosted classifier ước lượng **xác suất quay lại** p (LightGBM 400 trees, lr 0.03).
- Stage 2: gradient-boosted regressor trên `log Y` (khách dương) + **hiệu chỉnh lognormal** `m = exp(μ̂ + σ̂²/2)` (LightGBM 500 trees, lr 0.03).

### 7.2 Uplift models (RQ3) — base learner = LightGBM (300 trees, lr 0.05, seed 42)
| Model | Cách làm |
|-------|----------|
| **Random** | Baseline ngẫu nhiên |
| **ResponseModel** | Rank theo P(target), **bỏ qua treatment** (chọn “ai dễ mua”) |
| **S-Learner** | 1 model, treatment là feature (sklift SoloModel) |
| **T-Learner** | 2 model treated/control riêng (sklift TwoModels) |
| **X-Learner** | Meta-learner 2 tầng (tự cài, LightGBM regressors, propensity 0.5) |
| **ClassTransformation** | Nhãn biến đổi `z = Y·T + (1−Y)(1−T)`, propensity ½ → `τ̂ = 2·P(z=1) − 1` |

### 7.3 Targeting policies (RQ4)
Random · RFM-only · Value-only (rank theo V̂) · Uplift-only (rank theo τ̂) · **Value-adjusted** (rank theo **τ̂ · V̂**).

**Đánh giá profit (held-out, KHÔNG circular):** score `τ̂·V̂` **chỉ để chọn** top-K set A_K; profit đo bằng chênh lệch **treated − control THỰC TẾ** trong A_K:
$$\widehat{\text{Profit}}@K = m\cdot\widehat{\Delta V}_K - c\,|A_K|,\quad \widehat{\Delta V}_K = |A_K|\big(\overline{Y_iV̂_i}\big|_{T=1,A_K} - \overline{Y_iV̂_i}\big|_{T=0,A_K}\big)$$

---

## 8. METRICS (ĐO CÁI GÌ, VÌ SAO)

| Nhóm | Metric | Vì sao dùng |
|------|--------|-------------|
| **CLV ranking** | **Normalized Gini (NG)**, **Revenue Capture@10% (RC@10)**, Precision@10, Spearman, decile calibration | Mục tiêu là **xếp hạng VIP**, không phải sai số dollar → NG/RC@10 quan trọng hơn MAE/R² trên nhãn zero-inflated |
| **Uplift** | **Qini AUC**, **AUUC**, uplift@K | Đo chất lượng ranking *incremental* (treated−control), không phải response rate |
| **Policy** | incremental conversions, incremental value, **profit@K**, overlap | Đánh giá giá trị kinh doanh thật của chính sách |
| **Uncertainty** | CQR empirical coverage vs nominal, interval width | Interval đáng tin cho risk-aware targeting |
| **Significance** | Walk-forward paired t-test (CLV), bootstrap CI (uplift/profit) | Phân biệt claim vững vs directional |

> **Điểm cốt lõi:** metric khác nhau cho câu hỏi kinh doanh khác nhau — ranking ≠ incremental targeting.

---

## 9. EVALUATION PROTOCOL

- **CLV = walk-forward (temporal)** — bài toán forecasting, random split sẽ **rò rỉ tương lai**. OR-II **tái dùng 5 cửa sổ Phase 1** (benchmark W1–W3); Dunnhumby xây D1–D4 (obs→pred: 26→8, 39→13, 52→13, 78→13 tuần). Báo **mean ± std qua windows**. Trong mỗi window: stratified 80/20 theo VIP top-decile (seed 42).
- **Uplift = X5 RCT** — split 70/30 stratified theo (treatment × outcome). Vì treatment randomized → mọi top-K chứa cả treated & control ⇒ group-mean difference là ước lượng incremental low-bias.
- **Uncertainty** — percentile bootstrap **B=1000 (CLV), B=400 (uplift)**, báo paired difference + 95% CI.
- **Implementation details:** LightGBM cho mọi GBM stage; SHAP = TreeExplainer trên **test fold**; CQR = split-conformal (train/test 80/20, trong train tách proper-train/calibration 50/50, quantile LightGBM α/2 & 1−α/2 + q̂ conformal).

---

## 10. KẾT QUẢ CHÍNH (RQ1–RQ4) — HONEST

### RQ1 — CLV cross-context: *competitive, không dominant; cơ chế đổi theo context*
| Context | Model tốt nhất (NG) | Hurdle |
|---------|---------------------|--------|
| Online Retail II (W1–W3) | **Hurdle 0.825** (top) — nhưng ≈ Monetary 0.822 (paired **p=0.55**, không khác biệt) | RC@10 = 59.0% |
| Dunnhumby (D1–D4) | RandomForest **0.862** > Hurdle 0.855 (chênh không significant) | NG ≈ 0.855 |

### RQ2 — Feature ablation (LOO, walk-forward)
- Grocery: bỏ **sequence** làm giảm mạnh (ΔNG **+0.0206**, **p=0.001**) → sequence là driver thật.
- **Semantic** không đổi ranking đáng kể ở cả hai (p=0.58 / 0.84) → **interpretation, không phải ranking booster**.
- extra-features > RFM chỉ significant ở grocery (p=0.002), **không** ở e-com (p=0.17).

### RQ3 — Uplift (X5, n_test=60,012)
| Model | Qini AUC | uplift@10% |
|-------|----------|-----------|
| **S-Learner** | **0.0136** | 0.112 |
| X-Learner | 0.0121 | 0.104 |
| T-Learner | 0.0101 | 0.087 |
| ClassTransform | 0.0077 | 0.086 |
| Random | −0.0024 | 0.026 |
| **ResponseModel** | **−0.0071** | 0.004 |

**Phát hiện:** **Response model TỆ HƠN random** — rank “ai dễ mua” chọn trúng sure-buyers → incremental ~0. ⇒ *“ai dễ mua” ≠ “ai bị campaign tác động”*.

### RQ4 — Value-adjusted policy
- Value-only **lỗ ở budget nhỏ** (khách high-value là sure-buyers); Uplift-only tối đa **số** conversions; **Value-adjusted có profit point-estimate cao nhất** (cả **9/9** setting cost×margin).
- **NHƯNG** bootstrap: profit CI **rộng & chồng 0** (Value-adjusted @10% +956K, CI [−3.5M, 5.9M]) → **directional, chưa statistically significant** (báo trung thực, không overclaim).
- **Overlap** top-10% giữa Value và Uplift = **0.007** → chọn 99.3% khách KHÁC nhau ⇒ **value ≠ persuadability** (bằng chứng mạnh nhất cho thesis).

---

## 11. EXPLAINABILITY & UNCERTAINTY

- **SHAP (Stage-2):** e-commerce → **monetary** trội; grocery → **recent-sequence spend** trội. Khớp với ablation + cơ chế cross-context. Semantic xuất hiện top-5 nhưng thứ yếu.
- **CQR (conformal):** empirical coverage **gần nominal** (90% → 90.2% e-com / 89.0% grocery; 80% → 79.3%); interval grocery **hẹp hơn** (đuôi nhẹ). ⇒ interval đáng tin để làm risk-aware targeting.

---

## 12. GAPS / LIMITATIONS (trung thực)

1. **Sample/scope:** Dunnhumby nhỏ (~2,500 households); uplift chỉ từ **1 campaign SMS, 1 market, 1 giai đoạn** → external validity chưa kiểm chứng.
2. **Value proxy:** V̂ = historical monetary, **proxy** cho future value, **không phải actual future CLV**.
3. **Statistical power:** chỉ **3–4 walk-forward windows** → paired-test yếu; profit dollar CI rộng → chỉ directional.
4. **Causal scope:** uplift chỉ identified trên X5 (randomized); **không có post-campaign revenue** (chỉ conversion) → profit là *expected value* theo proxy, không phải observed revenue.
5. **Transfer:** chỉ là *transferable implication* cho budget-constrained retail campaigns; **không claim numerical transfer** sang market cụ thể.

---

## 13. DELIVERABLES & REPRODUCIBILITY

- **Paper:** Springer sn-jnl, **12 trang**, 17 references, 2 figures (SHAP cross-context + profit), 7 bảng; honest reporting (0 first-person, 0 overclaim).
- **Dashboard:** Plotly Dash (5 trang, có trang Phase 2 `/uplift`) + **bản public GitHub Pages** có upload + tương tác: `https://minhanh0709-oss.github.io/ADY201m/phase2.html`.
- **AI Audit Log:** entries có Human Delta (tư duy phản biện) + evidence thật + hallucination detection.
- **Reproducibility:** `requirements.txt` pin version; `reproduce_all.sh` chạy p2_01→p2_11; tables_auto.tex auto-sync số từ CSV; `.gitignore` loại raw gz 670MB.

---

### CÂU CHỐT KHI THUYẾT TRÌNH
> Marketing không nên chỉ target khách **giá trị cao**, mà target khách **vừa giá trị vừa bị campaign tác động** — đó là ý nghĩa của **value-adjusted uplift**.
> *CLV = who is valuable · Uplift = who is persuadable · Value-adjusted uplift = who is worth targeting.*
