import pandas as pd, numpy as np, sys
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('D:/SU26/ADY201m/paper/results/semantic_walkforward_v2.csv')

print('=== V2 RESULTS DETAILED ANALYSIS ===')
print()
print('Per-window NG:')
print('Window  RFM      Seq      SemV2    AllV2    Best')
for w in [1,2,3,4,5]:
    r = df[df.Window==w].set_index('Model')
    best = r['Norm_Gini'].idxmax()
    print(f'  W{w}   {r.loc["Hurdle-RFM","Norm_Gini"]:.4f}  {r.loc["Hurdle-Seq","Norm_Gini"]:.4f}  '
          f'{r.loc["Hurdle-SemV2","Norm_Gini"]:.4f}  {r.loc["Hurdle-AllV2","Norm_Gini"]:.4f}  {best}')

print()
print('Per-window Stage1 AUC (churn prediction):')
print('Window  RFM      Seq      SemV2    AllV2')
for w in [1,2,3,4,5]:
    r = df[df.Window==w].set_index('Model')
    print(f'  W{w}   {r.loc["Hurdle-RFM","Stage1_AUC"]:.4f}  {r.loc["Hurdle-Seq","Stage1_AUC"]:.4f}  '
          f'{r.loc["Hurdle-SemV2","Stage1_AUC"]:.4f}  {r.loc["Hurdle-AllV2","Stage1_AUC"]:.4f}')

print()
print('=== Paired t-tests (5 windows) ===')
rfm = df[df.Model=='Hurdle-RFM'].sort_values('Window')
sem = df[df.Model=='Hurdle-SemV2'].sort_values('Window')

for metric in ['Norm_Gini','Revenue_Capture_10','Stage1_AUC']:
    t,p = stats.ttest_rel(sem[metric].values, rfm[metric].values)
    print(f'  SemV2 vs RFM ({metric:18s}): t={t:+.3f}, p={p:.4f}')

print()
print('=== Mean across 5 windows ===')
print(f'{"Variant":<14} {"NG":>9} {"RC10":>10} {"S1_AUC":>10} {"MAE":>7}')
for v in ['Hurdle-RFM','Hurdle-Seq','Hurdle-SemV2','Hurdle-AllV2']:
    sub = df[df.Model==v]
    print(f'{v:<14} {sub.Norm_Gini.mean():>9.4f} {sub.Revenue_Capture_10.mean():>9.2f}%  '
          f'{sub.Stage1_AUC.mean():>9.4f}  ${sub.MAE.mean():>5.0f}')

print()
print('=== V1 vs V2 comparison (Sem variant only) ===')
v1 = pd.read_csv('D:/SU26/ADY201m/paper/results/semantic_walkforward.csv')
v1_sem = v1[v1.Model=='Hurdle-Sem'].sort_values('Window')
v2_sem = df[df.Model=='Hurdle-SemV2'].sort_values('Window')
print(f'  V1 Sem  (PCA-16):               NG={v1_sem.Norm_Gini.mean():.4f}, RC10={v1_sem.Revenue_Capture_10.mean():.2f}%')
print(f'  V2 SemV2 (sup-8 + recency):     NG={v2_sem.Norm_Gini.mean():.4f}, RC10={v2_sem.Revenue_Capture_10.mean():.2f}%')
print(f'  Delta:                          +{v2_sem.Norm_Gini.mean()-v1_sem.Norm_Gini.mean():.4f} NG')

t_v1v2,p_v1v2 = stats.ttest_rel(v2_sem.Norm_Gini.values, v1_sem.Norm_Gini.values)
print(f'  Paired t-test V2 vs V1: t={t_v1v2:+.3f}, p={p_v1v2:.4f}')

# Per-window delta
print()
print('=== V1 -> V2 per-window NG delta ===')
for w in [1,2,3,4,5]:
    v1_w = v1_sem[v1_sem.Window==w].Norm_Gini.values[0]
    v2_w = v2_sem[v2_sem.Window==w].Norm_Gini.values[0]
    print(f'  W{w}: V1={v1_w:.4f} -> V2={v2_w:.4f}  delta={v2_w-v1_w:+.4f}')

# Top dim selection robustness — check SHAP
print()
print('=== SHAP for AllV2 (W3) ===')
shap = pd.read_csv('D:/SU26/ADY201m/paper/results/shap_hurdle_allv2_stage2.csv')
print(shap.head(20).to_string())

# Count semantic features in top 15
sem_in_top = sum(1 for f in shap.head(15).feature if 'sem_' in str(f))
print(f'\nSemantic features in top 15: {sem_in_top}')
print(f'Total sem SHAP / total SHAP: {shap[shap.feature.str.contains("sem_", na=False)].mean_abs_shap.sum()/shap.mean_abs_shap.sum()*100:.1f}%')
