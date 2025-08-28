# Statistical Comparison — Hill Climbing vs Genetic Algorithm vs Selector Hill Climbing

**Timestamp:** 2025-08-28 20:03:07

**Sources:**
- `C:\Users\anany\Desktop\spa_analysis\results\Hill_climbing_runs\hc_summary_log.csv`
- `C:\Users\anany\Desktop\spa_analysis\results\Genetic_algorithm_runs\ga_summary_log.csv`
- `C:\Users\anany\Desktop\spa_analysis\results\selector_hill_climbing_runs\selectorhc_summary_log.csv`

**Multiple Testing:** BH/FDR on 7 KW p-values; BH/FDR on 21 pairwise p-values


## Fitness Score (lower = better)
Hill Climbing: n=30, Genetic Algorithm: n=30, Selector Hill Climbing: n=30
- Hill Climbing: median=13326.5  IQR=[12359.75–14326]  95% CI=[12382–13433]
- Genetic Algorithm: median=21167  IQR=[20661.5–21472]  95% CI=[20852–21417.5]
- Selector Hill Climbing: median=7304  IQR=[6701.5–7667]  95% CI=[6998.5–7475]
KW: H=79.122, p=6.591e-18, q=4.614e-17, ε²=0.886

## Total Violations (lower = better)
Hill Climbing: n=30, Genetic Algorithm: n=30, Selector Hill Climbing: n=30
- Hill Climbing: median=13.5  IQR=[12.25–14.75]  95% CI=[13–14]
- Genetic Algorithm: median=75  IQR=[73–76]  95% CI=[73–76]
- Selector Hill Climbing: median=23  IQR=[21–24]  95% CI=[22–23.5]
KW: H=78.779, p=7.822e-18, q=2.738e-17, ε²=0.883

## Average Assigned Preference Rank (lower = better)
Hill Climbing: n=30, Genetic Algorithm: n=30, Selector Hill Climbing: n=30
- Hill Climbing: median=3.46  IQR=[3.17–3.69]  95% CI=[3.31–3.61]
- Genetic Algorithm: median=3.56  IQR=[3.43–3.66]  95% CI=[3.5–3.62]
- Selector Hill Climbing: median=4.01  IQR=[3.9–4.09]  95% CI=[3.92–4.06]
KW: H=57.347, p=3.525e-13, q=2.738e-17, ε²=0.636

## Fairness — Gini Satisfaction (lower = fairer)
Hill Climbing: n=30, Genetic Algorithm: n=30, Selector Hill Climbing: n=30
- Hill Climbing: median=0.31  IQR=[0.28–0.33]  95% CI=[0.3–0.32]
- Genetic Algorithm: median=0.28  IQR=[0.27–0.29]  95% CI=[0.28–0.29]
- Selector Hill Climbing: median=0.36  IQR=[0.35–0.36]  95% CI=[0.35–0.36]
KW: H=62.724, p=2.397e-14, q=2.738e-17, ε²=0.698

## Top-1 Preference Match (%) (higher = better)
Hill Climbing: n=30, Genetic Algorithm: n=30, Selector Hill Climbing: n=30
- Hill Climbing: median=22.55  IQR=[19.61–29.41]  95% CI=[19.61–28.92]
- Genetic Algorithm: median=15.69  IQR=[12.75–18.63]  95% CI=[13.73–18.14]
- Selector Hill Climbing: median=14.71  IQR=[12.75–18.38]  95% CI=[13.73–16.67]
KW: H=34.887, p=2.657e-08, q=2.738e-17, ε²=0.378

## Top-3 Preference Match (%) (higher = better)
Hill Climbing: n=30, Genetic Algorithm: n=30, Selector Hill Climbing: n=30
- Hill Climbing: median=52.45  IQR=[49.02–58.33]  95% CI=[50.49–55.88]
- Genetic Algorithm: median=48.53  IQR=[44.12–50.98]  95% CI=[45.59–50.49]
- Selector Hill Climbing: median=41.67  IQR=[39.22–44.12]  95% CI=[40.2–43.63]
KW: H=49.605, p=1.692e-11, q=2.738e-17, ε²=0.547

## Runtime (seconds, lower = better)
Hill Climbing: n=30, Genetic Algorithm: n=30, Selector Hill Climbing: n=30
- Hill Climbing: median=44.89  IQR=[41.88–56.42]  95% CI=[43–51.34]
- Genetic Algorithm: median=79.61  IQR=[78.97–85.89]  95% CI=[79.02–84.65]
- Selector Hill Climbing: median=47.79  IQR=[45.97–65.19]  95% CI=[46.46–62.05]
KW: H=32.801, p=7.539e-08, q=2.738e-17, ε²=0.354

[CSV] Saved: C:\Users\anany\Desktop\spa_analysis\results\comparison.csv

[ALLOC] Loaded allocation records: 9282
[PLOT] pref_match_distribution.png

## Preference Match Distribution — Chi-square Goodness-of-Fit (per algorithm)
- Hill Climbing: χ²=598.90, p=3.499e-127 (n=3060, ranks=6)
- Genetic Algorithm: χ²=6.24, p=0.2833 (n=3060, ranks=6)
- Selector Hill Climbing: χ²=1286.68, p=4.902e-276 (n=3162, ranks=6)
[PLOT] avg_vs_project.png

## Average vs Project — Kruskal–Wallis across projects (per algorithm)
- Hill Climbing: H=1047.57, p=1.627e-196 (projects=37)
- Genetic Algorithm: H=184.18, p=8.592e-22 (projects=37)
- Selector Hill Climbing: H=966.16, p=1.967e-179 (projects=37)
[PLOT] supervisor_loads_top12.png

## Supervisor Fairness — Gini(load) per algorithm (mean ± SD across runs)
- Hill Climbing: 0.178 ± 0.016  (runs=30)
- Genetic Algorithm: 0.240 ± 0.036  (runs=30)
- Selector Hill Climbing: 0.203 ± 0.019  (runs=31)

## Supervisor Satisfaction Variation — Kruskal–Wallis across supervisors (per algorithm)
- Hill Climbing: H=502.62, p=8.24e-94 (supervisors=21)
- Genetic Algorithm: H=131.57, p=1.974e-18 (supervisors=21)
- Selector Hill Climbing: H=244.83, p=1.258e-40 (supervisors=21)

## Supervisor Load Variance Test (Levene’s)
- Hill Climbing: std across runs (n=30): 1.68 ± 0.14
- Genetic Algorithm: std across runs (n=30): 2.18 ± 0.29
- Selector Hill Climbing: std across runs (n=31): 1.84 ± 0.15
Levene’s Test: stat=3.973, p=0.02229

## Supervisor Load Comparison (KW on supervisor means)
- Hill Climbing: mean per supervisor per run = 4.86 ± 0.00 (n=30)
- Genetic Algorithm: mean per supervisor per run = 4.93 ± 0.11 (n=30)
- Selector Hill Climbing: mean per supervisor per run = 4.86 ± 0.04 (n=31)
KW Test: H=16.515, p=0.0002593
