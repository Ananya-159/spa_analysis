# EDA Summary (SPA Dataset — Raw Only)
```
Dataset Summary:

- Total Students: 102
- Total Projects: 37
- Total Supervisors: 21
- Mean GPA: 67.96
- Client Eligible Students: 49.0%
- Preference overlap (top 3): 100.0% of projects
- Project Min Capacity: 1
- Project Max Capacity: 5
- Supervisor Max Student Capacity: 6
- Supervisor Min Student Capacity: 1
```

## Descriptive Statistics (GPA & Courses)
|       |   Average |   Course 1 |   Course 2 |   Course 3 |
|:------|----------:|-----------:|-----------:|-----------:|
| count |    102    |     102    |     102    |     102    |
| mean  |     67.96 |      68.25 |      67.67 |      67.97 |
| std   |      6.34 |      10.96 |      10.56 |      10.41 |
| min   |     54    |      50    |      50    |      50    |
| 25%   |     64    |      57.25 |      58    |      58.5  |
| 50%   |     68    |      69    |      70    |      68.5  |
| 75%   |     72    |      78    |      75    |      77    |
| max   |     82    |      84    |      84    |      84    |

## Top 5 Most In‑Demand Projects by Top‑3 Preferences
| Project ID   |   Top-3 Preference Count | Project Title   |
|:-------------|-------------------------:|:----------------|
| P020         |                       16 | Project 20      |
| P018         |                       13 | Project 18      |
| P030         |                       13 | Project 30      |
| P006         |                       12 | Project 6       |
| P029         |                       11 | Project 29      |

## Variable Overview (Students)
| Variable                 | Type   |   Missing |   Uniques |           Min |           Max |          Mean |      SD |
|:-------------------------|:-------|----------:|----------:|--------------:|--------------:|--------------:|--------:|
| Student ID               | int64  |         0 |       102 |   9.65351e+06 |   9.65361e+06 |   9.65356e+06 |  29.589 |
| Student Name             | object |         0 |       102 | nan           | nan           | nan           | nan     |
| Course 1                 | int64  |         0 |        34 |  50           |  84           |  68.245       |  10.964 |
| Course 2                 | int64  |         0 |        33 |  50           |  84           |  67.667       |  10.555 |
| Course 3                 | int64  |         0 |        33 |  50           |  84           |  67.971       |  10.414 |
| Average                  | int64  |         0 |        28 |  54           |  82           |  67.961       |   6.335 |
| Client Based Eligibility | bool   |         0 |         2 |   0           |   1           |   0.49        |   0.502 |
| Preference 1             | object |         0 |        35 | nan           | nan           | nan           | nan     |
| Preference 2             | object |         0 |        34 | nan           | nan           | nan           | nan     |
| Preference 3             | object |         0 |        35 | nan           | nan           | nan           | nan     |
| Preference 4             | object |         0 |        34 | nan           | nan           | nan           | nan     |
| Preference 5             | object |         0 |        34 | nan           | nan           | nan           | nan     |
| Preference 6             | object |         0 |        33 | nan           | nan           | nan           | nan     |

## Outlier Summary
```
Outlier Summary (|z| > 3):
- Average: 0
- Course 1: 0
- Course 2: 0
- Course 3: 0
```

## Statistical Tests (Raw)
### Normality (Shapiro) & Homogeneity (Levene)
- Shapiro–Wilk (Eligible): W=0.976, p=0.3935
- Shapiro–Wilk (Not Eligible): W=0.983, p=0.6703
- Levene’s test (equal variances): W=0.226, p=0.6354 → fail to reject at α=0.05
### Welch’s t-test (Average GPA by Client Eligibility)
- Group 1 (Eligible): n=50, mean=68.12 (95% CI 66.36–69.88), sd=6.18
- Group 2 (Not Eligible): n=52, mean=67.81 (95% CI 65.99–69.63), sd=6.54
- t=0.248, df≈100.0, p=0.8046, Cohen’s d=0.049
- Interpretation: Difference is not significant at α=0.05
### Mann–Whitney U (Average GPA by Client Eligibility)
- Group sizes: Eligible n=50, Not Eligible n=52
- U=1321.000, p=0.8907, rank‑biserial r=-0.016
- Interpretation: Difference is not significant at α=0.05
### ANOVA (Average GPA by First Preference Project Type)
- Groups: Client based, Student sourced, Research based
- F=0.271, p=0.7632, η²=0.005
- Interpretation: Difference is not significant at α=0.05
### Kruskal–Wallis (Average GPA by First Preference Project Type)
- Groups: Client based, Student sourced, Research based
- H=0.761, p=0.6834
- Interpretation: not significant at α=0.05
### Chi-squared (Pref-count bucket × Client Eligibility)
- Skipped: contingency table has only one row/column (df=0) — all students listed the same pref count.
- Observed table:
Client Based Eligibility  False  True 
Pref Bucket                           
4–6                          52     50

## Correlation (Pearson) p‑values (upper triangle)
|          |   Average |   Course 1 |   Course 2 |   Course 3 |
|:---------|----------:|-----------:|-----------:|-----------:|
| Average  |       nan |          0 |     0      |     0      |
| Course 1 |       nan |        nan |     0.3937 |     0.0648 |
| Course 2 |       nan |        nan |   nan      |     0.8673 |
| Course 3 |       nan |        nan |   nan      |   nan      |
