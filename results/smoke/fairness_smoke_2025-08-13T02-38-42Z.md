# Fairness Smoke Test

- Timestamp: 2025-08-13T02-38-42Z
- num_preferences: 6

| case     | ranks                 | satisfaction               | gini   |
|----------|-----------------------|----------------------------|--------|
| equalish | [1, 2, 3, 2, 1, 3]    | [6, 5, 4, 5, 6, 4]         | 0.0889 |
| skewed   | [1, 6, 6, 6, 6, 6]    | [6, 1, 1, 1, 1, 1]         | 0.3788 |

**Leximin:** equalish > skewed (fairer)
