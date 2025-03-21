import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def cliffs_delta(lst1, lst2):
    n1, n2 = len(lst1), len(lst2)
    greater = less = 0
    for x in lst1:
        for y in lst2:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    delta = (greater - less) / (n1 * n2)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        size = "negligible"
    elif abs_delta < 0.33:
        size = "small"
    elif abs_delta < 0.474:
        size = "medium"
    else:
        size = "large"
    return delta, size

def analyse_metric(df, metric):
    print(f"\n{metric.upper()} Analysis")

    lr = pd.to_numeric(df[f'{metric}_baseline'], errors='coerce').values
    rf = pd.to_numeric(df[f'{metric}_rf'], errors='coerce').values

    # Remove pairs where either value is NaN
    mask = ~np.isnan(lr) & ~np.isnan(rf)
    lr_clean = lr[mask]
    rf_clean = rf[mask]

    if len(lr_clean) == 0 or len(rf_clean) == 0:
        print("No valid data to compare.")
        return

    if np.allclose(rf_clean, lr_clean):
        print('Values are identical, statistical test not applicable.')
    else:
        stat, p = wilcoxon(rf_clean, lr_clean)
        print(f'Wilcoxon statistic: {stat:.10f}')
        print(f'p-value: {p:.10f}')
        if p < 0.05:
            print('There is a significant difference.')
        else:
            print('No significant difference found.')

    delta, size = cliffs_delta(rf_clean.tolist(), lr_clean.tolist())
    print(f"Cliff's Delta: {delta:.10f} ({size} effect)")

baseline = pd.read_csv('baseline_results.csv')
rf = pd.read_csv('rf_results.csv')

for df in [baseline, rf]:
    for col in ['MAPE', 'MAE', 'RMSE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

merged = pd.merge(baseline, rf, on='Dataset', suffixes=('_baseline', '_rf'))

for metric in ['MAPE', 'MAE', 'RMSE']:
    analyse_metric(merged, metric)

print("\nMerged results (first few rows):")
print(merged[['Dataset', 'MAPE_baseline', 'MAPE_rf', 'MAE_baseline', 'MAE_rf', 'RMSE_baseline', 'RMSE_rf']].head())
