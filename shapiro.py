import pandas as pd
from scipy.stats import shapiro
import numpy as np

baseline_results_file = 'baseline_results.csv'
rf_results_file = 'rf_results.csv'

baseline_df = pd.read_csv(baseline_results_file)
rf_df = pd.read_csv(rf_results_file)


merged_df = pd.merge(baseline_df, rf_df, on='Dataset', suffixes=('_baseline', '_rf'))

def normality_test(metric_name):
    print(f"\n=== Shapiro-Wilk Test for {metric_name.upper()} Differences ===")


    diff = merged_df[f'{metric_name}_rf'] - merged_df[f'{metric_name}_baseline']


    stat, p_value = shapiro(diff)

    print(f'Shapiro-Wilk test statistic: {stat:.10f}')
    print(f'p-value: {p_value:.10f}')


    alpha = 0.05
    if p_value > alpha:
        print(f'The differences are normally distributed (p > {alpha})')

    else:
        print(f'The differences are NOT normally distributed (p <= {alpha})')


for metric in ['MAPE', 'MAE', 'RMSE']:
    normality_test(metric)

print("\n=== Merged Results Preview ===")
print(merged_df[['Dataset', 'MAPE_baseline', 'MAPE_rf']].head())
