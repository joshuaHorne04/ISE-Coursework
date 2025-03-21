import pandas as pd
from scipy.stats import shapiro
import numpy as np

# -----------------------------
# Load the Results CSVs
# -----------------------------
baseline_results_file = 'baseline_results.csv'
rf_results_file = 'rf_results.csv'

baseline_df = pd.read_csv(baseline_results_file)
rf_df = pd.read_csv(rf_results_file)

# -----------------------------
# Merge Data on Dataset Name
# -----------------------------
merged_df = pd.merge(baseline_df, rf_df, on='Dataset', suffixes=('_baseline', '_rf'))

# -----------------------------
# Function to Run Shapiro-Wilk Test
# -----------------------------
def normality_test(metric_name):
    print(f"\n=== Shapiro-Wilk Test for {metric_name.upper()} Differences ===")

    # Compute the differences between Random Forest and Linear Regression
    diff = merged_df[f'{metric_name}_rf'] - merged_df[f'{metric_name}_baseline']

    # Run Shapiro-Wilk test
    stat, p_value = shapiro(diff)

    # Print results with higher precision (10 decimal places)
    print(f'Shapiro-Wilk test statistic: {stat:.10f}')
    print(f'p-value: {p_value:.10f}')

    # Interpret the result
    alpha = 0.05
    if p_value > alpha:
        print(f'✅ The differences are normally distributed (p > {alpha})')
        print(f'➡️ You could use a Paired t-test for {metric_name}.\n')
    else:
        print(f'❌ The differences are NOT normally distributed (p <= {alpha})')
        print(f'➡️ A non-parametric test like Wilcoxon is more appropriate for {metric_name}.\n')

# -----------------------------
# Run Normality Test on All Metrics
# -----------------------------
for metric in ['MAPE', 'MAE', 'RMSE']:
    normality_test(metric)

# -----------------------------
# Optional: Preview Merged Results
# -----------------------------
print("\n=== Merged Results Preview ===")
print(merged_df[['Dataset', 'MAPE_baseline', 'MAPE_rf']].head())
