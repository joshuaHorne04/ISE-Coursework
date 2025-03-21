import pandas as pd
from scipy import stats

rf_df = pd.read_csv('rf_results.csv')
for col in ['MAPE', 'MAE', 'RMSE']:
    rf_df[col] = pd.to_numeric(rf_df[col], errors='coerce')

def summarise_metrics(df, label, trim_pct=0.1):
    print(f"{label}\n")
    
    for metric in ['MAPE', 'MAE', 'RMSE']:
        median = df[metric].median()
        trimmed_mean = stats.trim_mean(df[metric], proportiontocut=trim_pct)
        
        print(f"{metric}: Median = {median:.6f}, {int(trim_pct * 100)}% Trimmed Mean = {trimmed_mean:.6f}")
    print()

summarise_metrics(rf_df, "Random Forest Regression", trim_pct=0.1)
