import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np

def run_linear_regression_baseline():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3
    train_frac = 0.7
    random_seed = 1

    baseline_results = []

    for current_system in systems:
        datasets_location = f'ISE/lab2/datasets/{current_system}'

        csv_files = sorted([f for f in os.listdir(datasets_location) if f.endswith('.csv')])

        for csv_file in csv_files:
            

            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                model = LinearRegression()

                model.fit(training_X, training_Y)

                predictions = model.predict(testing_X)

                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])

            print(f'Average MAPE: {avg_mape:.4f}')
            print(f'Average MAE: {avg_mae:.4f}')
            print(f'Average RMSE: {avg_rmse:.4f}')

            baseline_results.append({
                'System': current_system,
                'Dataset': csv_file,
                'MAPE': avg_mape,
                'MAE': avg_mae,
                'RMSE': avg_rmse
            })

    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv('baseline_results.csv', index=False)
    print("\nBaseline results saved to 'baseline_results.csv'")

run_linear_regression_baseline()
