import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from scipy.stats import boxcox

# --------- Feature Processing Functions ---------
def remove_low_importance_features(X, y, keep_ratio=0.75):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    feature_importances = model.feature_importances_

    sorted_indices = np.argsort(feature_importances)[::-1]
    keep_count = max(1, int(len(sorted_indices) * keep_ratio))

    important_indices = sorted_indices[:keep_count]
    X_filtered = X[:, important_indices]
    return X_filtered

def remove_high_vif_features(X):
    vif_threshold = 7.0
    while True:
        vif_data = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        high_vif_indices = [i for i, vif in enumerate(vif_data) if vif > vif_threshold]

        if not high_vif_indices:
            break

        ridge_model = RandomForestRegressor(n_estimators=10, random_state=42)
        ridge_model.fit(X, np.random.randn(X.shape[0]))
        feature_weights = np.abs(ridge_model.feature_importances_)
        weakest_feature = high_vif_indices[np.argmin(feature_weights[high_vif_indices])]
        X = np.delete(X, weakest_feature, axis=1)

    return X

def clean_features(df, variance_threshold=0.01, corr_threshold=0.95):
    df = df.loc[:, ~df.columns.duplicated()]
    constant_columns = [col for col in df.columns[:-1] if df[col].nunique() == 1]
    if constant_columns:
        df.drop(columns=constant_columns, inplace=True)

    selector = VarianceThreshold(threshold=variance_threshold)
    df_var_filtered = selector.fit_transform(df.iloc[:, :-1])
    df = pd.DataFrame(df_var_filtered, columns=df.columns[:-1][selector.get_support()])

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)

    return df

# --------- Random Forest Runner ---------
def run_random_forest():
    dataset_folder = 'ISE/lab2/datasets'
    csv_files = glob.glob(os.path.join(dataset_folder, '**', '*.csv'), recursive=True)
    rf_results = []

    for file in sorted(csv_files):
        dataset_name = os.path.basename(file)
        print(f"\n📊 Processing {dataset_name}...")

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"⚠ Skipping {dataset_name} due to error: {e}")
            continue

        target_column = df.columns[-1]
        y = df.iloc[:, -1].values
        df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')

        y, lam = boxcox(y + 1)

        categorical_columns = [col for col in df.columns[:-1] if df[col].dtype == 'object']
        numerical_columns = [col for col in df.columns[:-1] if df[col].dtype != 'object']

        df = clean_features(df)

        if categorical_columns:
            encoder = LabelEncoder()
            for col in categorical_columns:
                df[col] = encoder.fit_transform(df[col].astype(str))

        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(df[numerical_columns])

        X = scaled_numerical

        if X.shape[1] > 1:
            X = remove_high_vif_features(X)
        if X.shape[1] > 1:
            X = remove_low_importance_features(X, y, keep_ratio=0.75)

        pca = PCA(n_components=min(10, X.shape[1]))
        X = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f'MAPE: {mape:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}')

        rf_results.append({
            'Dataset': dataset_name,
            'MAPE': mape,
            'MAE': mae,
            'RMSE': rmse
        })

    rf_df = pd.DataFrame(rf_results)
    rf_df.to_csv('rf_results.csv', index=False)
    print("\n✅ Random Forest results saved to 'rf_results.csv'")

# Run the Random Forest function
run_random_forest()
