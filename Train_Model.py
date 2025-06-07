
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

pd.set_option('display.float_format', '{:.4f}'.format)

base_metabolite_file = "Metabolite_Data_2022.xlsx"
base_physiological_file = "Physiological_data_2022-2023.xlsx"

uploaded_dir = "uploaded_files"
os.makedirs(uploaded_dir, exist_ok=True)

used_files = [
    os.path.join(uploaded_dir, f) for f in os.listdir(uploaded_dir)
    if f.startswith("trained__") and f.endswith(".xlsx")
]

print("\n📂 Training will use these additional files:")
for f in used_files:
    print(f" - {f}")

def detect_type(df):
    lower_cols = [c.lower() for c in df.columns]
    if any("chlorophyll" in c for c in lower_cols) or any("stomatal" in c for c in lower_cols):
        return "physiological"
    if any("malic" in c or "tartaric" in c or "glucose" in c for c in lower_cols):
        return "metabolite"
    return "unknown"

def load_all_data(base_file, extra_files, feature_type):
    dfs = [pd.read_excel(base_file)]
    for file in extra_files:
        try:
            df = pd.read_excel(file)
            if detect_type(df) == feature_type:
                dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
    return pd.concat(dfs, ignore_index=True)

metabolite_df = load_all_data(base_metabolite_file, used_files, "metabolite")
physiological_df = load_all_data(base_physiological_file, used_files, "physiological")

non_feature_cols = ['Date', 'Time', 'Sample', 'Variety', 'Plot', 'obs', 'ID', 'Temperature (°C)', 'Avg_temp_72h (°C)', 'Max_temp_24h (°C)']
metabolite_features = [col for col in metabolite_df.columns if col not in non_feature_cols]
physiological_features = [col for col in physiological_df.columns if col not in non_feature_cols]

print("🔬 Metabolite features:", metabolite_features)
print("🌿 Physiological features:", physiological_features)

predictions_met = {}
r2_scores_met = {}
predictions_phy = {}
r2_scores_phy = {}

def train_models(df, features, name, store_dict, score_dict):
    print(f"\n--- TRAINING {name.upper()} MODELS ---")
    df = df[['Temperature (°C)', 'Variety'] + features].dropna(subset=['Temperature (°C)', 'Variety'])

    for variety in df['Variety'].unique():
        variety_df = df[df['Variety'] == variety]
        print(f"\n-- Variety: {variety} --")
        for feature in features:
            data_filtered = variety_df[['Temperature (°C)', feature]].dropna()
            if data_filtered.empty or len(data_filtered) < 3:
                continue
            X = data_filtered[['Temperature (°C)']]
            y = data_filtered[feature].reset_index(drop=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if len(X_train) == 0:
                continue

            models = {
                'XGBoost': XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression()
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
                r2 = r2_score(y_test, y_pred_test)
                mean_actual = y_test.mean()
                mae_percent = (mae / mean_actual) * 100 if mean_actual != 0 else float('nan')
                results[name] = (mae, rmse, r2, mae_percent, model)

            best_model_name = max(results, key=lambda k: results[k][2])
            best_mae, best_rmse, best_r2, best_mae_percent, best_model = results[best_model_name]

            print(f"{feature} - BEST: {best_model_name} - R²: {best_r2:.4f}, MAE%: {best_mae_percent:.2f}%")

            store_dict[(variety, feature)] = best_model
            score_dict[(variety, feature)] = best_r2

train_models(metabolite_df, metabolite_features, "metabolite", predictions_met, r2_scores_met)
train_models(physiological_df, physiological_features, "physiological", predictions_phy, r2_scores_phy)
