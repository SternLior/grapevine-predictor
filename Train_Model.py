from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import joblib

pd.set_option('display.float_format', '{:.4f}'.format)

def train_and_save_models():
    base_metabolite_file = "Metabolite_Data_2022.xlsx"
    base_physiological_file = "Physiological_data_2022-2023.xlsx"
    uploaded_dir = "uploaded_files"
    os.makedirs(uploaded_dir, exist_ok=True)

    used_files = [
        os.path.join(uploaded_dir, f) for f in os.listdir(uploaded_dir)
        if f.startswith("trained__") and f.endswith(".xlsx")
    ]

    print("\nExtra training files:")
    for f in used_files:
        print(f"  - {f}")

    classified_files = {"metabolite": [], "physiological": [], "unknown": []}
    file_dataframes = {}

    def detect_type(df):
        lower_cols = [c.lower() for c in df.columns]
        metabolite_keywords = ["malic", "tartaric", "glucose", "fructose", "sucrose"]
        physiological_keywords = ["chlorophyll", "stomatal", "conductance", "photosynthesis"]
        if any(any(k in c for k in physiological_keywords) for c in lower_cols):
            return "physiological"
        if any(any(k in c for k in metabolite_keywords) for c in lower_cols):
            return "metabolite"
        return "unknown"

    for fpath in used_files:
        try:
            df = pd.read_excel(fpath)
            dtype = detect_type(df)
            file_dataframes[fpath] = df
            classified_files[dtype].append(fpath)
            print(f"Classified {os.path.basename(fpath)} as {dtype}")
        except Exception as e:
            print(f"Failed to read {fpath}: {e}")
            classified_files["unknown"].append(fpath)

    def load_all_data(base_file, extra_file_paths, feature_type):
        dfs = [pd.read_excel(base_file)]
        for path in extra_file_paths:
            if path in file_dataframes:
                dfs.append(file_dataframes[path])
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined)} rows for {feature_type}")
        return combined

    metabolite_df = load_all_data(base_metabolite_file, classified_files["metabolite"], "metabolite")
    physiological_df = load_all_data(base_physiological_file, classified_files["physiological"], "physiological")

    non_feature_cols = ['Date', 'Time', 'Sample', 'Variety', 'Plot', 'obs', 'ID', 'Temperature (°C)', 'Avg_temp_72h (°C)', 'Max_temp_24h (°C)']
    metabolite_features = [col for col in metabolite_df.columns if col not in non_feature_cols]
    physiological_features = [col for col in physiological_df.columns if col not in non_feature_cols]

    predictions_met, r2_scores_met = {}, {}
    predictions_phy, r2_scores_phy = {}, {}
    model_types_met, mae_scores_met, sample_counts_met = {}, {}, {}
    model_types_phy, mae_scores_phy, sample_counts_phy = {}, {}, {}

    def train_models(df, features, name, store_dict, score_dict):
        print(f"\nTraining models for: {name}")
        df = df[['Temperature (°C)', 'Variety'] + features].dropna(subset=['Temperature (°C)', 'Variety'])

        for variety in df['Variety'].unique():
            variety_df = df[df['Variety'] == variety]
            print(f"  Variety: {variety}")
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
                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred) ** 0.5
                    r2 = r2_score(y_test, y_pred)
                    mean_actual = y_test.mean()
                    mae_percent = (mae / mean_actual) * 100 if mean_actual != 0 else float('nan')
                    results[model_name] = (mae, rmse, r2, mae_percent, model)

                best_model_name = max(results, key=lambda k: results[k][2])
                best_mae, best_rmse, best_r2, best_mae_percent, best_model = results[best_model_name]
                print(f"    {feature}: Best={best_model_name}, R²={best_r2:.4f}, MAE%={best_mae_percent:.2f}, Samples={len(data_filtered)}")

                store_dict[(variety, feature)] = best_model
                score_dict[(variety, feature)] = best_r2

                if name == "metabolite":
                    model_types_met[(variety, feature)] = best_model_name
                    mae_scores_met[(variety, feature)] = best_mae_percent
                    sample_counts_met[(variety, feature)] = len(data_filtered)
                else:
                    model_types_phy[(variety, feature)] = best_model_name
                    mae_scores_phy[(variety, feature)] = best_mae_percent
                    sample_counts_phy[(variety, feature)] = len(data_filtered)

    train_models(metabolite_df, metabolite_features, "metabolite", predictions_met, r2_scores_met)
    train_models(physiological_df, physiological_features, "physiological", predictions_phy, r2_scores_phy)

    print("\nFinished training.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(predictions_met, "models/predictions_met.pkl")
    joblib.dump(predictions_phy, "models/predictions_phy.pkl")
    joblib.dump(r2_scores_met, "models/r2_scores_met.pkl")
    joblib.dump(r2_scores_phy, "models/r2_scores_phy.pkl")
    joblib.dump(mae_scores_met, "models/mae_scores_met.pkl")
    joblib.dump(mae_scores_phy, "models/mae_scores_phy.pkl")
    joblib.dump(model_types_met, "models/model_types_met.pkl")
    joblib.dump(model_types_phy, "models/model_types_phy.pkl")
    joblib.dump(sample_counts_met, "models/sample_counts_met.pkl")
    joblib.dump(sample_counts_phy, "models/sample_counts_phy.pkl")
    joblib.dump(metabolite_features, "models/metabolite_features.pkl")
    joblib.dump(physiological_features, "models/physiological_features.pkl")
