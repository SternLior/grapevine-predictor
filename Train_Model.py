from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm

pd.set_option('display.float_format', '{:.4f}'.format)

class XGBoostRandomForestEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble model that combines XGBoost and Random Forest predictions
    """
    def __init__(self, xgb_params=None, rf_params=None, ensemble_method='average'):
        self.xgb_params = xgb_params or {
            'n_estimators': 25, 
            'max_depth': 3, 
            'learning_rate': 0.1, 
            'random_state': 42, 
            'n_jobs': 1
        }
        self.rf_params = rf_params or {
            'n_estimators': 25, 
            'max_depth': 3, 
            'random_state': 42, 
            'n_jobs': 1,
            'criterion': 'squared_error',
            'bootstrap': True,
            'oob_score': False,
            'warm_start': False
        }
        self.ensemble_method = ensemble_method
        self.xgb_model = XGBRegressor(**self.xgb_params)
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.is_fitted = False
    
    def fit(self, X, y):
        # Train XGBoost model
        self.xgb_model.fit(X, y)
        
        # Train Random Forest model
        self.rf_model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from both models
        xgb_pred = self.xgb_model.predict(X)
        rf_pred = self.rf_model.predict(X)
        
        # Combine predictions (simple average)
        if self.ensemble_method == 'average':
            return (xgb_pred + rf_pred) / 2
        else:
            return (xgb_pred + rf_pred) / 2  # Default to average

def train_and_save_models():
    base_metabolite_file = "Metabolite_Data_2022.xlsx"
    base_physiological_file = "Physiological_data_2022-2023.xlsx"
    uploaded_dir = "uploaded_files"
    os.makedirs(uploaded_dir, exist_ok=True)

    used_files = [
        os.path.join(uploaded_dir, f) for f in os.listdir(uploaded_dir)
        if f.startswith("trained__") and f.endswith(".xlsx")
    ]

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
        except Exception:
            classified_files["unknown"].append(fpath)

    def load_all_data(base_file, extra_file_paths):
        dfs = [pd.read_excel(base_file)]
        for path in extra_file_paths:
            if path in file_dataframes:
                dfs.append(file_dataframes[path])
        combined = pd.concat(dfs, ignore_index=True)
        return combined

    metabolite_df = load_all_data(base_metabolite_file, classified_files["metabolite"])
    physiological_df = load_all_data(base_physiological_file, classified_files["physiological"])

    non_feature_cols = ['Date', 'Time', 'Sample', 'Variety', 'Plot', 'obs', 'ID', 'Temperature (°C)', 'Avg_temp_72h (°C)', 'Max_temp_24h (°C)']
    metabolite_features = [col for col in metabolite_df.columns if col not in non_feature_cols]
    physiological_features = [col for col in physiological_df.columns if col not in non_feature_cols]

    predictions_met, r2_scores_met = {}, {}
    predictions_phy, r2_scores_phy = {}, {}
    model_types_met, mae_scores_met, sample_counts_met = {}, {}, {}
    model_types_phy, mae_scores_phy, sample_counts_phy = {}, {}, {}

    def train_models(df, features, name, store_dict, score_dict):
        print(f"🔁 Training started for {name} features...")
        df = df[['Temperature (°C)', 'Variety'] + features].dropna(subset=['Temperature (°C)', 'Variety'])

        # Count total combinations for progress bar
        total_combinations = 0
        for variety in df['Variety'].unique():
            variety_df = df[df['Variety'] == variety]
            for feature in features:
                data_filtered = variety_df[['Temperature (°C)', feature]].dropna()
                if not data_filtered.empty and len(data_filtered) >= 10:
                    total_combinations += 1

        print(f"📊 Training {total_combinations} variety-feature combinations...")
        
        with tqdm(total=total_combinations, desc=f"Training {name} models") as pbar:
            for variety in df['Variety'].unique():
                variety_df = df[df['Variety'] == variety]
                for feature in features:
                    data_filtered = variety_df[['Temperature (°C)', feature]].dropna()
                    if data_filtered.empty or len(data_filtered) < 10:
                        continue

                    X = data_filtered[['Temperature (°C)']]
                    y = data_filtered[feature].reset_index(drop=True)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    if len(X_train) == 0:
                        continue

                    # Simplified model selection - start with fastest models
                    models = {
                        'LinearRegression': LinearRegression(),
                        'RandomForest': RandomForestRegressor(n_estimators=25, max_depth=3, random_state=42, n_jobs=1),
                        'XGBoost': XGBRegressor(n_estimators=25, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1),
                        'XGB+RF': XGBoostRandomForestEnsemble()
                    }

                    results = {}
                    for model_name, model in models.items():
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = mean_squared_error(y_test, y_pred) ** 0.5
                            r2 = r2_score(y_test, y_pred)
                            mean_actual = float(pd.Series(y_test).mean())  # Fix linter error
                            mae_percent = (mae / mean_actual) * 100 if mean_actual != 0 else float('nan')
                            results[model_name] = (mae, rmse, r2, mae_percent, model)
                        except Exception as e:
                            print(f"⚠️ Error training {model_name} for {variety}-{feature}: {e}")
                            continue

                    if results:
                        best_model_name = max(results, key=lambda k: results[k][2])
                        best_mae, best_rmse, best_r2, best_mae_percent, best_model = results[best_model_name]

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

                    pbar.update(1)

        print(f"✅ Finished training for {name}.")

    train_models(metabolite_df, metabolite_features, "metabolite", predictions_met, r2_scores_met)
    train_models(physiological_df, physiological_features, "physiological", predictions_phy, r2_scores_phy)

    print("📦 Saving models...")
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

    print("✅ Model training and saving completed.")

if __name__ == "__main__":
    train_and_save_models()
