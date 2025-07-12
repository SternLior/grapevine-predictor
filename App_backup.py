import streamlit as st
st.set_page_config(page_title="Grapevine Prediction App", layout="wide")
import os
import pandas as pd
import plotly.express as px
import numpy as np
import importlib
import joblib
import io
from Train_Model import train_and_save_models

# === Load Models Efficiently (cached) ===
@st.cache_resource
def load_model_state():
    predictions_met = joblib.load("models/predictions_met.pkl")
    predictions_phy = joblib.load("models/predictions_phy.pkl")
    metabolite_features = joblib.load("models/metabolite_features.pkl")
    physiological_features = joblib.load("models/physiological_features.pkl")
    r2_scores_met = joblib.load("models/r2_scores_met.pkl")
    r2_scores_phy = joblib.load("models/r2_scores_phy.pkl")
    model_types_met = joblib.load("models/model_types_met.pkl")
    model_types_phy = joblib.load("models/model_types_phy.pkl")
    mae_scores_met = joblib.load("models/mae_scores_met.pkl")
    mae_scores_phy = joblib.load("models/mae_scores_phy.pkl")
    sample_counts_met = joblib.load("models/sample_counts_met.pkl")
    sample_counts_phy = joblib.load("models/sample_counts_phy.pkl")

    return (
        predictions_met, predictions_phy,
        metabolite_features, physiological_features,
        r2_scores_met, r2_scores_phy,
        model_types_met, model_types_phy,
        mae_scores_met, mae_scores_phy,
        sample_counts_met, sample_counts_phy
    )

# Load model state
(
    predictions_met, predictions_phy,
    metabolite_features, physiological_features,
    r2_scores_met, r2_scores_phy,
    model_types_met, model_types_phy,
    mae_scores_met, mae_scores_phy,
    sample_counts_met, sample_counts_phy
) = load_model_state()

st.title("🍇 Grapevine Prediction App")
st.markdown("Use this tool to predict metabolite and physiological measurements for different grape varieties and temperatures.")

uploaded_dir = "uploaded_files"
os.makedirs(uploaded_dir, exist_ok=True)

# === Template headers ===
metabolite_required_cols = pd.read_excel("Metabolite_Data_Year_Template.xlsx", nrows=0).columns.tolist()
physiological_required_cols = pd.read_excel("Physiological_Data_Year_Template.xlsx", nrows=0).columns.tolist()

# === Sidebar ===
st.sidebar.title("Model Data Management")

# Training logs will appear here during model training

# Section 1: Show already used data
trained_files = sorted([f for f in os.listdir(uploaded_dir) if f.startswith("trained__") and f.endswith(".xlsx")])
st.sidebar.subheader("📁 Extra Data Used in Model Training")
if trained_files:
    for file in trained_files:
        display_name = file.replace("trained__", "")
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.markdown(f"📄 {display_name}")
        if col2.button("🗑️", key=f"delete_{file}"):
            os.remove(os.path.join(uploaded_dir, file))
            # Clear the processed flag for this file so it can be re-uploaded
            file_name = file.replace("trained__", "")
            processed_key = f"processed_{file_name}"
            if processed_key in st.session_state:
                st.session_state.pop(processed_key)
            st.sidebar.info("The file was removed. Please click **Retrain Model** below to update the model without this file.")
            st.session_state["retrain_after_delete"] = True
else:
    st.sidebar.info("No additional files have been used yet.")

# Trigger retraining only after delete
if st.session_state.get("retrain_after_delete"):
    st.sidebar.subheader("🔁 Retrain Model")
    if st.sidebar.button("Run Training (after delete)"):
        with st.spinner("Training model... Please wait."):
            # Capture training logs
            import io
            import sys
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                train_and_save_models()
            
            # Display logs in sidebar temporarily
            logs = f.getvalue()
            st.sidebar.text_area("📊 Training Logs (will disappear after refresh)", logs, height=200)
            
        st.cache_resource.clear()  # Refresh cached models
        st.sidebar.success("Model retrained successfully!")
        st.session_state.pop("retrain_after_delete")
        st.rerun()

# Section 2: Upload new data
st.sidebar.subheader("📤  Add New Data for Training")
st.sidebar.info("""
Adding more data can improve model accuracy.

Download the appropriate template, fill it with your data, and upload the completed file. Make sure not to change the column names so the system can read it correctly.
""")

with open("Metabolite_Data_Year_Template.xlsx", "rb") as metabofile:
    st.sidebar.download_button("Download Metabolite template", metabofile, file_name="Metabolite_template.xlsx")
with open("Physiological_Data_Year_Template.xlsx", "rb") as physiopfile:
    st.sidebar.download_button("Download Physiological template", physiopfile, file_name="Physiological_template.xlsx")

uploaded = st.sidebar.file_uploader("Upload your .xlsx file", type="xlsx")
file_valid = False
file_type = None
save_path = None

if uploaded:
    df = pd.read_excel(uploaded, nrows=1)
    uploaded_cols = set(df.columns)
    if set(metabolite_required_cols).issubset(uploaded_cols):
        file_type = "metabolite data"
        file_valid = True
    elif set(physiological_required_cols).issubset(uploaded_cols):
        file_type = "physiological data"
        file_valid = True

    if file_valid:
        save_path = os.path.join(uploaded_dir, uploaded.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
        st.session_state["retrain_needed"] = uploaded.name
    else:
        st.sidebar.error("❌ Uploaded file does not match the required format. Please use the provided templates.")
        st.session_state["retrain_needed"] = None

if uploaded and file_valid and st.session_state.get("retrain_needed") == uploaded.name:
    # Check if this specific file was already processed
    file_processed_key = f"processed_{uploaded.name}"
    
    if file_processed_key not in st.session_state:
        st.sidebar.success(f"Uploaded {uploaded.name} (detected as {file_type}). Click **Retrain Model** to include this file in model training.")
        st.sidebar.subheader("🔁 Retrain Model to include uploaded data")
        if st.sidebar.button("Retrain Model"):
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
            trained_path = os.path.join(uploaded_dir, f"trained__{uploaded.name}")
            with open(trained_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("Training model... Please wait."):
                # Capture training logs
                import io
                import sys
                from contextlib import redirect_stdout
                
                f = io.StringIO()
                with redirect_stdout(f):
                    train_and_save_models()
                
                # Display logs in sidebar temporarily
                logs = f.getvalue()
                st.sidebar.text_area("📊 Training Logs (will disappear after refresh)", logs, height=200)
                
            st.cache_resource.clear()
            st.sidebar.success("Model retrained successfully!")
            st.session_state[file_processed_key] = True
            st.session_state.pop("retrain_needed")
            st.rerun()
    else:
        st.sidebar.info(f"File {uploaded.name} has already been processed. Upload a different file or delete this one first.")

# === Prediction Interface ===
user_temp = st.slider("Select Temperature (°C)", min_value=10, max_value=45, step=1, value=35)
all_varieties_met = sorted(set([k[0] for k in predictions_met.keys()]))
all_varieties_phy = sorted(set([k[0] for k in predictions_phy.keys()]))
all_varieties = sorted(set(all_varieties_met + all_varieties_phy))
selected_varieties = st.multiselect("Select grape varieties", all_varieties, default=[])
all_features = metabolite_features + physiological_features
selected_features = st.multiselect("Select features to display", all_features, default=[])

selected_metabolite_features = [f for f in selected_features if f in metabolite_features]
selected_physiological_features = [f for f in selected_features if f in physiological_features]

def render_table_as_html(df):
    styled_rows = []
    for _, row in df.iterrows():
        row_cells = []
        for col in df.columns:
            val = row[col]
            style = "text-align:left; direction:ltr"
            if col == "R²":
                try:
                    float_val = float(val)
                    color = "green" if float_val >= 0.8 else "red"
                    style += f"; color:{color}"
                except:
                    pass
            row_cells.append(f"<td style='{style}'>{val}</td>")
        styled_rows.append(f"<tr>{''.join(row_cells)}</tr>")
    headers = "".join([f"<th style='text-align:left; direction:ltr'>{col}</th>" for col in df.columns])
    html_table = f"""
    <table style='width:100%; border-collapse: collapse; direction: ltr;'>
        <thead><tr>{headers}</tr></thead>
        <tbody>{"".join(styled_rows)}</tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

if st.button("Generate Predictions"):
    # Store all prediction data for download
    all_predictions_data = []
    filtered_predictions_data = []
    
    if selected_metabolite_features:
        st.header("Metabolite Predictions")
        for feature in selected_metabolite_features:
            preds = {}
            for variety in all_varieties_met:
                if selected_varieties and variety not in selected_varieties:
                    continue
                model = predictions_met.get((variety, feature))
                r2 = r2_scores_met.get((variety, feature))
                if model is None:
                    continue
                try:
                    pred = model.predict(pd.DataFrame([[user_temp]], columns=pd.Index(["Temperature (°C)"])))[0]
                except Exception:
                    continue
                preds[variety] = {"Prediction": pred, "R²": r2}
            if preds:
                df = pd.DataFrame.from_dict(preds, orient="index").reset_index().rename(columns={"index": "Variety"})
                df["MAE%"] = df.apply(lambda row: mae_scores_met.get((row["Variety"], feature), np.nan), axis=1)
                df["MAE%"] = df["MAE%"].astype(float).map("{:.2f}".format)
                df["Model"] = df.apply(lambda row: model_types_met.get((row["Variety"], feature), ""), axis=1)
                df["Samples"] = df.apply(lambda row: sample_counts_met.get((row["Variety"], feature), np.nan), axis=1)
                df = df[["Variety", "Prediction", "R²", "MAE%", "Model", "Samples"]]
                df["Prediction"] = pd.Series(df["Prediction"]).map("{:,.2f}".format)
                df["R²"] = pd.Series(df["R²"]).map("{:.4f}".format)
                
                # Store data for download (numerical format)
                df_numerical = pd.DataFrame.from_dict(preds, orient="index").reset_index().rename(columns={"index": "Variety"})
                df_numerical["MAE%"] = df_numerical.apply(lambda row: mae_scores_met.get((row["Variety"], feature), np.nan), axis=1)
                df_numerical["Model"] = df_numerical.apply(lambda row: model_types_met.get((row["Variety"], feature), ""), axis=1)
                df_numerical["Samples"] = df_numerical.apply(lambda row: sample_counts_met.get((row["Variety"], feature), np.nan), axis=1)
                df_numerical["Feature"] = feature
                df_numerical["Type"] = "Metabolite"
                df_numerical["Temperature"] = user_temp
                df_numerical = df_numerical[["Variety", "Feature", "Type", "Temperature", "Prediction", "R²", "MAE%", "Model", "Samples"]]
                
                filtered_predictions_data.append(df_numerical)
                all_predictions_data.append(df_numerical)
                
                st.markdown(f"#### {feature}")
                fig = px.bar(df, x="Variety", y="Prediction", text=df["Prediction"], height=500)
                fig.update_traces(textposition="outside")
                fig.update_layout(margin=dict(t=100, b=40))
                st.plotly_chart(fig, use_container_width=True)
                render_table_as_html(df)
                st.markdown("<hr style='margin-top:30px;margin-bottom:30px;'>", unsafe_allow_html=True)

    if selected_physiological_features:
        st.header("Physiological Predictions")
        for feature in selected_physiological_features:
            preds = {}
            for variety in all_varieties_phy:
                if selected_varieties and variety not in selected_varieties:
                    continue
                model = predictions_phy.get((variety, feature))
                r2 = r2_scores_phy.get((variety, feature))
                if model is None:
                    continue
                try:
                    pred = model.predict(pd.DataFrame([[user_temp]], columns=pd.Index(["Temperature (°C)"])))[0]
                except Exception:
                    continue
                preds[variety] = {"Prediction": pred, "R²": r2}
            if preds:
                df = pd.DataFrame.from_dict(preds, orient="index").reset_index().rename(columns={"index": "Variety"})
                df["MAE%"] = df.apply(lambda row: mae_scores_phy.get((row["Variety"], feature), np.nan), axis=1)
                df["MAE%"] = df["MAE%"].astype(float).map("{:.2f}".format)
                df["Model"] = df.apply(lambda row: model_types_phy.get((row["Variety"], feature), ""), axis=1)
                df["Samples"] = df.apply(lambda row: sample_counts_phy.get((row["Variety"], feature), np.nan), axis=1)
                df = df[["Variety", "Prediction", "R²", "MAE%", "Model", "Samples"]]
                df["Prediction"] = pd.Series(df["Prediction"]).map("{:,.2f}".format)
                df["R²"] = pd.Series(df["R²"]).map("{:.4f}".format)
                
                # Store data for download (numerical format)
                df_numerical = pd.DataFrame.from_dict(preds, orient="index").reset_index().rename(columns={"index": "Variety"})
                df_numerical["MAE%"] = df_numerical.apply(lambda row: mae_scores_phy.get((row["Variety"], feature), np.nan), axis=1)
                df_numerical["Model"] = df_numerical.apply(lambda row: model_types_phy.get((row["Variety"], feature), ""), axis=1)
                df_numerical["Samples"] = df_numerical.apply(lambda row: sample_counts_phy.get((row["Variety"], feature), np.nan), axis=1)
                df_numerical["Feature"] = feature
                df_numerical["Type"] = "Physiological"
                df_numerical["Temperature"] = user_temp
                df_numerical = df_numerical[["Variety", "Feature", "Type", "Temperature", "Prediction", "R²", "MAE%", "Model", "Samples"]]
                
                filtered_predictions_data.append(df_numerical)
                all_predictions_data.append(df_numerical)
                
                st.markdown(f"#### {feature}")
                fig = px.bar(df, x="Variety", y="Prediction", text=df["Prediction"], height=500)
                fig.update_traces(textposition="outside")
                fig.update_layout(margin=dict(t=100, b=40))
                st.plotly_chart(fig, use_container_width=True)
                render_table_as_html(df)
                st.markdown("<hr style='margin-top:30px;margin-bottom:30px;'>", unsafe_allow_html=True)

    # Generate all data for download
    def generate_all_predictions():
        all_data = []
        temperatures = range(10, 46)  # 10°C to 45°C
        
        for temp in temperatures:
            for variety in all_varieties_met:
                for feature in metabolite_features:
                    model = predictions_met.get((variety, feature))
                    r2 = r2_scores_met.get((variety, feature))
                    if model is None:
                        continue
                    try:
                        pred = model.predict(pd.DataFrame([[temp]], columns=pd.Index(["Temperature (°C)"])))[0]
                    except Exception:
                        continue
                    
                    row = {
                        "Variety": variety,
                        "Feature": feature,
                        "Type": "Metabolite",
                        "Temperature": temp,
                        "Prediction": pred,
                        "R²": r2,
                        "MAE%": mae_scores_met.get((variety, feature), np.nan),
                        "Model": model_types_met.get((variety, feature), ""),
                        "Samples": sample_counts_met.get((variety, feature), np.nan)
                    }
                    all_data.append(row)
            
            for variety in all_varieties_phy:
                for feature in physiological_features:
                    model = predictions_phy.get((variety, feature))
                    r2 = r2_scores_phy.get((variety, feature))
                    if model is None:
                        continue
                    try:
                        pred = model.predict(pd.DataFrame([[temp]], columns=pd.Index(["Temperature (°C)"])))[0]
                    except Exception:
                        continue
                    
                    row = {
                        "Variety": variety,
                        "Feature": feature,
                        "Type": "Physiological",
                        "Temperature": temp,
                        "Prediction": pred,
                        "R²": r2,
                        "MAE%": mae_scores_phy.get((variety, feature), np.nan),
                        "Model": model_types_phy.get((variety, feature), ""),
                        "Samples": sample_counts_phy.get((variety, feature), np.nan)
                    }
                    all_data.append(row)
        
        return pd.DataFrame(all_data)
    
    # Add download buttons at the end
    if filtered_predictions_data or all_predictions_data:
        st.markdown("---")
        st.subheader("Download Data")
        
        # Download filtered data button
        if filtered_predictions_data:
            filtered_df = pd.concat(filtered_predictions_data, ignore_index=True)
            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
                filtered_df.to_excel(writer, sheet_name='Filtered_Predictions', index=False)
            output.seek(0)
            st.download_button(
                label="Download filtered data",
                data=output.getvalue(),
                file_name=f"filtered_predictions_{user_temp}C.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Download all data button
        all_df = generate_all_predictions()
        if not all_df.empty:
            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
                all_df.to_excel(writer, sheet_name='All_Predictions', index=False)
            output.seek(0)
            st.download_button(
                label="Download all data",
                data=output.getvalue(),
                file_name="all_predictions_all_temperatures.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
