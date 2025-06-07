
import streamlit as st
import os
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Grapevine Prediction App", layout="wide")
st.title("🍇 Grapevine Prediction App")
st.markdown("Use this tool to predict metabolite and physiological measurements for different grape varieties and temperatures.")

# === Sidebar ===
st.sidebar.title("Model Data Management")

uploaded_dir = "uploaded_files"
os.makedirs(uploaded_dir, exist_ok=True)

# Section 1: Extra data used in training
trained_files = sorted([
    f for f in os.listdir(uploaded_dir)
    if f.startswith("trained__") and f.endswith(".xlsx")
])

st.sidebar.subheader("📁 Extra Data Used in Model Training")
if trained_files:
    for file in trained_files:
        display_name = file.replace("trained__", "")
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.markdown(f"📄 {display_name}")
        if col2.button("🗑️", key=f"delete_{file}"):
            os.remove(os.path.join(uploaded_dir, file))
            trained_files.remove(file)
            st.sidebar.success(f"Deleted {display_name}.")
            st.sidebar.info("The file was removed. Please click **Retrain Model** below to update the model without this file.")
            st.sidebar.subheader("🔁 Retrain Model")
            if st.sidebar.button("Run Training (after delete)", key=f"retrain_after_delete_{file}"):
                with st.spinner("Training models... Please wait."):
                    exit_code = os.system("python Train_Model.py")
                    if exit_code == 0:
                        st.sidebar.success("Model retrained successfully!")
                    else:
                        st.sidebar.error("Training failed. Please check Train_Model.py.")
                st.rerun()
else:
    st.sidebar.info("No additional files have been used yet.")

# Section 2: Upload and train
st.sidebar.subheader("📤 Add New Data for Training")
st.sidebar.info("""Adding more data can improve model accuracy.

Download the appropriate template, fill it with your data, and upload the completed file.
Make sure not to change the column names so the system can read it correctly.""")

with open("Metabolite_Data_Year_Template.xlsx", "rb") as metabofile:
    st.sidebar.download_button(
        label="Download Metabolite template",
        data=metabofile,
        file_name="Metabolite_data_year_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with open("Physiological_Data_Year_Template.xlsx", "rb") as physiopfile:
    st.sidebar.download_button(
        label="Download Physiological template",
        data=physiopfile,
        file_name="Physiological_data_year_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

uploaded = st.sidebar.file_uploader("**Upload your .xlsx file**", type="xlsx")

if uploaded:
    save_path = os.path.join(uploaded_dir, uploaded.name)
    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"Uploaded {uploaded.name}.")
        st.sidebar.markdown("### ✅ File Received")
    st.sidebar.info("Your file has been uploaded. Click **Retrain Model** below to include it in the model's training.")
    st.sidebar.subheader("🔁 Retrain Model")
    if st.sidebar.button("Run Training"):
        os.remove(save_path)
        trained_path = os.path.join(uploaded_dir, f"trained__{uploaded.name}")
        with open(trained_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("Training models... Please wait."):
            exit_code = os.system("python Train_Model.py")
            if exit_code == 0:
                st.sidebar.success("Model retrained successfully!")
            else:
                st.sidebar.error("Training failed. Please check Train_Model.py.")
        st.rerun()

# === Load and Predict ===
from Train_Model import predictions_met, predictions_phy, metabolite_features, physiological_features, r2_scores_met, r2_scores_phy

user_temp = st.slider("Select Temperature (°C)", min_value=10.0, max_value=45.0, step=0.5, value=35.0)
all_varieties_met = sorted(set([k[0] for k in predictions_met.keys()]))
all_varieties_phy = sorted(set([k[0] for k in predictions_phy.keys()]))
all_varieties = sorted(set(all_varieties_met + all_varieties_phy))
selected_varieties = st.multiselect("Select grape varieties", all_varieties, default=[])
all_features = metabolite_features + physiological_features
selected_features = st.multiselect("Select features to display", all_features, default=[])

def color_r2(val):
    if pd.isnull(val):
        return ''
    color = 'green' if val >= 0.8 else 'red'
    return f'color: {color}; text-align: left; direction: ltr'

if st.button("Generate Predictions"):
    st.header("Metabolite Predictions")
    for feature in metabolite_features:
        if feature not in selected_features:
            continue
        preds = {}
        for variety in all_varieties_met:
            if selected_varieties and variety not in selected_varieties:
                continue
            model = predictions_met.get((variety, feature))
            r2 = r2_scores_met.get((variety, feature))
            if model is None:
                continue
            X_input = pd.DataFrame([[user_temp]], columns=["Temperature (°C)"])
            pred = model.predict(X_input)[0]
            preds[variety] = {"Prediction": pred, "R²": r2 if r2 is not None else np.nan}
        if preds:
            df = pd.DataFrame.from_dict(preds, orient='index').reset_index().rename(columns={'index': 'Variety'})
            st.markdown(f"#### {feature}")
            fig = px.bar(df, x="Variety", y="Prediction",
                         text=df["Prediction"].map(lambda x: f"{x:,.2f}"),
                         height=500, color_discrete_sequence=["#6a0dad"])
            fig.update_traces(textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', margin=dict(t=100, b=40))
            st.plotly_chart(fig, use_container_width=True)
            styled_df = df.style.format({'Prediction': '{:,.2f}', 'R²': '{:.4f}'}).map(color_r2, subset=['R²']).hide(axis='index')
            st.table(styled_df)
            st.markdown("<hr style='margin-top:30px;margin-bottom:30px;'>", unsafe_allow_html=True)

    st.header("Physiological Predictions")
    for feature in physiological_features:
        if feature not in selected_features:
            continue
        preds = {}
        for variety in all_varieties_phy:
            if selected_varieties and variety not in selected_varieties:
                continue
            model = predictions_phy.get((variety, feature))
            r2 = r2_scores_phy.get((variety, feature))
            if model is None:
                continue
            X_input = pd.DataFrame([[user_temp]], columns=["Temperature (°C)"])
            pred = model.predict(X_input)[0]
            preds[variety] = {"Prediction": pred, "R²": r2 if r2 is not None else np.nan}
        if preds:
            df = pd.DataFrame.from_dict(preds, orient='index').reset_index().rename(columns={'index': 'Variety'})
            st.markdown(f"#### {feature}")
            fig = px.bar(df, x="Variety", y="Prediction",
                         text=df["Prediction"].map(lambda x: f"{x:,.2f}"),
                         height=500, color_discrete_sequence=["#800020"])
            fig.update_traces(textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', margin=dict(t=100, b=40))
            st.plotly_chart(fig, use_container_width=True)
            styled_df = df.style.format({'Prediction': '{:,.2f}', 'R²': '{:.4f}'}).map(color_r2, subset=['R²']).hide(axis='index')
            st.table(styled_df)
            st.markdown("<hr style='margin-top:30px;margin-bottom:30px;'>", unsafe_allow_html=True)
