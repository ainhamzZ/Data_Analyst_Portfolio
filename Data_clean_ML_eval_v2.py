import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from pycaret.classification import *
from pycaret.regression import *
from ydata_profiling import ProfileReport
from imblearn.over_sampling import SMOTE
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

# -------------------------
# Streamlit App Config
# -------------------------
st.set_page_config(page_title="AutoML App with PyCaret", layout="wide")
st.title("🤖 AutoML Predictive Analytics App")

# -------------------------
# Upload Dataset
# -------------------------
st.sidebar.header("Upload your dataset")
file = st.sidebar.file_uploader("Upload your CSV, TXT, or Excel file", type=["csv", "txt", "xlsx"])
generate_profile = st.sidebar.checkbox("Generate Profiling Report (can be slow on large datasets)", value=False)

if file:
    ext = file.name.split('.')[-1].lower()
    if ext == 'csv':
        df = pd.read_csv(file)
    elif ext == 'txt':
        df = pd.read_csv(file, delimiter='\t')
    elif ext == 'xlsx':
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format!")
        st.stop()

    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Dataset Profiling
    # -------------------------
    if generate_profile:
        st.subheader("📈 Dataset Profiling Report")
        plt.close('all')
        profile = ProfileReport(df, explorative=True)
        profile_html = profile.to_html()
        st.components.v1.html(profile_html, height=800, scrolling=True)

        # Allow download of HTML report
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            profile.to_file(tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button(
                    label="📥 Download Profiling Report (HTML)",
                    data=f,
                    file_name="dataset_profile_report.html",
                    mime="text/html"
                )

    # -------------------------
    # Data Cleaning Options
    # -------------------------
    st.subheader("🧹 Data Cleaning")
    if st.checkbox("Drop columns with >50% missing values"):
        df = df.loc[:, df.isnull().mean() < 0.5]

    if st.checkbox("Fill missing numeric values with mean"):
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    if st.checkbox("Fill missing categorical values with mode"):
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove near-constant columns
    df = pd.DataFrame(VarianceThreshold().fit_transform(df), columns=df.columns[:df.shape[1]])

    # -------------------------
    # Target Column and Options
    # -------------------------
    target_col = st.text_input("🎯 Enter Target Column Name")
    if target_col and target_col in df.columns:
        st.success(f"Target column set to: {target_col}")

        problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"])

        st.subheader("⚙️ Advanced Preprocessing Options")
        normalize_data = st.checkbox("Normalize/Transform Skewed Features", value=False)
        remove_outliers = st.checkbox("Remove Outliers", value=False)
        outlier_threshold = 0.05
        if remove_outliers:
            outlier_threshold = st.slider("Outlier Removal Threshold", 0.01, 0.2, 0.05, 0.01)

        fix_imbalance = False
        if problem_type == "Classification":
            fix_imbalance = st.checkbox("Apply SMOTE to Handle Class Imbalance", value=False)
            stratify_folds = st.checkbox("Use Stratified Sampling", value=True)
        else:
            stratify_folds = False

        # -------------------------
        # Run PyCaret Setup & Training
        # -------------------------
        if st.button("🚀 Run PyCaret Setup and Train"):
            with st.spinner('Training models... please wait ⏳'):
                # Apply SMOTE if needed
                if fix_imbalance:
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    smote = SMOTE(random_state=123)
                    X_res, y_res = smote.fit_resample(X, y)
                    df_resampled = X_res.copy()
                    df_resampled[target_col] = y_res
                else:
                    df_resampled = df.copy()

                # -------------------------
                # Classification
                # -------------------------
                if problem_type == "Classification":
                    s = setup(
                        data=df_resampled,
                        target=target_col,
                        verbose=False,
                        session_id=123,
                        normalize=normalize_data,
                        remove_outliers=remove_outliers,
                        outliers_threshold=outlier_threshold,
                        fold_strategy='stratifiedkfold' if stratify_folds else 'kfold'
                    )

                # -------------------------
                # Regression
                # -------------------------
                else:
                    s = setup(
                        data=df_resampled,
                        target=target_col,
                        verbose=False,
                        session_id=123,
                        normalize=normalize_data,
                        remove_outliers=remove_outliers,
                        outliers_threshold=outlier_threshold
                    )

                # Compare models
                compare_models_list = compare_models(n_select=5)
                results = pull()

                # -------------------------
                # Display Results
                # -------------------------
                st.subheader("🏆 Best Model")
                st.write(compare_models_list[0])

                st.subheader("📊 Model Comparison Dashboard")
                metric_options = results.columns.tolist()
                selected_metric = st.selectbox("Select metric to visualize", metric_options)
                fig = px.bar(results, x=results.index, y=selected_metric,
                             title=f"Model Comparison by {selected_metric}")
                st.plotly_chart(fig, use_container_width=True)

                # Detailed evaluation
                st.subheader("🔍 Detailed Evaluation of Selected Model")
                selected_model_name = st.selectbox("Select Model", results.index.tolist())
                selected_model = compare_models_list[results.index.get_loc(selected_model_name)]
                st.write(selected_model)
                evaluate_model(selected_model)

                # Save and download model
                save_model(selected_model, 'best_model')
                with open('best_model.pkl', 'rb') as model_file:
                    st.download_button(label="📦 Download Trained Model", data=model_file,
                                       file_name="best_model.pkl")

    elif target_col:
        st.error("The column name you entered does not exist in the dataset.")
else:
    st.info("👈 Please upload a dataset to get started.")
