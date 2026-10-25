from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.custom_generator import generate_custom_dataset
from src.data_io import read_uploaded_csv
from src.profiler import profile_dataset
from src.generation import generate_synthetic
from src.evaluation import evaluate_datasets, add_sdv_quality, report_json
from src.ctgan_generator import (
    train_ctgan_model,
    load_ctgan_model,
    generate_ctgan_data,
    select_reference_columns,
)

try:
    from sdv.evaluation.single_table import evaluate_quality
except Exception:
    evaluate_quality = None


# ============================================================
# PATHS AND APPLICATION CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
GENERATED_DIR = OUTPUT_DIR / "generated"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_CATALOG_PATH = MODEL_DIR / "model_catalog.json"

for directory in [MODEL_DIR, GENERATED_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="SynData AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        padding: 14px;
        background: rgba(128,128,128,.04);
    }
    .hero {
        padding: 1.5rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(100,100,100,.12), rgba(100,100,100,.03));
        margin-bottom: 1rem;
    }
    .small-note { opacity: .75; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_filename(value: str, fallback: str = "dataset") -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")
    return value or fallback


def file_fingerprint(uploaded_file) -> str:
    content = uploaded_file.getvalue()
    return hashlib.sha256(content).hexdigest()[:16]


def dataframe_fingerprint(data: pd.DataFrame) -> str:
    try:
        raw = pd.util.hash_pandas_object(data, index=True).values.tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(data.shape).encode()).hexdigest()[:16]


def read_csv_safely(path: Path):
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as error:
        return None, str(error)


def load_catalog() -> list:
    if not MODEL_CATALOG_PATH.exists():
        return []
    try:
        content = json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))
        return content if isinstance(content, list) else []
    except Exception:
        return []


def save_catalog(catalog: list):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, default=str),
        encoding="utf-8",
    )


def register_model(model_path: Path, metadata: dict | None = None):
    catalog = load_catalog()
    model_path = model_path.resolve()
    entry = {
        "name": model_path.stem,
        "path": str(model_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metadata:
        entry.update(metadata)

    catalog = [item for item in catalog if Path(item.get("path", "")).resolve() != model_path]
    catalog.append(entry)
    save_catalog(catalog)


def discover_models() -> list:
    models = []
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        models.append(
            {
                "name": path.stem,
                "path": str(path.resolve()),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )

    catalog = {item.get("path"): item for item in load_catalog()}
    for item in models:
        saved = catalog.get(item["path"], {})
        item["dataset_name"] = saved.get("dataset_name", "Unknown")
        item["columns"] = saved.get("columns", [])
        item["rows"] = saved.get("rows", "Unknown")
        item["epochs"] = saved.get("epochs", "Unknown")
    return models


def clear_active_generation():
    for key in [
        "ctgan_synthesizer",
        "ctgan_model_path",
        "ctgan_generated_data",
        "current_real_data",
        "current_synthetic_data",
        "current_run_name",
        "current_run_type",
        "current_metadata",
    ]:
        st.session_state.pop(key, None)


def set_current_run(real_data, synthetic_data, run_name, run_type, model_path=None, metadata=None):
    st.session_state.current_real_data = real_data.copy()
    st.session_state.current_synthetic_data = synthetic_data.copy()
    st.session_state.current_run_name = run_name
    st.session_state.current_run_type = run_type
    if model_path:
        st.session_state.ctgan_model_path = str(model_path)
    if metadata is not None:
        st.session_state.current_metadata = metadata


def numeric_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def comparable_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> list:
    return [column for column in real.columns if column in synthetic.columns]


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def distribution_similarity(real_values, synthetic_values) -> float | None:
    real_values = safe_numeric(real_values)
    synthetic_values = safe_numeric(synthetic_values)
    if real_values.empty or synthetic_values.empty:
        return None
    low = min(real_values.min(), synthetic_values.min())
    high = max(real_values.max(), synthetic_values.max())
    if low == high:
        return 1.0
    real_hist, _ = np.histogram(real_values, bins=20, range=(low, high), density=True)
    synth_hist, _ = np.histogram(synthetic_values, bins=20, range=(low, high), density=True)
    denominator = np.maximum(np.abs(real_hist), 1e-12)
    relative_error = np.mean(np.abs(real_hist - synth_hist) / denominator)
    return float(max(0.0, min(1.0, 1.0 - relative_error)))


def numerical_comparison(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in comparable_columns(real, synthetic):
        r = safe_numeric(real[column])
        s = safe_numeric(synthetic[column])
        if r.empty or s.empty:
            continue
        similarity = distribution_similarity(r, s)
        rows.append({
            "Column": column,
            "Real_Mean": r.mean(),
            "Synthetic_Mean": s.mean(),
            "Mean_Difference": abs(r.mean() - s.mean()),
            "Real_Std": r.std(),
            "Synthetic_Std": s.std(),
            "Real_Min": r.min(),
            "Synthetic_Min": s.min(),
            "Real_Max": r.max(),
            "Synthetic_Max": s.max(),
            "Distribution_Similarity_Percent": round((similarity or 0) * 100, 2),
        })
    return pd.DataFrame(rows)


def categorical_comparison(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in comparable_columns(real, synthetic):
        if column not in categorical_columns(real) and column not in categorical_columns(synthetic):
            continue
        real_freq = real[column].fillna("<MISSING>").astype(str).value_counts(normalize=True)
        synth_freq = synthetic[column].fillna("<MISSING>").astype(str).value_counts(normalize=True)
        categories = sorted(set(real_freq.index).union(synth_freq.index))
        tvd = 0.5 * sum(abs(real_freq.get(item, 0) - synth_freq.get(item, 0)) for item in categories)
        rows.append({
            "Column": column,
            "Unique_Real": int(real[column].nunique(dropna=False)),
            "Unique_Synthetic": int(synthetic[column].nunique(dropna=False)),
            "Distribution_Difference": round(float(tvd), 4),
            "Distribution_Similarity_Percent": round((1 - float(tvd)) * 100, 2),
        })
    return pd.DataFrame(rows)


def correlation_comparison(real: pd.DataFrame, synthetic: pd.DataFrame):
    columns = [c for c in comparable_columns(real, synthetic) if c in numeric_columns(real) and c in numeric_columns(synthetic)]
    if len(columns) < 2:
        return pd.DataFrame(), None
    real_corr = real[columns].corr(numeric_only=True)
    synth_corr = synthetic[columns].corr(numeric_only=True)
    common = real_corr.index.intersection(synth_corr.index)
    difference = (real_corr.loc[common, common] - synth_corr.loc[common, common]).abs()
    np.fill_diagonal(difference.values, 0)
    average_difference = float(difference.values.sum() / max(1, len(common) * (len(common) - 1)))
    return difference, average_difference


def exact_match_count(real: pd.DataFrame, synthetic: pd.DataFrame) -> int:
    columns = comparable_columns(real, synthetic)
    if not columns:
        return 0
    real_rows = set(map(tuple, real[columns].fillna("<MISSING>").astype(str).to_numpy()))
    synthetic_rows = synthetic[columns].fillna("<MISSING>").astype(str).to_numpy()
    return sum(tuple(row) in real_rows for row in synthetic_rows)


def evaluate_current_pair(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    result = {
        "real_rows": len(real),
        "synthetic_rows": len(synthetic),
        "real_columns": len(real.columns),
        "synthetic_columns": len(synthetic.columns),
        "common_columns": len(comparable_columns(real, synthetic)),
        "exact_matches": exact_match_count(real, synthetic),
    }
    num = numerical_comparison(real, synthetic)
    cat = categorical_comparison(real, synthetic)
    result["numeric_similarity"] = round(float(num["Distribution_Similarity_Percent"].mean()), 2) if not num.empty else None
    result["categorical_similarity"] = round(float(cat["Distribution_Similarity_Percent"].mean()), 2) if not cat.empty else None
    result["average_correlation_difference"] = correlation_comparison(real, synthetic)[1]
    return result


def run_sdv_quality(real: pd.DataFrame, synthetic: pd.DataFrame):
    if evaluate_quality is None:
        return None, "SDV quality evaluator is not available in this installation."
    try:
        from sdv.metadata import Metadata
        metadata = Metadata.detect_from_dataframe(
            data=real,
            table_name="evaluation_table",
        )
        report = evaluate_quality(
            real_data=real,
            synthetic_data=synthetic,
            metadata=metadata,
        )
        properties = report.get_properties()
        score = float(report.get_score())
        return {"overall": score, "properties": properties}, None
    except Exception as error:
        return None, str(error)


def save_generated_csv(data: pd.DataFrame, name: str) -> Path:
    path = GENERATED_DIR / f"{clean_filename(name)}.csv"
    data.to_csv(path, index=False)
    return path


def render_download(data: pd.DataFrame, filename: str, label="📥 Download CSV"):
    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

 
def display_pair_summary(real: pd.DataFrame, synthetic: pd.DataFrame):
    metrics = evaluate_current_pair(real, synthetic)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real Rows", f"{metrics['real_rows']:,}")
    c2.metric("Synthetic Rows", f"{metrics['synthetic_rows']:,}")
    c3.metric("Common Columns", metrics["common_columns"])
    c4.metric("Exact Matches", metrics["exact_matches"])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🤖 SynData")
st.sidebar.caption("AI-Based Synthetic Dataset Generation and Evaluation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠  Dashboard",
        "📊 Dataset Analysis",
        "Profiler & Experiment",
        "🛠️ Custom Dataset Builder",
        "🤖 AI Generator",
        "📚 Model Library",
        "🆚 Dynamic Evaluation",
        "🔐 Privacy Screening",
        "📥 Downloads",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### Technology")
st.sidebar.caption("Python • Pandas • NumPy")
st.sidebar.caption("SDV • CTGAN • Streamlit")
st.sidebar.caption("Dynamic evaluation and visualization")
st.sidebar.divider()
st.sidebar.info("Name: YOGESH SAHU - SMIT2627179\n\nProject :AI-Based Synthetic Dataset Generation and Evaluation")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🤖 SynData AI</h1>
        <p><b>AI-Based Synthetic Dataset Generation and Evaluation</b></p>
        <p class="small-note">Create custom datasets, reuse trained models, generate synthetic records, and evaluate results dynamically.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠  Dashboard":
    st.header("📊 Project Dashboard")

    default_real_path = DATA_DIR / "telco_cleaned.csv"
    real_data, error = read_csv_safely(default_real_path)
    models = discover_models()
    generated_files = list(GENERATED_DIR.glob("*.csv"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved Models", len(models))
    c2.metric("Generated Datasets", len(generated_files))
    c3.metric("Reference Rows", f"{len(real_data):,}" if real_data is not None else "â€”")
    c4.metric("Reference Columns", len(real_data.columns) if real_data is not None else "â€”")

    st.subheader(" 🔄 Application Workflow")
    workflow = pd.DataFrame({
        "Step": ["1. Define / Upload", "2. Train or Reuse", "3. Generate", "4. Evaluate", "5. Download"],
        "Description": [
            "Create a schema or upload a reference CSV",
            "Reuse a saved model or train a new CTGAN model",
            "Generate synthetic records",
            "Compare distributions, correlations and similarity",
            "Export datasets and evaluation tables",
        ],
    })
    st.table(workflow)

    st.subheader("📌 Current Session")
    if "current_synthetic_data" in st.session_state:
        display_pair_summary(
            st.session_state.current_real_data,
            st.session_state.current_synthetic_data,
        )
        st.success(f"Active run: {st.session_state.get('current_run_name', 'Unnamed run')}")
        st.caption(f"Mode: {st.session_state.get('current_run_type', 'Unknown')}")
    else:
        st.info("No active generated dataset in this session. Use AI Generator or Custom Dataset Builder.")

    if real_data is not None:
        st.subheader("📋 Reference Dataset Preview")
        st.dataframe(real_data.head(10), use_container_width=True)
    elif error:
        st.warning(error)

    if models:
        st.subheader("📚 Available Models")
        st.dataframe(pd.DataFrame(models), use_container_width=True)


# ============================================================
# ============================================================
# INTELLIGENT PROFILER AND UNIFIED EXPERIMENT
# ============================================================

elif page == "Profiler & Experiment":
    st.header("Intelligent Dataset Profiler & Experiment")
    uploaded = st.file_uploader("Upload a reference CSV", type=["csv"], key="profiler_upload")
    if uploaded is None:
        st.info("Upload a CSV to profile, generate, and evaluate synthetic data.")
        st.stop()
    try:
        data = read_uploaded_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        st.stop()
    fingerprint = file_fingerprint(uploaded)
    if st.session_state.get("profile_fingerprint") != fingerprint:
        st.session_state.profile_fingerprint = fingerprint
        st.session_state.profile_result = profile_dataset(data)
        st.session_state.pop("experiment_report", None)
    profile = st.session_state.profile_result; overview = profile["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['rows']:,}"); c2.metric("Columns", overview["columns"]); c3.metric("Memory", f"{overview['memory_bytes'] / 1024:.1f} KB"); c4.metric("Duplicates", overview["duplicate_rows"])
    st.subheader("Preview"); st.dataframe(data.head(50), use_container_width=True)
    st.subheader("Column profile"); st.dataframe(pd.DataFrame(profile["columns"]), use_container_width=True)
    missing_chart = pd.DataFrame({"Column": [x["column"] for x in profile["columns"]], "Missing %": [x["missing_percent"] for x in profile["columns"]]}).set_index("Column")
    if not missing_chart.empty and float(missing_chart["Missing %"].max()) > 0: st.bar_chart(missing_chart)
    for warning in profile["warnings"]:
        (st.warning if warning["type"] == "heuristic" else st.info)(f"{warning['column']}: {warning['message']}")
    if profile["numerical"]:
        st.subheader("Numerical statistics and outliers"); st.dataframe(pd.DataFrame(profile["numerical"]), use_container_width=True)
        if profile["outliers"]: st.dataframe(pd.DataFrame(profile["outliers"]), use_container_width=True)
    if profile["categorical"]:
        selected_cat = st.selectbox("Categorical column", [x["column"] for x in profile["categorical"]], key="profile_category")
        selected = next(x for x in profile["categorical"] if x["column"] == selected_cat); st.dataframe(pd.DataFrame(selected["values"]), use_container_width=True)
    st.subheader("Generation")
    method = st.selectbox("Method", ["Custom Generator", "Gaussian Copula", "CTGAN"], key="experiment_method")
    rows = st.number_input("Synthetic rows", min_value=1, max_value=100000, value=min(1000, max(1, len(data))), key="experiment_rows")
    seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, key="experiment_seed")
    epochs = st.number_input("CTGAN epochs", min_value=10, max_value=2000, value=300, step=10, key="experiment_epochs") if method == "CTGAN" else 300
    if st.button("Generate and evaluate", type="primary", key="run_experiment"):
        try:
            model_path = str(MODEL_DIR / f"experiment_{dataframe_fingerprint(data)}.pkl") if method == "CTGAN" else None
            with st.spinner(f"Generating with {method}..."):
                synthetic, generation_metadata = generate_synthetic(data, method, int(rows), seed=int(seed), model_path=model_path, epochs=int(epochs))
                report = add_sdv_quality(evaluate_datasets(data, synthetic, metadata=generation_metadata), data, synthetic)
            set_current_run(data, synthetic, "Profiler experiment", method, model_path, generation_metadata); st.session_state.experiment_report = report; st.session_state.experiment_synthetic = synthetic
            if model_path and Path(model_path).exists(): register_model(Path(model_path), {"columns": list(data.columns), "rows": len(data), "epochs": int(epochs)})
            save_generated_csv(synthetic, f"experiment_{method.lower().replace(' ', '_')}"); st.success(f"Generated {len(synthetic):,} rows.")
        except Exception as error: st.error(f"Generation or evaluation failed: {error}")
    if "experiment_synthetic" in st.session_state:
        st.subheader("Generated preview"); st.dataframe(st.session_state.experiment_synthetic.head(50), use_container_width=True); render_download(st.session_state.experiment_synthetic, "synthetic_experiment.csv")
    if "experiment_report" in st.session_state:
        report = st.session_state.experiment_report; st.subheader("Unified evaluation report"); st.json({"schema": report["schema"], "warnings": report["warnings"], "metadata": report["metadata"], "sdv_quality": report["sdv_quality"]})
        if report["numerical_fidelity"]: st.dataframe(pd.DataFrame(report["numerical_fidelity"]), use_container_width=True)
        if report["categorical_fidelity"]: st.dataframe(pd.DataFrame(report["categorical_fidelity"]), use_container_width=True)
        if report["numerical_fidelity"]: st.bar_chart(pd.DataFrame(report["numerical_fidelity"]).set_index("column")["normalized_wasserstein"])
        if report["categorical_fidelity"]: st.bar_chart(pd.DataFrame(report["categorical_fidelity"]).set_index("column")["total_variation_distance"])
        st.download_button("Download evaluation JSON", report_json(report), "evaluation_report.json", "application/json")

# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "📊 Dataset Analysis":
    st.header("📊 Dataset Analysis")
    uploaded = st.file_uploader("Upload CSV for analysis", type=["csv"], key="analysis_upload")

    if uploaded is not None:
        try:
            data = read_uploaded_csv(uploaded)
            st.success("Dataset loaded successfully.")
        except Exception as error:
            st.error(f"Could not read the CSV: {error}")
            st.stop()
    else:
        # Fallback to default dataset if it exists in your DATA_DIR
        data, error = read_csv_safely(DATA_DIR / "telco_cleaned.csv")
        if data is None:
            st.info("Upload a CSV file to start analysis.")
            st.stop()
        st.info("Showing the default project dataset. Upload another CSV to analyze it.")

    # Top Level Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", len(data.columns))
    c3.metric("Missing Values", int(data.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(data.duplicated().sum()))

    # Organize the analysis into easy-to-read tabs
    tab_overview, tab_numeric, tab_categorical, tab_correlation = st.tabs([
        "📋 Overview", "📈 Numerical Analysis", "📊 Categorical Analysis", "🔗 Correlation"
    ])

    # ------------------------------------------------------------
    # TAB 1: OVERVIEW & MISSING DATA
    # ------------------------------------------------------------
    with tab_overview:
        st.subheader("Data Preview")
        st.dataframe(data.head(50), use_container_width=True)

        st.subheader("Column Information")
        info = pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values,
            "Missing Values": [int(data[column].isna().sum()) for column in data.columns],
            "Unique Values": [int(data[column].nunique(dropna=False)) for column in data.columns],
            "Missing %": [round(data[column].isna().sum() / len(data) * 100, 2) for column in data.columns]
        })
        st.dataframe(info, use_container_width=True)

        # Missing Values Visualizer
        if data.isna().sum().sum() > 0:
            st.subheader("Missing Values Distribution")
            missing_data = data.isna().sum().reset_index()
            missing_data.columns = ["Column", "Missing Count"]
            missing_data = missing_data[missing_data["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
            
            fig_missing = px.bar(
                missing_data, x="Column", y="Missing Count",
                title="Count of Missing Values per Column",
                color="Missing Count", color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✨ This dataset has no missing values!")

    # ------------------------------------------------------------
    # TAB 2: NUMERICAL ANALYSIS
    # ------------------------------------------------------------
    with tab_numeric:
        numerical = data.select_dtypes(include=[np.number])
        if numerical.empty:
            st.info("No numerical columns found in this dataset.")
        else:
            st.subheader("Numerical Statistics")
            st.dataframe(numerical.describe().T, use_container_width=True)

            st.subheader("Interactive Distribution Chart")
            selected_num = st.selectbox("Choose a numerical column", numerical.columns, key="num_col_select")
            
            # Advanced Plotly Histogram with Boxplot Margins
            fig_num = px.histogram(
                data, 
                x=selected_num, 
                title=f"Distribution of {selected_num}",
                marginal="box",  # Adds a boxplot on top
                color_discrete_sequence=['#1f77b4'],
                opacity=0.8
            )
            fig_num.update_layout(
                xaxis_title=selected_num,
                yaxis_title="Frequency",
                bargap=0.1
            )
            st.plotly_chart(fig_num, use_container_width=True)

    # ------------------------------------------------------------
    # TAB 3: CATEGORICAL ANALYSIS
    # ------------------------------------------------------------
    with tab_categorical:
        categorical = categorical_columns(data)
        if not categorical:
            st.info("No categorical columns found in this dataset.")
        else:
            st.subheader("Categorical Distribution")
            selected_cat = st.selectbox("Choose a categorical column", categorical, key="cat_col_select")
            
            # Calculate value counts
            counts = data[selected_cat].fillna("<MISSING>").astype(str).value_counts().reset_index()
            counts.columns = ["Category", "Count"]
            
            # Limit to top 20 for readability
            if len(counts) > 20:
                st.info(f"Showing top 20 categories out of {len(counts)} for {selected_cat}.")
                counts = counts.head(20)

            # Advanced Plotly Bar Chart
            fig_cat = px.bar(
                counts, 
                x="Category", 
                y="Count",
                title=f"Top Categories in {selected_cat}",
                color="Count",
                color_continuous_scale="Blues",
                text="Count"
            )
            fig_cat.update_traces(texttemplate='%{text}', textposition='outside')
            fig_cat.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cat, use_container_width=True)
            
            st.dataframe(counts.set_index("Category"), use_container_width=True)

    # ------------------------------------------------------------
    # TAB 4: CORRELATION MATRIX
    # ------------------------------------------------------------
    with tab_correlation:
        numerical = data.select_dtypes(include=[np.number])
        if len(numerical.columns) < 2:
            st.info("At least two numerical columns are required for a correlation matrix.")
        else:
            st.subheader("Correlation Heatmap")
            st.write("This shows how strongly numerical columns are related to each other.")
            
            corr_matrix = numerical.corr()
            
            # Advanced Plotly Heatmap
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",  # Display values inside the boxes
                aspect="auto",
                color_continuous_scale="RdBu_r", # Red for negative, Blue for positive
                zmin=-1, zmax=1, # Fix scale from -1 to 1
                title="Pearson Correlation Matrix"
            )
            fig_corr.update_layout(height=max(500, len(corr_matrix) * 50))
            st.plotly_chart(fig_corr, use_container_width=True)

# ============================================================
# CUSTOM DATASET BUILDER
# ============================================================

elif page == "🛠️ Custom Dataset Builder":
    st.header("🛠️ Custom Dataset Builder")
    st.write("Create a synthetic dataset without a reference CSV. This mode uses user-defined rules rather than CTGAN.")

    dataset_name = st.text_input("Dataset Name", "custom_dataset", key="custom_dataset_name")
    number_of_rows = st.number_input("Number of Rows", min_value=1, max_value=100000, value=1000, step=100, key="custom_rows")

    if "custom_columns" not in st.session_state:
        st.session_state.custom_columns = [{"name": "Age", "type": "Integer", "min": 18, "max": 65}]

    data_types = ["Integer", "Decimal", "Category", "Boolean", "Name", "Email", "Date", "Unique ID"]
    st.subheader("Define Columns")

    if st.button("📅 Add Column", key="add_custom_column"):
        index = len(st.session_state.custom_columns) + 1
        st.session_state.custom_columns.append({"name": f"Column_{index}", "type": "Integer", "min": 0, "max": 100})
        st.rerun()

    remove_index = None
    for index, column in enumerate(st.session_state.custom_columns):
        with st.expander(f"Column {index + 1}: {column.get('name', '')}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                column["name"] = st.text_input("Column Name", column.get("name", f"Column_{index + 1}"), key=f"custom_name_{index}")
            with c2:
                selected_type = st.selectbox("Data Type", data_types, index=data_types.index(column.get("type", "Integer")), key=f"custom_type_{index}")
                column["type"] = selected_type

            if selected_type in ["Integer", "Decimal"]:
                c1, c2 = st.columns(2)
                with c1:
                    column["min"] = st.number_input("Minimum", value=float(column.get("min", 0)), key=f"custom_min_{index}")
                with c2:
                    column["max"] = st.number_input("Maximum", value=float(column.get("max", 100)), key=f"custom_max_{index}")
            elif selected_type == "Category":
                current = ", ".join(column.get("categories", ["A", "B", "C"]))
                text = st.text_input("Categories separated by commas", current, key=f"custom_categories_{index}")
                column["categories"] = [item.strip() for item in text.split(",") if item.strip()]
            elif selected_type == "Date":
                c1, c2 = st.columns(2)
                with c1:
                    column["start_date"] = st.date_input("Start Date", pd.Timestamp(column.get("start_date", "2020-01-01")).date(), key=f"custom_start_{index}")
                with c2:
                    column["end_date"] = st.date_input("End Date", pd.Timestamp(column.get("end_date", "2025-12-31")).date(), key=f"custom_end_{index}")

            if st.button("🗑️ Remove Column", key=f"remove_custom_{index}"):
                remove_index = index

    if remove_index is not None:
        st.session_state.custom_columns.pop(remove_index)
        st.rerun()

    if st.button("🚀 Generate Custom Dataset", type="primary", key="generate_custom"):
        try:
            generated = generate_custom_dataset(st.session_state.custom_columns, int(number_of_rows))
            st.session_state.custom_generated_data = generated
            st.session_state.custom_generated_data = generated
            st.session_state.current_synthetic_data = generated.copy()
            st.session_state.current_run_name = dataset_name
            st.session_state.current_run_type = "Custom Builder"
            path = save_generated_csv(generated, dataset_name)
            st.success(f"Generated {len(generated):,} rows and saved {path.name}.")
        except Exception as error:
            st.error(f"Generation failed: {error}")

    if "custom_generated_data" in st.session_state:
        generated = st.session_state.custom_generated_data
        st.subheader("Generated Dataset")
        st.dataframe(generated.head(50), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(generated):,}")
        c2.metric("Columns", len(generated.columns))
        render_download(generated, f"{clean_filename(dataset_name)}.csv")


# ============================================================
# AI GENERATOR
# ============================================================

elif page == "🤖 AI Generator":
    st.header("🤖 AI Generator ” CTGAN")
    st.write("Reuse a saved model or train a new model from a reference CSV.")

    models = discover_models()
    tab_existing, tab_train = st.tabs(["📚 Use Existing Model", "🧠 Train New Model"])

    with tab_existing:
        if not models:
            st.info("No saved .pkl models were found in the models folder.")
        else:
            model_labels = [f"{item['name']} | {item['modified']} | {item['size_mb']} MB" for item in models]
            selected_label = st.selectbox("Choose a saved model", model_labels, key="existing_model_choice")
            selected_model = models[model_labels.index(selected_label)]
            st.write(f"**Model path:** `{selected_model['path']}`")
            st.write(f"**Training dataset:** {selected_model.get('dataset_name', 'Unknown')}")
            st.write(f"**Known columns:** {', '.join(selected_model.get('columns', [])) or 'Not recorded'}")
            rows = st.number_input("Synthetic rows", min_value=1, max_value=100000, value=1000, step=100, key="existing_rows")
            reference_upload = st.file_uploader("Optional reference CSV for evaluation", type=["csv"], key="existing_reference")

            if st.button("🚀 Load Model and Generate", type="primary", key="load_generate"):
                try:
                    synthesizer = load_ctgan_model(selected_model["path"])
                    synthetic = generate_ctgan_data(synthesizer, int(rows))
                    st.session_state.ctgan_synthesizer = synthesizer
                    st.session_state.ctgan_model_path = selected_model["path"]
                    real = None
                    if reference_upload is not None:
                        real = read_uploaded_csv(reference_upload)
                    if real is not None:
                        common = [c for c in real.columns if c in synthetic.columns]
                        if common:
                            real_for_eval = real[common].copy()
                            synthetic_for_eval = synthetic[common].copy()
                            set_current_run(real_for_eval, synthetic_for_eval, selected_model["name"], "Existing CTGAN Model", selected_model["path"])
                    st.session_state.ctgan_generated_data = synthetic
                    path = save_generated_csv(synthetic, f"{selected_model['name']}_generated")
                    st.success(f"Generated {len(synthetic):,} rows. Saved as {path.name}.")
                except Exception as error:
                    st.error(f"Could not load or generate: {error}")

    with tab_train:
        uploaded = st.file_uploader("Upload reference CSV", type=["csv"], key="train_reference")
        if uploaded is not None:
            try:
                reference_data = read_uploaded_csv(uploaded)
                st.success(f"Loaded {len(reference_data):,} rows and {len(reference_data.columns)} columns.")
                st.subheader("Reference Dataset Preview")
                st.dataframe(reference_data.head(10), use_container_width=True)

                selected_columns = st.multiselect(
                    "Columns to use for training",
                    list(reference_data.columns),
                    default=list(reference_data.columns),
                    key="train_columns",
                )

                if selected_columns:
                    selected_reference_data = select_reference_columns(
                        reference_data,
                        selected_columns,
                    )

                    st.write(f"Selected columns: {len(selected_columns)}")
                    st.caption(
                        "The preview below updates to show only the selected training columns."
                    )
                    st.dataframe(
                        selected_reference_data.head(10),
                        use_container_width=True,
                    )

                    st.info(
                        "Automatic metadata detection is enabled. "
                        "Advanced manual metadata settings have been removed "
                        "to prevent invalid PII/type combinations."
                    )

                    epochs = st.number_input(
                        "Training epochs",
                        min_value=10,
                        max_value=2000,
                        value=300,
                        step=50,
                        key="train_epochs",
                    )
                    model_name = clean_filename(
                        st.text_input(
                            "Model name",
                            "custom_ctgan",
                            key="train_model_name",
                        )
                    )
                    model_path = MODEL_DIR / f"{model_name}.pkl"
                    rows = st.number_input(
                        "Synthetic rows after training",
                        min_value=1,
                        max_value=100000,
                        value=1000,
                        step=100,
                        key="train_rows",
                    )

                    if model_path.exists():
                        st.warning(
                            "A model with this name already exists and will be replaced if you train."
                        )

                    if st.button(
                        "⚙️ Train, Save and Generate",
                        type="primary",
                        key="train_generate",
                    ):
                        try:
                            with st.spinner(
                                "Training CTGAN. This may take several minutes..."
                            ):
                                synthesizer, metadata = train_ctgan_model(
                                    data=selected_reference_data,
                                    model_path=str(model_path),
                                    epochs=int(epochs),
                                )
                                synthetic = generate_ctgan_data(
                                    synthesizer,
                                    int(rows),
                                )

                            register_model(
                                model_path,
                                {
                                    "dataset_name": uploaded.name,
                                    "columns": selected_columns,
                                    "rows": len(selected_reference_data),
                                    "epochs": int(epochs),
                                },
                            )

                            st.session_state.ctgan_synthesizer = synthesizer
                            st.session_state.ctgan_model_path = str(model_path)
                            st.session_state.ctgan_generated_data = synthetic
                            set_current_run(
                                selected_reference_data,
                                synthetic,
                                model_name,
                                "New CTGAN Model",
                                model_path,
                                metadata,
                            )
                            output_path = save_generated_csv(
                                synthetic,
                                f"{model_name}_generated",
                            )
                            st.success(
                                f"Model saved and {len(synthetic):,} rows generated. "
                                f"Output: {output_path.name}"
                            )
                        except Exception as error:
                            st.error(f"Training failed: {error}")
                else:
                    st.warning("Select at least one column.")
            except Exception as error:
                st.error(f"Could not read the CSV: {error}")
        else:
            st.info("Upload a reference CSV to train a new CTGAN model.")

    if "ctgan_generated_data" in st.session_state:
        st.divider()
        st.subheader("Latest Generated Data")
        generated = st.session_state.ctgan_generated_data
        st.dataframe(generated.head(50), use_container_width=True)
        render_download(generated, "synthetic_data.csv")


# ============================================================
# MODEL LIBRARY
# ============================================================

elif page == "📚 Model Library":
    st.header("📚 Model Library")
    st.write("Manage, inspect, and reuse trained CTGAN models. Files are stored locally in the `models/` directory.")
    
    models = discover_models()
    
    if not models:
        st.info("No saved models found. Train a new model in the AI Generator to get started.")
        st.stop()

    # ------------------------------------------------------------
    # 1. TOP KPI METRICS
    # ------------------------------------------------------------
    total_models = len(models)
    total_size = sum(item["size_mb"] for item in models)
    latest_model = sorted(models, key=lambda x: x["modified"], reverse=True)[0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Models", total_models)
    c2.metric("Total Storage Used", f"{total_size:.2f} MB")
    c3.metric("Latest Model", latest_model["name"], latest_model["modified"])
    
    st.divider()

    # ------------------------------------------------------------
    # 2. SEARCH & SORT CONTROLS
    # ------------------------------------------------------------
    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Search models by name or dataset", placeholder="e.g., 'telco' or 'ctgan'").lower()
    with col_sort:
        sort_by = st.selectbox("Sort by", ["Date (Newest)", "Name (A-Z)", "Size (Largest)"])

    # Filtering
    filtered_models = [
        m for m in models 
        if search_query in m["name"].lower() 
        or search_query in m.get("dataset_name", "").lower()
    ]

    # Sorting
    if sort_by == "Date (Newest)":
        filtered_models = sorted(filtered_models, key=lambda x: x["modified"], reverse=True)
    elif sort_by == "Name (A-Z)":
        filtered_models = sorted(filtered_models, key=lambda x: x["name"].lower())
    elif sort_by == "Size (Largest)":
        filtered_models = sorted(filtered_models, key=lambda x: x["size_mb"], reverse=True)

    if not filtered_models:
        st.warning("No models match your search criteria.")
        st.stop()

    # ------------------------------------------------------------
    # 3. MODEL LIST (USING EXPANDERS FOR CLEAN UI)
    # ------------------------------------------------------------
    for model in filtered_models:
        # Fallbacks for missing metadata
        dataset_name = model.get("dataset_name", "Unknown")
        rows = model.get("rows", "Unknown")
        epochs = model.get("epochs", "Unknown")
        cols = model.get("columns", [])
        
        # Expander Header
        expander_label = f"🤖 **{model['name']}** | {model['size_mb']} MB | Dataset: {dataset_name} | Updated: {model['modified']}"
        
        with st.expander(expander_label):
            # Metadata Grid
            m1, m2, m3 = st.columns(3)
            m1.info(f"**Training Rows:** {rows:,}" if isinstance(rows, int) else f"**Training Rows:** {rows}")
            m2.info(f"**Epochs:** {epochs}")
            m3.info(f"**File Size:** {model['size_mb']} MB")
            
            # Path & Columns
            st.markdown("**File Path**")
            st.code(model["path"], language="text")
            
            st.markdown("**Trained Columns**")
            if cols:
                # Display as a formatted comma-separated string inside a code block
                st.code(", ".join(cols), language="text", wrap_lines=True)
            else:
                st.caption("Column metadata not recorded in the model catalog.")
            
            st.divider()
            
            # --------------------------------------------------------
            # SAFE DELETION WORKFLOW (SENIOR DEV PRACTICE)
            # --------------------------------------------------------
            with st.popover("🗑️ Delete Model"):
                st.warning(f"Are you sure you want to permanently delete **{model['name']}**?")
                st.caption("This will remove the `.pkl` file from your disk and delete it from the catalog.")
                
                # Require explicit confirmation
                confirm = st.checkbox("Yes, I want to delete this model.", key=f"confirm_del_{model['name']}")
                
                if st.button("Permanently Delete", type="primary", disabled=not confirm, key=f"btn_del_{model['name']}"):
                    try:
                        # 1. Delete the physical file
                        if os.path.exists(model["path"]):
                            os.remove(model["path"])
                        
                        # 2. Remove from catalog.json
                        catalog = load_catalog()
                        catalog = [item for item in catalog if item.get("path") != model["path"]]
                        save_catalog(catalog)
                        
                        st.success(f"Model `{model['name']}` deleted successfully.")
                        time.sleep(1.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete model. Error: {e}")

    st.caption("💡 **Developer Tip:** To add an external model, place a compatible SDV CTGAN `.pkl` file directly into the `models/` folder and click 'Rerun' in the top right.")

# ============================================================
# DYNAMIC EVALUATION
# ============================================================

elif page == "📈 Dynamic Evaluation" or page == "🆚 Dynamic Evaluation":
    st.header("📈 Dynamic Evaluation")
    st.write("Evaluation is calculated from the active real and synthetic datasets. No fixed Telco scores are displayed.")

    real = st.session_state.get("current_real_data")
    synthetic = st.session_state.get("current_synthetic_data")

    if real is None or synthetic is None:
        st.info("Generate data first, or upload both datasets below.")
        c1, c2 = st.columns(2)
        with c1:
            real_upload = st.file_uploader("Real reference CSV", type=["csv"], key="eval_real_upload")
        with c2:
            synthetic_upload = st.file_uploader("Synthetic CSV", type=["csv"], key="eval_synth_upload")
        if real_upload is not None and synthetic_upload is not None:
            try:
                real = read_uploaded_csv(real_upload)
                synthetic = read_uploaded_csv(synthetic_upload)
                common = comparable_columns(real, synthetic)
                real = real[common].copy()
                synthetic = synthetic[common].copy()
                set_current_run(real, synthetic, "Uploaded Evaluation", "Manual Evaluation")
            except Exception as error:
                st.error(f"Could not load evaluation datasets: {error}")
                st.stop()
        else:
            st.stop()

    display_pair_summary(real, synthetic)
    st.caption(f"Active run: {st.session_state.get('current_run_name', 'Unnamed')}")

    st.subheader("🧪 SDV Quality Evaluation")
    sdv_result, sdv_error = run_sdv_quality(real, synthetic)
    if sdv_result is not None:
        st.metric("SDV Overall Quality", f"{sdv_result['overall'] * 100:.2f}%")
        
        # Advanced Plotly chart for SDV Properties
        props = sdv_result["properties"]
        if isinstance(props, pd.DataFrame):
            fig_props = px.bar(
                props, 
                x="Property", 
                y="Score", 
                title="SDV Quality Properties Score",
                color="Score",
                color_continuous_scale="Tealgrn",
                text="Score"
            )
            fig_props.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_props.update_layout(yaxis_range=[0, 1.1])
            st.plotly_chart(fig_props, use_container_width=True)
        else:
            st.dataframe(props, use_container_width=True)
    else:
        st.warning(sdv_error or "SDV quality evaluation could not be completed.")

    st.subheader("📊 Numerical Comparison")
    num = numerical_comparison(real, synthetic)
    if num.empty:
        st.info("No comparable numerical columns were found.")
    else:
        st.dataframe(num, use_container_width=True)
        
        # Advanced Horizontal Bar Chart
        fig_num = px.bar(
            num.sort_values("Distribution_Similarity_Percent", ascending=True),
            x="Distribution_Similarity_Percent",
            y="Column",
            orientation="h",
            title="Numerical Columns: Distribution Similarity (%)",
            color="Distribution_Similarity_Percent",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            text="Distribution_Similarity_Percent"
        )
        fig_num.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_num.update_layout(xaxis_range=[0, 105], height=max(400, len(num) * 45))
        st.plotly_chart(fig_num, use_container_width=True)
        
        render_download(num, "numerical_comparison_dynamic.csv", "📥 Download Numerical Comparison")

    st.subheader("📊 Categorical Comparison")
    cat = categorical_comparison(real, synthetic)
    if cat.empty:
        st.info("No comparable categorical columns were found.")
    else:
        st.dataframe(cat, use_container_width=True)
        
        # Advanced Horizontal Bar Chart
        fig_cat = px.bar(
            cat.sort_values("Distribution_Similarity_Percent", ascending=True),
            x="Distribution_Similarity_Percent",
            y="Column",
            orientation="h",
            title="Categorical Columns: Distribution Similarity (%)",
            color="Distribution_Similarity_Percent",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            text="Distribution_Similarity_Percent"
        )
        fig_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_cat.update_layout(xaxis_range=[0, 105], height=max(400, len(cat) * 45))
        st.plotly_chart(fig_cat, use_container_width=True)
        
        render_download(cat, "categorical_comparison_dynamic.csv", "📥 Download Categorical Comparison")

    st.subheader("🔗 Correlation Difference")
    corr_difference, average_difference = correlation_comparison(real, synthetic)
    if corr_difference.empty:
        st.info("At least two comparable numerical columns are required for correlation analysis.")
    else:
        st.metric("Average Absolute Correlation Difference", f"{average_difference:.4f}")
        
        # Advanced Heatmap Chart
        fig_heat = px.imshow(
            corr_difference,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="Reds",
            title="Absolute Correlation Difference Heatmap (Real vs Synthetic)"
        )
        fig_heat.update_layout(height=max(400, len(corr_difference) * 60))
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.dataframe(corr_difference, use_container_width=True)
        render_download(corr_difference.reset_index(), "correlation_difference_dynamic.csv", "📥 Download Correlation Difference")

    st.subheader("📈 Dynamic Distribution Graph")
    common_numeric = [c for c in comparable_columns(real, synthetic) if c in numeric_columns(real) and c in numeric_columns(synthetic)]
    if common_numeric:
        selected = st.selectbox("Choose a numerical column", common_numeric, key="eval_numeric_column")
        
        # Overlaid Plotly Histogram (Probability Density)
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=real[selected].dropna(),
            name="Real Data",
            opacity=0.65,
            histnorm="probability density",
            marker_color="#1f77b4", # Blue
            nbinsx=30
        ))
        
        fig_dist.add_trace(go.Histogram(
            x=synthetic[selected].dropna(),
            name="Synthetic Data",
            opacity=0.65,
            histnorm="probability density",
            marker_color="#ff7f0e", # Orange
            nbinsx=30
        ))
        
        fig_dist.update_layout(
            title=f"Probability Density Distribution: {selected}",
            barmode="overlay",
            xaxis_title=selected,
            yaxis_title="Probability Density",
            legend_title="Dataset",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No numerical columns available for distribution comparison.")

    st.subheader("💾 Save Current Evaluation")
    if st.button("Save Evaluation Tables", key="save_dynamic_eval"):
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        num.to_csv(EVALUATION_DIR / "numerical_comparison_dynamic.csv", index=False)
        cat.to_csv(EVALUATION_DIR / "categorical_comparison_dynamic.csv", index=False)
        if not corr_difference.empty:
            corr_difference.to_csv(EVALUATION_DIR / "correlation_difference_dynamic.csv")
        st.success(f"Evaluation tables saved in {EVALUATION_DIR}.")
        
# ============================================================
# PRIVACY SCREENING
# ============================================================

elif page == "🔐 Privacy Screening":
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.neighbors import NearestNeighbors

    st.header("🔐 Privacy Screening")
    st.write("Advanced privacy evaluation. This tool checks for direct data copying (memorization) and calculates Distance to Closest Record (DCR).")

    real = st.session_state.get("current_real_data")
    synthetic = st.session_state.get("current_synthetic_data")

    if real is None or synthetic is None:
        st.info("Generate data first or upload real and synthetic CSV files.")
        c1, c2 = st.columns(2)
        with c1:
            real_upload = st.file_uploader("Real CSV", type=["csv"], key="privacy_real")
        with c2:
            synthetic_upload = st.file_uploader("Synthetic CSV", type=["csv"], key="privacy_synthetic")
        if real_upload is not None and synthetic_upload is not None:
            try:
                real = read_uploaded_csv(real_upload)
                synthetic = read_uploaded_csv(synthetic_upload)
                common = comparable_columns(real, synthetic)
                real = real[common].copy()
                synthetic = synthetic[common].copy()
                set_current_run(real, synthetic, "Privacy Upload", "Manual Privacy Screening")
            except Exception as e:
                st.error(f"Error loading files: {e}")
                st.stop()
        else:
            st.stop()

    # ------------------------------------------------------------
    # DATA PREPARATION FOR DCR
    # ------------------------------------------------------------
    with st.spinner("Calculating privacy metrics... This may take a moment."):
        # Get common columns and drop ones with too many missing values for calculation
        common_cols = [c for c in real.columns if c in synthetic.columns]
        eval_real = real[common_cols].copy()
        eval_synth = synthetic[common_cols].copy()

        # Fill missing values for the distance calculation
        eval_real = eval_real.fillna(eval_real.median(numeric_only=True)).fillna("MISSING")
        eval_synth = eval_synth.fillna(eval_synth.median(numeric_only=True)).fillna("MISSING")

        # Separate numeric and categorical
        num_cols = eval_real.select_dtypes(include=[np.number]).columns
        cat_cols = eval_real.select_dtypes(exclude=[np.number]).columns

        # Preprocess: Scale numerical, Encode categorical
        scaler = StandardScaler()
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        real_processed = pd.DataFrame(index=eval_real.index)
        synth_processed = pd.DataFrame(index=eval_synth.index)

        if len(num_cols) > 0:
            real_processed[num_cols] = scaler.fit_transform(eval_real[num_cols])
            synth_processed[num_cols] = scaler.transform(eval_synth[num_cols])

        if len(cat_cols) > 0:
            real_processed[cat_cols] = encoder.fit_transform(eval_real[cat_cols])
            synth_processed[cat_cols] = encoder.transform(eval_synth[cat_cols])

        # Calculate Distance to Closest Record (DCR)
        # For large datasets, we must sample to prevent memory crashes
        max_sample = 5000
        if len(real_processed) > max_sample:
            real_sample = real_processed.sample(max_sample, random_state=42)
            st.caption(f"Note: Real data sampled down to {max_sample} rows for performance.")
        else:
            real_sample = real_processed

        if len(synth_processed) > max_sample:
            synth_sample = synth_processed.sample(max_sample, random_state=42)
        else:
            synth_sample = synth_processed

        # Fit NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(real_sample)
        distances, _ = nn.kneighbors(synth_sample)
        
        dcr_values = distances.flatten()
        exact_matches = (dcr_values == 0).sum()
        mean_dcr = np.mean(dcr_values)
        min_dcr = np.min(dcr_values)

    # ------------------------------------------------------------
    # PRIVACY RISK SCORE CALCULATION
    # ------------------------------------------------------------
    # Base score starts at 100% privacy (Low Risk)
    # Deduct points if DCR is very low or exact matches exist
    exact_match_pct = (exact_matches / len(synth_sample)) * 100
    
    risk_score = 100
    risk_score -= exact_match_pct * 5  # Heavy penalty for exact copies
    risk_score -= max(0, (0.5 - mean_dcr) * 100)  # Penalty if distances are dangerously small
    risk_score = max(0, min(100, risk_score)) # Clamp between 0-100

    if risk_score >= 85:
        risk_label = "🟢 Low Risk (Safe)"
        risk_color = "normal" # Streamlit green
    elif risk_score >= 60:
        risk_label = "🟡 Moderate Risk"
        risk_color = "off"
    else:
        risk_label = "🔴 High Risk (Memorization Detected)"
        risk_color = "inverse"

    # ------------------------------------------------------------
    # UI LAYOUT
    # ------------------------------------------------------------
    st.divider()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real Rows", f"{len(real):,}")
    c2.metric("Synthetic Rows", f"{len(synthetic):,}")
    c3.metric("Exact Matches", f"{exact_matches:,}")
    c4.metric("Privacy Score", f"{risk_score:.1f}%", risk_label, delta_color=risk_color)

    if exact_matches > 0:
        st.error(f"⚠️ **Memorization Detected:** {exact_matches} synthetic rows are exact duplicates of real rows.")
    else:
        st.success("✅ No exact row duplicates found.")

    # Tabs for Advanced Analytics
    tab_dcr, tab_columns, tab_limitations = st.tabs([
        "📊 Distance to Closest Record", 
        "⚠️ Column Vulnerability", 
        "📖 Methodology & Limits"
    ])

    with tab_dcr:
        st.subheader("Distance to Closest Record (DCR) Distribution")
        st.write("DCR measures how mathematically close a synthetic row is to its nearest real neighbor. A distance of `0.0` means it's an exact match. Higher distances imply better privacy.")
        
        # Plotly Histogram for DCR
        fig_dcr = px.histogram(
            x=dcr_values, 
            nbins=50, 
            title="Distribution of DCR Values",
            labels={'x': 'Euclidean Distance to Closest Real Record', 'y': 'Synthetic Record Count'},
            color_discrete_sequence=['#636EFA']
        )
        # Add a vertical line for the mean
        fig_dcr.add_vline(x=mean_dcr, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_dcr:.3f}")
        st.plotly_chart(fig_dcr, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean DCR", f"{mean_dcr:.4f}")
        m2.metric("Min DCR", f"{min_dcr:.4f}")
        m3.metric("Median DCR", f"{np.median(dcr_values):.4f}")

    with tab_columns:
        st.subheader("Column-Level Privacy Risk")
        st.write("Identifies which specific columns have the highest rate of exact matches or near-matches.")
        
        # Calculate exact matches per column
        col_matches = []
        for col in common_cols:
            # Check if the exact value exists in the real dataset
            synth_vals = synth_sample.index.map(lambda idx: synthetic.loc[idx, col]) # align with sample
            real_vals_set = set(real[col].dropna().astype(str))
            matches_in_col = sum(str(val) in real_vals_set for val in synth_vals)
            col_matches.append({"Column": col, "Matches": matches_in_col, "Percentage": (matches_in_col / len(synth_sample))*100})
        
        col_df = pd.DataFrame(col_matches).sort_values("Percentage", ascending=False)
        
        fig_cols = px.bar(
            col_df.head(15), 
            x="Percentage", 
            y="Column", 
            orientation='h',
            title="Top 15 Columns with Exact Value Matches",
            color="Percentage",
            color_continuous_scale="Reds",
            text_auto=".1f"
        )
        fig_cols.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_cols, use_container_width=True)

    with tab_limitations:
        st.subheader("Screening Limitations & Methodology")
        st.warning(
            "**Important:** This tool provides a statistical estimation of privacy risks "
            "(specifically related to overfitting and memorization). It does **not** guarantee differential privacy."
        )
        st.markdown("""
        * **How DCR works:** We normalize all numerical columns (mean=0, std=1) and encode categorical columns. We then calculate the Euclidean distance between every synthetic row and its nearest real row. 
        * **Exact Matches:** If the DCR is exactly `0.0`, the synthetic row is an exact copy of a real row.
        * **Near Matches:** If the distance is very small (e.g., `< 0.1`), the synthetic data is likely overfitted and may leak sensitive individual information.
        * **Why this matters:** If a model outputs data that is too close to the training data, an attacker could re-identify individuals. To fix this, you generally need to decrease training epochs, add noise (Differential Privacy), or use a different synthesizer.
        """)

    # ------------------------------------------------------------
    # EXPORT REPORT
    # ------------------------------------------------------------
    st.divider()
    if st.button("💾 Export Privacy Report"):
        report_data = pd.DataFrame({
            "Metric": ["Real Rows", "Synthetic Rows", "Exact Matches", "Mean DCR", "Min DCR", "Privacy Score"],
            "Value": [len(real), len(synthetic), exact_matches, f"{mean_dcr:.4f}", f"{min_dcr:.4f}", f"{risk_score:.2f}%"]
        })
        csv = report_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Privacy Report (CSV)",
            data=csv,
            file_name="privacy_screening_report.csv",
            mime="text/csv",
        )

## ============================================================
# DOWNLOADS
# ============================================================

elif page == "📥 Downloads":
    st.header("📥 Downloads & Artifacts")
    st.write("Manage and download generated synthetic datasets and evaluation reports.")

    # Helper function for file sizes
    def format_size(bytes_size):
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        else:
            return f"{bytes_size / (1024 * 1024):.2f} MB"

    # Get all files
    generated_files = sorted(GENERATED_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    evaluation_files = sorted(EVALUATION_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    total_files = len(generated_files) + len(evaluation_files)
    
    if total_files == 0:
        st.info("No files available for download yet. Generate some data first!")
        st.stop()

    # Top Level Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Datasets", len(generated_files))
    c2.metric("Evaluation Reports", len(evaluation_files))
    total_size = sum(f.stat().st_size for f in generated_files + evaluation_files)
    c3.metric("Total Storage Used", format_size(total_size))

    st.divider()

    # Search Bar
    search_query = st.text_input("🔍 Search files by name", "").lower()

    # Tabs for separation of concerns
    tab_data, tab_reports = st.tabs(["📁 Synthetic Datasets", "📊 Evaluation Reports"])

    # ------------------------------------------------------------
    # TAB 1: GENERATED DATASETS
    # ------------------------------------------------------------
    with tab_data:
        if not generated_files:
            st.info("No synthetic datasets found.")
        else:
            filtered_gen = [f for f in generated_files if search_query in f.name.lower()]
            
            if not filtered_gen:
                st.warning("No datasets match your search.")
            
            for path in filtered_gen:
                data, error = read_csv_safely(path)
                if data is not None:
                    # Metadata
                    file_size = format_size(path.stat().st_size)
                    mod_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    rows = len(data)
                    cols = len(data.columns)
                    
                    # Card Layout
                    with st.container(border=True):
                        c_info, c_action = st.columns([3, 1])
                        
                        with c_info:
                            # Fixed encoding for em-dash and multiplication sign
                            st.markdown(f"### 📄 {path.name}")
                            st.caption(f"**Created:** {mod_time} | **Size:** {file_size} | **Shape:** {rows:,} rows × {cols} columns")
                        
                        with c_action:
                            # HTML styling to align the download button nicely
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                            st.download_button(
                                label="📥 Download CSV",
                                data=data.to_csv(index=False).encode("utf-8"),
                                file_name=path.name,
                                mime="text/csv",
                                key=f"dl_{path.name}",
                                use_container_width=True
                            )
                        
                        # Expandable Data Preview
                        with st.expander("👀 Preview Data"):
                            st.dataframe(data.head(10), use_container_width=True)
                else:
                    st.error(f"Could not read {path.name}: {error}")

    # ------------------------------------------------------------
    # TAB 2: EVALUATION REPORTS
    # ------------------------------------------------------------
    with tab_reports:
        if not evaluation_files:
            st.info("No evaluation reports found.")
        else:
            filtered_eval = [f for f in evaluation_files if search_query in f.name.lower()]
            
            if not filtered_eval:
                st.warning("No reports match your search.")

            for path in filtered_eval:
                data, error = read_csv_safely(path)
                if data is not None:
                    file_size = format_size(path.stat().st_size)
                    mod_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    rows = len(data)
                    
                    with st.container(border=True):
                        c_info, c_action = st.columns([3, 1])
                        
                        with c_info:
                            st.markdown(f"### 📊 {path.name}")
                            st.caption(f"**Created:** {mod_time} | **Size:** {file_size} | **Rows:** {rows:,}")
                        
                        with c_action:
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                            st.download_button(
                                label="📥 Download Report",
                                data=data.to_csv(index=False).encode("utf-8"),
                                file_name=path.name,
                                mime="text/csv",
                                key=f"dl_eval_{path.name}",
                                use_container_width=True
                            )
                        
                        with st.expander("👀 Preview Report"):
                            st.dataframe(data.head(10), use_container_width=True)
                else:
                    st.error(f"Could not read {path.name}: {error}")
                    
# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("SynData AI | MSc IT Research Project |  AI-Based Synthetic Dataset Generation and Evaluation")
