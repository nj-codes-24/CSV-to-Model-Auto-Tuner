import streamlit as st
import pandas as pd
import sys
import contextlib

class StreamlitCapture:
    def __init__(self, st_placeholder):
        self.st_placeholder = st_placeholder
        self.logs = []
    
    def write(self, text):
        if text.strip():
            self.logs.append(text.strip())
            # Keep only the last 15 lines so the UI doesn't lag
            display_text = "\n".join(self.logs[-15:])
            self.st_placeholder.code(display_text, language='text')
            
    def flush(self):
        pass


from pipeline import (
    build_preprocessor,
    run_baseline,
    run_tuning,
    auto_drop_columns
)
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Engine",
    page_icon="⚙️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark card-style containers */
    .block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 1rem 1.5rem;
    }
    div[data-testid="metric-container"] label { color: #a6adc8 !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #cdd6f4 !important;
        font-size: 2rem !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 1rem; }
    /* Section headers */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #89b4fa;
        margin-bottom: 0.25rem;
    }
    /* Winner badge */
    .winner-badge {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e2d40 100%);
        border: 1px solid #89b4fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #cdd6f4;
        font-size: 1rem;
    }
    .winner-badge strong { color: #89b4fa; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ AutoML Engine")
st.write("Upload a classification CSV → pick your target → click **Execute**. The engine handles everything else.")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        all_cols = df.columns.tolist()

        target_var = st.selectbox("Target variable (Y)", all_cols)
        
        st.divider()
        st.header("2. Parameters")
        test_size = st.slider("Test split ratio", 0.10, 0.40, 0.30, 0.05)
        cv_folds = st.slider("Cross-validation folds", 2, 10, 5, 1)

        st.divider()
        run_btn = st.button("⚡ Execute AutoML Engine", use_container_width=True, type="primary")

# ── Main area (no file yet) ───────────────────────────────────────────────────
if not uploaded_file:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

if not run_btn:
    st.info("Configure your target variable and parameters in the sidebar, then click **Execute AutoML Engine** to begin the pipeline.")
    st.stop()

# ==============================================================================
# ── EXECUTION PIPELINE ────────────────────────────────────────────────────────
# ==============================================================================

st.markdown("## 📂 Stage 0 — Original Dataset Analysis")

st.markdown('<p class="section-label">Original Dataset Preview (Top 5 Rows)</p>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)

st.markdown('<p class="section-label">Original Dataset Statistics</p>', unsafe_allow_html=True)
st.dataframe(df.describe(include="all"), use_container_width=True)
st.caption(f"Original Shape: {df.shape[0]:,} rows · {df.shape[1]} columns")

st.markdown('<p class="section-label">Missing Values</p>', unsafe_allow_html=True)
st.dataframe(df.isnull().sum().to_frame(name="Missing Count").T, use_container_width=True)

st.divider()

# ── Stage 1: Data Cleaning (Auto-Cleaner) ─────────────────────────────────────
st.markdown("## 🧹 Stage 1 — Data Cleaning")

drop_reasons = auto_drop_columns(df, target_var)
all_dropped = sum(drop_reasons.values(), [])

if all_dropped:
    with st.expander(f"Deleted {len(all_dropped)} noisy column(s)", expanded=True):
        st.markdown("These columns provide zero predictive value or confuse models, so they were safely removed:")
        if drop_reasons["empty"]:
            st.markdown(f"- **Empty (All NaNs):** `{', '.join(drop_reasons['empty'])}`")
        if drop_reasons["zero_variance"]:
            st.markdown(f"- **Zero Variance (Constant values):** `{', '.join(drop_reasons['zero_variance'])}`")
        if drop_reasons["high_cardinality"]:
            st.markdown(f"- **High Cardinality (Likely IDs/Names):** `{', '.join(drop_reasons['high_cardinality'])}`")
        if drop_reasons["multicollinear"]:
            st.markdown(f"- **Multicollinearity (Duplicates/Highly correlated):** `{', '.join(drop_reasons['multicollinear'])}`")
else:
    st.success("No noisy columns detected! All features look mathematically healthy.")

columns_kept = [c for c in df.columns if c not in all_dropped and c != target_var]
st.success(f"**Final columns to be used for training ({len(columns_kept)}):** `{', '.join(columns_kept)}`")

cols_to_remove = [target_var] + all_dropped
X = df.drop(columns=cols_to_remove)
y = df[target_var]

st.markdown('<p class="section-label">Cleaned Dataset Preview (Top 5 Rows)</p>', unsafe_allow_html=True)
st.dataframe(X.head(), use_container_width=True)

st.markdown('<p class="section-label">Cleaned Dataset Statistics</p>', unsafe_allow_html=True)
st.dataframe(X.describe(include="all"), use_container_width=True)

st.markdown('<p class="section-label">Missing Values</p>', unsafe_allow_html=True)
st.dataframe(X.isnull().sum().to_frame(name="Missing Count").T, use_container_width=True)

st.divider()

# ── Stage 2: Baseline Model Comparison ────────────────────────────────────────
st.markdown("## 📊 Stage 2 — Baseline Model Comparison")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
preprocessor, _, _ = build_preprocessor(X)

progress_bar = st.progress(0, text="Starting baseline evaluation…")
status_text = st.empty()

def on_progress(idx, total, name, status="end"):
    if status == "start":
        progress_bar.progress(idx / total, text=f"⚙️ Training {name} ({idx + 1}/{total})...")
    else:
        progress_bar.progress(idx / total, text=f"✅ Trained {name} ({idx}/{total})")
        status_text.caption(f"Last completed: **{name}**")

results, best_model_name, best_accuracy = run_baseline(
    X_train, X_test, y_train, y_test, preprocessor, progress_callback=on_progress
)

progress_bar.progress(1.0, text="Baseline evaluation complete.")
status_text.empty()

res_df = (
    pd.DataFrame(list(results.items()), columns=["Algorithm", "Accuracy"])
    .sort_values("Accuracy", ascending=False)
    .reset_index(drop=True)
)

col_table, col_chart = st.columns([1, 1])
with col_table:
    st.dataframe(
        res_df.style.highlight_max(axis=0, subset=["Accuracy"], color="#1e3a5f"),
        use_container_width=True,
    )
with col_chart:
    st.bar_chart(data=res_df, x="Algorithm", y="Accuracy", use_container_width=True)

st.divider()

# ── Stage 3: Hyperparameter Tuning ────────────────────────────────────────────
st.markdown(f"## ⚙️ Stage 3 — Hyperparameter Tuning ({best_model_name})")
st.write("Running exhaustive hyperparameter search. This may take a few minutes...")

log_placeholder = st.empty()

with st.spinner(f"Running RandomizedSearchCV on {best_model_name} with {cv_folds}-fold CV…"):
    with contextlib.redirect_stdout(StreamlitCapture(log_placeholder)):
        final_accuracy, best_params, report, _ = run_tuning(
            X_train, X_test, y_train, y_test, preprocessor, best_model_name, cv_folds
        )

m1, m2, m3 = st.columns(3)
m1.metric("Baseline Accuracy", f"{best_accuracy:.4f}")
m2.metric("Tuned Accuracy", f"{final_accuracy:.4f}", delta=f"{final_accuracy - best_accuracy:+.4f}")
m3.metric("CV Folds Used", cv_folds)

st.divider()

# ── Stage 4: The Solution ─────────────────────────────────────────────────────
st.markdown("## 🏆 Stage 4 — The Solution")

if final_accuracy <= best_accuracy:
    st.markdown(
        f'<div class="winner-badge">We recommend deploying <strong>{best_model_name}</strong> using its <strong>Default Parameters</strong>. Tuning did not improve accuracy.</div><br>',
        unsafe_allow_html=True,
    )
    st.markdown("### Optimal Hyperparameters")
    st.info("No need for custom hyperparameters. Use the default configuration.")
else:
    st.markdown(
        f'<div class="winner-badge">We recommend deploying <strong>{best_model_name}</strong> with the tuned parameters below.</div><br>',
        unsafe_allow_html=True,
    )
    st.markdown("### Optimal Hyperparameters")
    st.json(best_params)

st.markdown("### Classification Report")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format(precision=4), use_container_width=True)