import streamlit as st
import pandas as pd

from pipeline import build_preprocessor, run_baseline, run_tuning
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
        potential_drops = [c for c in all_cols if c != target_var]
        columns_to_drop = st.multiselect(
            "Drop columns (IDs, names, high-cardinality)",
            options=potential_drops,
        )

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

# ── Data preview ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Dataset Preview</p>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)
st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")

if not run_btn:
    st.stop()

# ── Execution ─────────────────────────────────────────────────────────────────
st.divider()

X = df.drop(columns=[target_var] + columns_to_drop)
y = df[target_var]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

# Feature schema
with st.expander("🔍 Feature Schema Detected", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<p class="section-label">Numerical ({len(numeric_cols)})</p>', unsafe_allow_html=True)
        st.write(numeric_cols if numeric_cols else "—")
    with c2:
        st.markdown(f'<p class="section-label">Categorical ({len(categorical_cols)})</p>', unsafe_allow_html=True)
        st.write(categorical_cols if categorical_cols else "—")

# ── Stage 1: Baseline ─────────────────────────────────────────────────────────
st.markdown("## 📊 Stage 1 — Baseline Model Comparison")

progress_bar = st.progress(0, text="Starting baseline evaluation…")
status_text = st.empty()

def on_progress(idx, total, name):
    progress_bar.progress(idx / total, text=f"Trained {name} ({idx}/{total})")
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

st.markdown(
    f'<div class="winner-badge">🏆 Baseline winner: <strong>{best_model_name}</strong> — {best_accuracy:.4f} accuracy</div>',
    unsafe_allow_html=True,
)

# ── Stage 2: Hyperparameter tuning ───────────────────────────────────────────
st.markdown(f"## ⚙️ Stage 2 — Tuning {best_model_name}")

with st.spinner(f"Running GridSearchCV on {best_model_name} with {cv_folds}-fold CV…"):
    final_accuracy, best_params, report, _ = run_tuning(
        X_train, X_test, y_train, y_test, preprocessor, best_model_name, cv_folds
    )

m1, m2, m3 = st.columns(3)
m1.metric("Baseline Accuracy", f"{best_accuracy:.4f}")
m2.metric("Tuned Accuracy", f"{final_accuracy:.4f}", delta=f"{final_accuracy - best_accuracy:+.4f}")
m3.metric("CV Folds Used", cv_folds)

st.markdown("**Optimal hyperparameters found:**")
st.json(best_params)

# ── Classification report ─────────────────────────────────────────────────────
st.markdown("### 📋 Classification Report")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format(precision=4), use_container_width=True)