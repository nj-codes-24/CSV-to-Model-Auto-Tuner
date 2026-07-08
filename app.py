"""
app.py — AutoML Engine (Linear Flow)
======================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import contextlib
import joblib

from sklearn.model_selection import train_test_split
from eda_engine import (
    basic_trim, detect_task_type, recommend_metric,
    smart_impute, feature_selection, handle_outliers_and_skew,
    hybrid_encode, safe_scale, rf_importance_scan,
)
from pipeline import (
    run_baseline_classification, run_tuning_classification,
    run_baseline_regression, run_tuning_regression,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoML Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — Glassmorphic Dark Theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
/* Hide sidebar */
section[data-testid="stSidebar"],
button[data-testid="collapsedControl"] { display: none !important; }

.stApp { background: #0b0b1a !important; }
.block-container { padding-top: 2.5rem; max-width: 900px; margin: 0 auto; }

.hero-title {
    font-size: 3.2rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #89b4fa 0%, #b4befe 40%, #cba6f7 70%, #f5c2e7 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align: center;
    margin-bottom: 0.3rem; letter-spacing: -0.02em;
}
.hero-sub {
    text-align: center; color: #585b70;
    font-size: 1.15rem; font-weight: 400;
    letter-spacing: 0.1em; margin-bottom: 2.5rem;
}

.vb {
    background: linear-gradient(135deg, rgba(137,180,250,0.06) 0%, rgba(203,166,247,0.06) 100%);
    border: 1px solid rgba(137,180,250,0.2);
    border-radius: 20px; padding: 2.5rem 2rem;
    text-align: center; margin: 1rem 0 1.5rem 0;
}
.vb .vb-trophy { font-size: 3rem; margin-bottom: 0.5rem; }
.vb .vb-label {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.2em; color: #6c7086; margin-bottom: 0.4rem;
}
.vb .vb-model {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #89b4fa, #cba6f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem;
}
.vb .vb-detail { color: #a6adc8; font-size: 0.95rem; }

div[data-testid="metric-container"] {
    background: rgba(17,17,34,0.5);
    border: 1px solid rgba(137,180,250,0.1);
    border-radius: 12px; padding: 1rem;
}
div[data-testid="metric-container"] label { color: #585b70 !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #cdd6f4 !important; font-size: 1.6rem !important;
}

button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #89b4fa 0%, #b4befe 50%, #cba6f7 100%) !important;
    border: none !important; color: #0b0b1a !important;
    font-weight: 700 !important; font-size: 1.05rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(137,180,250,0.2) !important;
    transition: all 0.3s ease !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 6px 25px rgba(137,180,250,0.35) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(137,180,250,0.12), rgba(203,166,247,0.08)) !important;
    border: 1px solid rgba(137,180,250,0.2) !important;
    border-radius: 12px !important; color: #cdd6f4 !important;
    font-weight: 600 !important; padding: 0.8rem !important;
    transition: all 0.3s ease !important;
}
.stDownloadButton > button:hover {
    border-color: rgba(137,180,250,0.45) !important;
    box-shadow: 0 0 25px rgba(137,180,250,0.12) !important;
}
</style>""", unsafe_allow_html=True)


class StreamlitCapture:
    """Redirect print() output to a Streamlit code block."""
    def __init__(self, placeholder):
        self.ph, self.logs = placeholder, []
    def write(self, text):
        if text.strip():
            self.logs.append(text.strip())
            self.ph.code("\n".join(self.logs[-8:]), language="text")
    def flush(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "app_phase" not in st.session_state:
    st.session_state.app_phase = "setup"


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PHASE 1 — SETUP SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_phase == "setup":
    st.markdown("")
    st.markdown('<h1 class="hero-title">⚙️ AutoML Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Upload · Configure · Deploy</p>', unsafe_allow_html=True)

    _, center, _ = st.columns([1, 4, 1])
    with center:
        uploaded = st.file_uploader(
            "Drop your dataset here", type=["csv"], key="setup_csv",
            help="Any classification or regression CSV.",
        )

        if uploaded:
            fk = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("app_fkey") != fk:
                uploaded.seek(0)
                st.session_state.app_setup_df = pd.read_csv(uploaded)
                st.session_state.app_fkey = fk

            df = st.session_state.app_setup_df
            st.caption(f"✅  **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.dataframe(df.head(3), use_container_width=True, height=145)

            target_var = st.selectbox("🎯  Target variable", df.columns.tolist())

            with st.expander("⚙️  Advanced Parameters"):
                p1, p2 = st.columns(2)
                with p1:
                    test_size = st.slider("Test split ratio", 0.10, 0.40, 0.20, 0.05)
                    cv_folds  = st.slider("CV folds", 2, 10, 5, 1)
                with p2:
                    top_n = st.slider("Top N features", 5, 30, 10, 1)
                    leakage_cols = st.multiselect(
                        "Leakage columns",
                        [c for c in df.columns if c != target_var],
                    )

            st.markdown("")
            if st.button("⚡  Launch Engine", use_container_width=True, type="primary"):
                st.session_state.app_phase    = "results"
                st.session_state.app_df       = df.copy()
                st.session_state.app_target   = target_var
                st.session_state.app_tsize    = test_size
                st.session_state.app_cv       = cv_folds
                st.session_state.app_topn     = top_n
                st.session_state.app_leakage  = leakage_cols if leakage_cols else None
                if "app_done" in st.session_state:
                    del st.session_state["app_done"] # Force re-run if launched again
                st.rerun()

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PHASE 2 — EXECUTION & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

# Add the Back arrow button at the very top left
col_back, _ = st.columns([1, 8])
with col_back:
    if st.button("⬅️ Back", use_container_width=True):
        st.session_state.app_phase = "setup"
        st.rerun()

st.title("Automated Pipeline Results")
st.markdown("---")

df         = st.session_state.app_df
target_var = st.session_state.app_target
test_size  = st.session_state.app_tsize
cv_folds   = st.session_state.app_cv
top_n      = st.session_state.app_topn
leakage_cols = st.session_state.app_leakage


# Set up UI slots for progressive rendering
st.header("Stage 1: EDA")
eda_log_slot = st.empty()

st.header("Stage 2: Model Selection")
mdl_prog_slot = st.empty()
mdl_log_slot = st.empty()
mdl_res_slot = st.empty()

st.header("Stage 3: Hyperparameter Tuning")
tune_prog_slot = st.empty()
tune_log_slot = st.empty()
tune_res_slot = st.empty()

st.header("Final Conclusion")
final_badge_slot = st.empty()
final_metrics_slot = st.empty()
st.markdown("<br>", unsafe_allow_html=True)
final_downloads_slot = st.empty()


# ── RUN PIPELINE (First Pass) ──
if "app_done" not in st.session_state:
    
    eda_log = ""
    def log(msg):
        nonlocal eda_log
        eda_log += f"- {msg}\n"
        eda_log_slot.markdown(eda_log)
    def sublog(msg):
        nonlocal eda_log
        eda_log += f"  - {msg}\n"
        eda_log_slot.markdown(eda_log)

    # ── Stage 1: EDA ──
    log(f"**Dataset loaded** — {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # 1. Cleaning
    df_clean, trim = basic_trim(df, target_var, leakage_cols)
    log("**Cleaning**")
    if trim["duplicates_removed"]: sublog(f"Removed {trim['duplicates_removed']} duplicate rows")
    if trim["high_missing_cols"]: sublog(f"Dropped {len(trim['high_missing_cols'])} cols (>50% missing): `{', '.join(trim['high_missing_cols'])}`")
    if trim["near_constant_cols"]: sublog(f"Dropped {len(trim['near_constant_cols'])} near-constant cols: `{', '.join(trim['near_constant_cols'])}`")
    if trim["leakage_cols_removed"]: sublog(f"Removed {len(trim['leakage_cols_removed'])} leakage cols: `{', '.join(trim['leakage_cols_removed'])}`")
    
    # 2. Task detection
    y_raw      = df_clean[target_var]
    task_type  = detect_task_type(y_raw)
    metric_info = recommend_metric(y_raw, task_type)
    log(f"**Task Detection**: {task_type.title()} | Primary metric: {metric_info['metric_display_name']}")
    
    # 3. Split
    X = df_clean.drop(columns=[target_var])
    y = df_clean[target_var]
    label_map = None
    if task_type == "classification" and y.dtype == "object":
        label_map = {lb: i for i, lb in enumerate(sorted(y.unique()))}
        y = y.map(label_map)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    log(f"**Split** — Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    
    # 4. Imputation
    X_train, X_test, imp = smart_impute(X_train, X_test)
    log("**Imputation** (train-only stats)")
    if imp["numeric_median"]: sublog(f"Median-filled: {len(imp['numeric_median'])} cols")
    if imp["numeric_mean"]: sublog(f"Mean-filled: {len(imp['numeric_mean'])} cols")
    if imp["categorical_imputed"]: sublog(f"Categorical filled: {len(imp['categorical_imputed'])} cols")
    
    # 5. Feature selection
    X_train, X_test, fs = feature_selection(X_train, X_test, y_train, task_type)
    log(f"**Feature Selection** ({fs['method'] or 'Correlation'})")
    if fs["multicollinear_dropped"]: sublog(f"Dropped {len(fs['multicollinear_dropped'])} multicollinear cols")
    if fs["weak_categorical_dropped"]: sublog(f"Dropped {len(fs['weak_categorical_dropped'])} weak categoricals")
    if not fs["multicollinear_dropped"] and not fs["weak_categorical_dropped"]: sublog("✅ All features passed significance tests")
    
    # 6. Outliers & Skewness
    log_tgt = metric_info.get("log_transform_target", False)
    X_train, X_test, y_train, y_test, out = handle_outliers_and_skew(X_train, X_test, y_train, y_test, task_type, log_transform_target=log_tgt)
    log("**Outlier & Skewness Treatment**")
    sublog(f"IQR-capped {out['capped_features']} features")
    if out["massive_flags_created"]: sublog(f"Created {len(out['massive_flags_created'])} outlier flags")
    if out["log_transformed_features"]: sublog(f"Log-transformed {len(out['log_transformed_features'])} skewed features")
    if out["target_log_transformed"]: sublog("Target was log-transformed (skewed)")
    
    # 7. Encoding
    X_train, X_test, enc = hybrid_encode(X_train, X_test, y_train)
    log("**Encoding**")
    if enc["target_encoded"]: sublog(f"Target Encoded: {len(enc['target_encoded'])} cols")
    if enc["one_hot_encoded"]: sublog(f"One-Hot Encoded: {len(enc['one_hot_encoded'])} cols")
    
    # 8. Scaling
    X_train, X_test, sc = safe_scale(X_train, X_test)
    log("**Scaling** (Z-score)")
    sublog(f"Scaled {sc['scaled_cols']} columns")
    
    # 9. RF Importance
    X_train, X_test, importance_df = rf_importance_scan(X_train, X_test, y_train, task_type, top_n=top_n)
    top_features = X_train.columns.tolist()
    log(f"**RF Importance Scan** — Selected top {len(top_features)} features: `{', '.join(top_features)}`")

    
    # ── Stage 2: Model Selection ──
    mdl_prog = mdl_prog_slot.progress(0, text="Training models...")
    def _mdl_prog(idx, total, name, status="end"):
        if status == "start":
            mdl_prog.progress(idx / total, text=f"⚙️ Training {name} ({idx+1}/{total})...")
        else:
            mdl_prog.progress(idx / total, text=f"✅ {name} trained ({idx}/{total})")

    if task_type == "classification":
        scoring = metric_info["primary_metric"]
        use_cw  = metric_info.get("use_class_weight", False)
        with contextlib.redirect_stdout(StreamlitCapture(mdl_log_slot)):
            mdl_results, best_model, best_score = run_baseline_classification(
                X_train, X_test, y_train, y_test, scoring_metric=scoring, use_class_weight=use_cw, progress_callback=_mdl_prog
            )
        sort_col = "Accuracy" if scoring == "accuracy" else "F1 (weighted)"
    else:
        scoring = "r2"
        with contextlib.redirect_stdout(StreamlitCapture(mdl_log_slot)):
            mdl_results, best_model, best_score = run_baseline_regression(
                X_train, X_test, y_train, y_test, progress_callback=_mdl_prog
            )
        sort_col = "R²"

    # Models finished. Clear progress and logs to make room for final UI
    mdl_prog_slot.empty()
    mdl_log_slot.empty()

    
    # ── Stage 3: Tuning ──
    tune_prog = tune_prog_slot.progress(0, text=f"Hyperparameter tuning {best_model}...")
    
    if task_type == "classification":
        with contextlib.redirect_stdout(StreamlitCapture(tune_log_slot)):
            final_acc, final_f1, best_params, cls_report, final_model = run_tuning_classification(
                X_train, X_test, y_train, y_test, best_model, cv_folds, scoring_metric=scoring, use_class_weight=use_cw
            )
        tuned_score = final_acc if scoring == "accuracy" else final_f1
        final_metrics = {"Accuracy": final_acc, "F1 (weighted)": final_f1}
    else:
        with contextlib.redirect_stdout(StreamlitCapture(tune_log_slot)):
            final_r2, final_mae, final_rmse, best_params, final_model = run_tuning_regression(
                X_train, X_test, y_train, y_test, best_model, cv_folds
            )
        tuned_score = final_r2
        final_metrics = {"R²": final_r2, "MAE": final_mae, "RMSE": final_rmse}
        
    improved = tuned_score > best_score
    
    tune_prog_slot.empty()
    tune_log_slot.empty()

    # Save everything to session state
    st.session_state.update({
        "app_eda_log":       eda_log,
        "app_task_type":     task_type,
        "app_metric_info":   metric_info,
        "app_mdl_results":   mdl_results,
        "app_best_model":    best_model,
        "app_best_score":    best_score,
        "app_tuned_score":   tuned_score,
        "app_improved":      improved,
        "app_best_params":   best_params,
        "app_final_model":   final_model,
        "app_final_metrics": final_metrics,
        "app_sort_col":      sort_col,
        "app_top_features":  top_features,
        "app_X_train":       X_train,
        "app_X_test":        X_test,
        "app_target_log":    out.get("target_log_transformed", False),
        "app_done":          True,
    })


# ── RENDER STORED UI (Happens instantly when session_state exists) ──

# Stage 1 UI
eda_log_slot.markdown(st.session_state.app_eda_log)

# Stage 2 UI
with mdl_res_slot.container():
    res_df = (
        pd.DataFrame(st.session_state.app_mdl_results).T
        .reset_index().rename(columns={"index": "Algorithm"})
        .sort_values(st.session_state.app_sort_col, ascending=False).reset_index(drop=True)
    )
    col_tbl, col_cht = st.columns([1, 1])
    with col_tbl:
        st.dataframe(
            res_df.style.highlight_max(axis=0, subset=[st.session_state.app_sort_col], color="#1e3a5f"),
            use_container_width=True,
        )
    with col_cht:
        st.bar_chart(data=res_df, x="Algorithm", y=st.session_state.app_sort_col)


# Stage 3 UI
with tune_res_slot.container():
    improved = st.session_state.app_improved
    best_params = st.session_state.app_best_params
    best_model = st.session_state.app_best_model
    metric_name = st.session_state.app_metric_info["metric_display_name"]

    if improved and best_params:
        st.success(f"Tuning improved the model! **{best_model}** {metric_name} increased from **{st.session_state.app_best_score:.4f}** to **{st.session_state.app_tuned_score:.4f}**.")
        st.markdown("### Optimal Hyperparameters:")
        st.json(best_params)
    else:
        st.info(f"Tuning did not yield a better score. Default hyperparameters for **{best_model}** are optimal.")
        st.markdown(f"**Best {metric_name}:** {st.session_state.app_best_score:.4f}")


# Final Conclusion UI
delta = st.session_state.app_tuned_score - st.session_state.app_best_score
if improved and best_params:
    detail = f"Tuning improved {metric_name} by <strong>+{delta:.4f}</strong>. Use the optimised parameters above."
else:
    detail = "Default parameters are optimal. No tuning improvement detected."

final_badge_slot.markdown(f"""
<div class="vb">
    <div class="vb-trophy">🏆</div>
    <div class="vb-label">Winning Model</div>
    <div class="vb-model">{best_model}</div>
    <div class="vb-detail">{detail}</div>
</div>
""", unsafe_allow_html=True)

with final_metrics_slot.container():
    final_metrics = st.session_state.app_final_metrics
    cols = st.columns(len(final_metrics))
    for col, (k, v) in zip(cols, final_metrics.items()):
        col.metric(k, f"{v:.4f}")
        
    if st.session_state.app_target_log:
        st.warning("Note: Metrics are in **log-space** (target was log-transformed). Apply `np.expm1()` to predictions for real-world values.")


with final_downloads_slot.container():
    dl1, dl2 = st.columns(2)
    with dl1:
        cleaned = pd.concat([st.session_state.app_X_train, st.session_state.app_X_test], axis=0)
        csv_buf = io.BytesIO()
        cleaned.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️  Download Cleaned Dataset (CSV)",
            data=csv_buf.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{cleaned.shape[0]:,} rows × {cleaned.shape[1]} cols")

    with dl2:
        mdl_buf = io.BytesIO()
        joblib.dump(st.session_state.app_final_model, mdl_buf)
        st.download_button(
            "⬇️  Download Trained Model (.pkl)",
            data=mdl_buf.getvalue(),
            file_name=f"tuned_{best_model.replace(' ', '_').lower()}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )
        st.caption(f"Model: {best_model}")