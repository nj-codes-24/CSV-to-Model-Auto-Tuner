from __future__ import annotations

"""
eda_engine.py — Industry-Standard EDA Pipeline (Generalized)
============================================================
Extracts the 8-phase EDA methodology from the master class notebook into
reusable, dataset-agnostic functions.  No UI code — pure data science logic.

Philosophy preserved from the notebook:
  • Split BEFORE stats (no data leakage)
  • Impute using train-only statistics
  • ANOVA (regression) / Chi² (classification) for feature selection
  • IQR capping with Is_Massive flags on important features
  • Log-transform skewed features (and target for regression)
  • Hybrid encoding: TargetEncoder (high-card) + OHE (low-card)
  • Z-score scaling using train-only mean/std
  • RF importance scan → top-N feature selection
  • Smart metric recommendation based on target distribution
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# TASK TYPE DETECTION & METRIC RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_task_type(y: pd.Series) -> str:
    """
    Determine whether the target variable represents a classification or
    regression task.

    Rules:
      • object / category / bool dtype → classification
      • Numeric with ≤ 20 unique values  → classification
      • Numeric with > 20 unique values  → regression
    """
    if y.dtype in ["object", "category", "bool"] or pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 20:
            return "classification"
        return "regression"
    return "classification"


def recommend_metric(y: pd.Series, task_type: str) -> dict:
    """
    Analyse the target distribution and recommend the best evaluation metric.

    Returns a dict with:
      • task_type: "classification" or "regression"
      • primary_metric: sklearn scoring string
      • metric_display_name: human-readable name
      • explanation: why this metric was chosen
      • imbalance_ratio (classification only)
      • use_class_weight (classification only)
      • target_skew (regression only)
      • log_transform_target (regression only)
    """
    info: dict = {"task_type": task_type}

    if task_type == "classification":
        vc = y.value_counts()
        imbalance_ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
        info["imbalance_ratio"] = round(imbalance_ratio, 2)

        if imbalance_ratio < 3:
            info["primary_metric"] = "accuracy"
            info["metric_display_name"] = "Accuracy"
            info["explanation"] = (
                f"Imbalance ratio is {imbalance_ratio:.1f}:1 — the dataset is "
                f"**balanced**. Accuracy is an acceptable metric, but F1-Score "
                f"is also reported for completeness."
            )
            info["use_class_weight"] = False
        elif imbalance_ratio <= 10:
            info["primary_metric"] = "f1_weighted"
            info["metric_display_name"] = "F1-Score (weighted)"
            info["explanation"] = (
                f"Imbalance ratio is {imbalance_ratio:.1f}:1 — **moderate "
                f"imbalance** detected. DO NOT trust Accuracy alone. The engine "
                f"will use **F1-Score (weighted)** and apply `class_weight='balanced'` "
                f"to models that support it."
            )
            info["use_class_weight"] = True
        else:
            info["primary_metric"] = "f1_weighted"
            info["metric_display_name"] = "F1-Score (weighted)"
            info["explanation"] = (
                f"Imbalance ratio is {imbalance_ratio:.1f}:1 — **severe imbalance** "
                f"detected (>10:1). Accuracy is meaningless here. The engine will "
                f"use **F1-Score (weighted)** and apply `class_weight='balanced'`. "
                f"For production use, consider SMOTE or PR-AUC."
            )
            info["use_class_weight"] = True
    else:
        skew = float(y.skew())
        info["target_skew"] = round(skew, 3)
        info["log_transform_target"] = abs(skew) > 1
        info["primary_metric"] = "r2"
        info["metric_display_name"] = "R²"
        info["explanation"] = (
            f"Target skewness is {skew:.3f}. "
            + (
                "Target is **highly skewed** — applying `log1p` transform to stabilise variance. "
                if abs(skew) > 1
                else "Target distribution is approximately normal — no transform needed. "
            )
            + "The engine reports **R²** (overall fit), **MAE** (average error), "
            + "and **RMSE** (penalises large errors)."
        )

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BASIC LOADING & TRIMMING
# ═══════════════════════════════════════════════════════════════════════════════

def basic_trim(df: pd.DataFrame, target_col: str,
               leakage_cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Safe pre-split cleaning (no stats calculated, so no leakage risk):
      1. Drop exact-duplicate rows
      2. Drop columns with >50% missing values
      3. Drop columns where >99% of values are identical (near-constant)
      4. Drop user-specified leakage columns

    Returns (cleaned_df, report_dict).
    """
    report: dict = {
        "original_shape": df.shape,
        "duplicates_removed": 0,
        "high_missing_cols": [],
        "near_constant_cols": [],
        "id_cols_removed": [],
        "leakage_cols_removed": [],
    }

    # 1. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    # 2. Drop columns with >50% missing
    missing_pct = df.isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > 0.5].index.tolist()
    # Never drop the target
    high_missing = [c for c in high_missing if c != target_col]
    report["high_missing_cols"] = high_missing
    df = df.drop(columns=high_missing)

    # 3. Drop near-constant columns (>99% identical)
    near_const = []
    for c in df.columns:
        if c == target_col:
            continue
        vc = df[c].value_counts(normalize=True, dropna=False)
        if len(vc) > 0 and vc.iloc[0] > 0.99:
            near_const.append(c)
    report["near_constant_cols"] = near_const
    df = df.drop(columns=near_const)

    # 3.5 Drop ID-like columns (high cardinality identifiers)
    id_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        n_unique = df[c].nunique()
        total_rows = len(df)
        if total_rows == 0:
            continue
            
        is_obj = df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])
        is_int = pd.api.types.is_integer_dtype(df[c])
        
        # Rule 1: High-cardinality text (like Name, Ticket, Hash) -> > 70% unique
        if is_obj and (n_unique / total_rows) > 0.70:
            id_cols.append(c)
        # Rule 2: 100% unique integers (like PassengerId, RowNum)
        elif is_int and n_unique == total_rows:
            id_cols.append(c)
        # Rule 3: Has 'id' in name and high cardinality (> 50%)
        elif "id" in str(c).lower() and (n_unique / total_rows) > 0.50:
            if c not in id_cols:
                id_cols.append(c)
                
    report["id_cols_removed"] = id_cols
    df = df.drop(columns=id_cols)

    # 4. Drop user-specified leakage columns
    if leakage_cols:
        existing = [c for c in leakage_cols if c in df.columns and c != target_col]
        report["leakage_cols_removed"] = existing
        df = df.drop(columns=existing)

    report["final_shape"] = df.shape
    return df, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: IMPUTATION  (Phase 2 is train/test split — done in the orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

def smart_impute(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Skew-aware imputation using ONLY training statistics.
      • Categorical: fill with 'None' (missing often means "does not exist")
      • Numeric skewed (|skew| > 1): fill with median
      • Numeric normal: fill with mean
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    report: dict = {"categorical_imputed": [], "numeric_median": [], "numeric_mean": []}

    # Categorical
    for col in X_train.select_dtypes(["object", "category"]).columns:
        X_train[col] = X_train[col].fillna("None")
        X_test[col] = X_test[col].fillna("None")
        if col not in report["categorical_imputed"]:
            report["categorical_imputed"].append(col)

    # Numeric
    for col in X_train.select_dtypes(["int64", "float64"]).columns:
        if X_train[col].isnull().sum() == 0 and X_test[col].isnull().sum() == 0:
            continue
        if abs(X_train[col].skew()) > 1:
            val = X_train[col].median()
            report["numeric_median"].append(col)
        else:
            val = X_train[col].mean()
            report["numeric_mean"].append(col)
        X_train[col] = X_train[col].fillna(val)
        X_test[col] = X_test[col].fillna(val)

    return X_train, X_test, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: FEATURE SELECTION (ANOVA / Chi² + Correlation)
# ═══════════════════════════════════════════════════════════════════════════════

def feature_selection(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, task_type: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Statistically test each feature against the target and drop weak ones.

    Numeric features:
      • Drop one from each pair with correlation > 0.80 (keep the one more
        correlated with the target).

    Categorical features:
      • Regression: ANOVA F-test → drop if p > 0.05
      • Classification: Chi-Square test → drop if p > 0.05
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    report: dict = {"multicollinear_dropped": [], "weak_categorical_dropped": [], "method": ""}

    # ── A. Numeric: feature-to-feature correlation ────────────────────────────
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = X_train[numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        # For each highly-correlated pair, drop the one less correlated to target
        target_corr = X_train[numeric_cols].corrwith(
            y_train.astype(float) if task_type == "regression" else y_train.astype("category").cat.codes
        ).abs()

        to_drop = set()
        for i in range(len(upper.columns)):
            for j in range(i + 1, len(upper.columns)):
                if upper.iloc[i, j] > 0.80:
                    col_i, col_j = upper.columns[i], upper.columns[j]
                    # Drop the one with lower target correlation
                    if target_corr.get(col_i, 0) < target_corr.get(col_j, 0):
                        to_drop.add(col_i)
                    else:
                        to_drop.add(col_j)

        report["multicollinear_dropped"] = sorted(to_drop)
        X_train = X_train.drop(columns=list(to_drop), errors="ignore")
        X_test = X_test.drop(columns=list(to_drop), errors="ignore")

    # ── B. Categorical: ANOVA (regression) or Chi² (classification) ──────────
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    dead_weight: list[str] = []

    if task_type == "regression":
        report["method"] = "ANOVA F-test"
        train_temp = pd.concat([X_train, y_train.rename("__target__")], axis=1)
        for col in cat_cols:
            groups = [
                train_temp.loc[train_temp[col] == cat, "__target__"].dropna()
                for cat in train_temp[col].unique()
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) > 1:
                _, p_val = stats.f_oneway(*groups)
                if p_val > 0.05:
                    dead_weight.append(col)
    else:
        report["method"] = "Chi-Square test"
        train_temp = pd.concat([X_train, y_train.rename("__target__")], axis=1)
        for col in cat_cols:
            try:
                contingency = pd.crosstab(train_temp[col], train_temp["__target__"])
                _, p_val, _, _ = stats.chi2_contingency(contingency)
                if p_val > 0.05:
                    dead_weight.append(col)
            except Exception:
                pass  # Skip if the contingency table is degenerate

    report["weak_categorical_dropped"] = dead_weight
    X_train = X_train.drop(columns=dead_weight, errors="ignore")
    X_test = X_test.drop(columns=dead_weight, errors="ignore")

    return X_train, X_test, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: OUTLIERS & SKEWNESS
# ═══════════════════════════════════════════════════════════════════════════════

def handle_outliers_and_skew(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    task_type: str, log_transform_target: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    1. IQR capping (winsorisation) on ALL numeric features using train-only bounds.
       – For features in the top quartile of variance, create Is_Massive_ flags.
    2. Club rare categorical values (<1% frequency) into 'Other'.
    3. Log-transform skewed numeric features (|skew| > 1).
    4. Optionally log-transform the target (regression with skewed target).
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()
    report: dict = {
        "capped_features": 0,
        "massive_flags_created": [],
        "rare_cats_clubbed": [],
        "log_transformed_features": [],
        "target_log_transformed": False,
    }

    numeric_cols = X_train.select_dtypes(["int64", "float64"]).columns.tolist()

    # Identify VIP features (top quartile of variance — these get Is_Massive flags)
    if len(numeric_cols) > 0:
        variances = X_train[numeric_cols].var()
        vip_threshold = variances.quantile(0.75)
        vip_features = variances[variances >= vip_threshold].index.tolist()
    else:
        vip_features = []

    # 1. IQR Capping
    for col in numeric_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if col in vip_features:
            flag_name = f"Is_Massive_{col}"
            X_train[flag_name] = (X_train[col] > upper).astype(int)
            X_test[flag_name] = (X_test[col] > upper).astype(int)
            report["massive_flags_created"].append(flag_name)

        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)
        report["capped_features"] += 1

    # 2. Club rare categorical values
    for col in X_train.select_dtypes("object").columns:
        freq = X_train[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        if len(rare) > 0:
            X_train[col] = X_train[col].replace(rare, "Other")
            X_test[col] = X_test[col].replace(rare, "Other")
            report["rare_cats_clubbed"].append(col)

    # 3. Log-transform skewed features (skip Is_Massive flags)
    numeric_cols_post = [
        c for c in X_train.select_dtypes(["int64", "float64"]).columns
        if not c.startswith("Is_Massive_")
    ]
    for col in numeric_cols_post:
        if abs(X_train[col].skew()) > 1:
            # Ensure no negative values before log1p
            if X_train[col].min() >= 0 and X_test[col].min() >= 0:
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])
                report["log_transformed_features"].append(col)

    # 4. Log-transform target (regression only)
    if task_type == "regression" and log_transform_target:
        if y_train.min() >= 0:
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)
            report["target_log_transformed"] = True

    return X_train, X_test, y_train, y_test, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: HYBRID ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def hybrid_encode(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    • High-cardinality categoricals (>10 unique): TargetEncoder
    • Low-cardinality categoricals (≤10 unique): OneHotEncoder
    • Numeric columns: passed through untouched.
    """
    report: dict = {"target_encoded": [], "one_hot_encoded": [], "numeric_passthrough": []}

    cat_cols = X_train.select_dtypes(["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(["int64", "float64"]).columns.tolist()
    report["numeric_passthrough"] = numeric_cols

    if not cat_cols:
        return X_train, X_test, report

    high_card = [c for c in cat_cols if X_train[c].nunique() > 10]
    low_card = [c for c in cat_cols if X_train[c].nunique() <= 10]

    parts_train = [X_train[numeric_cols].reset_index(drop=True)]
    parts_test = [X_test[numeric_cols].reset_index(drop=True)]

    # Target encoding
    if high_card:
        te = TargetEncoder(target_type="continuous")
        te.fit(X_train[high_card], y_train)
        train_te = pd.DataFrame(
            te.transform(X_train[high_card]), columns=high_card
        )
        test_te = pd.DataFrame(
            te.transform(X_test[high_card]), columns=high_card
        )
        parts_train.append(train_te.reset_index(drop=True))
        parts_test.append(test_te.reset_index(drop=True))
        report["target_encoded"] = high_card

    # One-hot encoding
    if low_card:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(X_train[low_card])
        ohe_cols = ohe.get_feature_names_out(low_card).tolist()
        train_ohe = pd.DataFrame(
            ohe.transform(X_train[low_card]), columns=ohe_cols
        )
        test_ohe = pd.DataFrame(
            ohe.transform(X_test[low_card]), columns=ohe_cols
        )
        parts_train.append(train_ohe.reset_index(drop=True))
        parts_test.append(test_ohe.reset_index(drop=True))
        report["one_hot_encoded"] = low_card

    X_train_enc = pd.concat(parts_train, axis=1)
    X_test_enc = pd.concat(parts_test, axis=1)

    return X_train_enc, X_test_enc, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: SCALING
# ═══════════════════════════════════════════════════════════════════════════════

def safe_scale(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Z-score scaling using train-only mean/std.
    Skip binary flag columns (Is_Massive_*) to preserve their 0/1 meaning.
    Drop zero-variance columns to avoid division by zero.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    report: dict = {"scaled_cols": 0, "zero_variance_dropped": []}

    cols_to_scale = [c for c in X_train.columns if not c.startswith("Is_Massive_")]

    # Drop zero-variance first
    zv = [c for c in cols_to_scale if X_train[c].std() == 0]
    report["zero_variance_dropped"] = zv
    X_train = X_train.drop(columns=zv)
    X_test = X_test.drop(columns=zv)
    cols_to_scale = [c for c in cols_to_scale if c not in zv]

    for col in cols_to_scale:
        mean = X_train[col].mean()
        std = X_train[col].std()
        if std > 0:
            X_train[col] = (X_train[col] - mean) / std
            X_test[col] = (X_test[col] - mean) / std
            report["scaled_cols"] += 1

    return X_train, X_test, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: RF IMPORTANCE SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def rf_importance_scan(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, task_type: str, top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train a quick Random Forest, extract feature importances, and reduce to
    the top-N most important features.

    Returns (X_train_reduced, X_test_reduced, importance_df).
    """
    if task_type == "regression":
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(X_train, y_train)

    importance_df = (
        pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": rf.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Cap top_n to the number of available features
    top_n = min(top_n, len(importance_df))
    top_features = importance_df["Feature"].head(top_n).tolist()

    return X_train[top_features], X_test[top_features], importance_df


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_eda(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    leakage_cols: list[str] | None = None,
    top_n_features: int = 10,
    progress_callback=None,
) -> dict:
    """
    Run all 8 EDA phases end-to-end and return a comprehensive results dict.

    progress_callback(phase_num, phase_name, status) is called at the start
    and end of each phase for UI updates.

    Returns a dict with:
      • X_train, X_test, y_train, y_test  (cleaned & feature-selected)
      • task_type, metric_info
      • importance_df
      • All phase reports
      • target_log_transformed (bool)
    """

    def _progress(phase, name, status="start"):
        if progress_callback:
            progress_callback(phase, name, status)

    results: dict = {}

    # ── Phase 1: Basic Trim ──────────────────────────────────────────────────
    _progress(1, "Basic Cleaning")
    df_clean, trim_report = basic_trim(df, target_col, leakage_cols)
    results["trim_report"] = trim_report
    _progress(1, "Basic Cleaning", "end")

    # ── Task Detection & Metric Recommendation ──────────────────────────────
    _progress(2, "Task Detection")
    y_raw = df_clean[target_col]
    task_type = detect_task_type(y_raw)
    metric_info = recommend_metric(y_raw, task_type)
    results["task_type"] = task_type
    results["metric_info"] = metric_info
    _progress(2, "Task Detection", "end")

    # ── Phase 2: Train/Test Split ────────────────────────────────────────────
    _progress(3, "Train/Test Split")
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # For classification, encode target labels to numeric if needed
    label_map = None
    if task_type == "classification" and y.dtype == "object":
        label_map = {label: idx for idx, label in enumerate(sorted(y.unique()))}
        y = y.map(label_map)
        results["label_map"] = label_map

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    results["split_shapes"] = {
        "X_train": X_train.shape, "X_test": X_test.shape,
    }
    _progress(3, "Train/Test Split", "end")

    # ── Phase 3: Imputation ──────────────────────────────────────────────────
    _progress(4, "Imputation")
    X_train, X_test, impute_report = smart_impute(X_train, X_test)
    results["impute_report"] = impute_report
    _progress(4, "Imputation", "end")

    # ── Phase 4: Feature Selection ───────────────────────────────────────────
    _progress(5, "Feature Selection")
    X_train, X_test, fs_report = feature_selection(X_train, X_test, y_train, task_type)
    results["feature_selection_report"] = fs_report
    _progress(5, "Feature Selection", "end")

    # ── Phase 5: Outliers & Skewness ─────────────────────────────────────────
    _progress(6, "Outlier Handling")
    log_target = metric_info.get("log_transform_target", False)
    X_train, X_test, y_train, y_test, outlier_report = handle_outliers_and_skew(
        X_train, X_test, y_train, y_test, task_type, log_transform_target=log_target
    )
    results["outlier_report"] = outlier_report
    results["target_log_transformed"] = outlier_report.get("target_log_transformed", False)
    _progress(6, "Outlier Handling", "end")

    # ── Phase 6: Hybrid Encoding ─────────────────────────────────────────────
    _progress(7, "Encoding")
    X_train, X_test, enc_report = hybrid_encode(X_train, X_test, y_train)
    results["encoding_report"] = enc_report
    _progress(7, "Encoding", "end")

    # ── Phase 7: Scaling ─────────────────────────────────────────────────────
    _progress(8, "Scaling")
    X_train, X_test, scale_report = safe_scale(X_train, X_test)
    results["scale_report"] = scale_report
    _progress(8, "Scaling", "end")

    # ── Phase 8: RF Importance Scan ──────────────────────────────────────────
    _progress(9, "Feature Importance Scan")
    X_train, X_test, importance_df = rf_importance_scan(
        X_train, X_test, y_train, task_type, top_n=top_n_features
    )
    results["importance_df"] = importance_df
    results["top_features"] = X_train.columns.tolist()
    _progress(9, "Feature Importance Scan", "end")

    # ── Pack final outputs ───────────────────────────────────────────────────
    results["X_train"] = X_train
    results["X_test"] = X_test
    results["y_train"] = y_train
    results["y_test"] = y_test

    return results
