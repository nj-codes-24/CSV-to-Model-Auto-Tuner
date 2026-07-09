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

    # Guard: zero rows after dedup (e.g. single-row CSV)
    if len(df) == 0:
        raise ValueError(
            "Dataset has 0 rows after removing duplicates. "
            "Please upload a dataset with at least 2 unique rows."
        )

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

    # Guard: all feature columns removed, only target remains
    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) == 0:
        raise ValueError(
            "All feature columns were removed during cleaning "
            "(high-missing, near-constant, ID-like, or leakage). "
            "The dataset has no usable features. Please review your data."
        )

    report["final_shape"] = df.shape
    return df, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2.5: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def smart_feature_engineering(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Automatically extracts powerful features from object columns before imputation.
    1. Datetime extraction (Year, Month, Day, DayOfWeek, Is_Weekend)
    2. Smart delimiter splitting (for structured strings like 'A/15/P')
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    state: dict = {
        "datetime_cols": [],
        "split_cols": {}, # {col: {'delimiter': delim, 'new_cols': [list]}}
        "dropped_cols": []
    }
    
    obj_cols = X_train.select_dtypes(["object", "string", "category"]).columns
    
    for col in obj_cols:
        # Check for Datetime first (ignore purely numeric columns that got stringified)
        if pd.to_numeric(X_train[col], errors='coerce').notna().mean() > 0.5:
            continue
            
        parsed = pd.to_datetime(X_train[col], errors='coerce')
        valid_ratio = parsed.notna().mean()
        
        if valid_ratio > 0.8:
            # Extract features on train
            X_train[f"{col}_Year"] = parsed.dt.year
            X_train[f"{col}_Month"] = parsed.dt.month
            X_train[f"{col}_Day"] = parsed.dt.day
            X_train[f"{col}_DayOfWeek"] = parsed.dt.dayofweek
            X_train[f"{col}_Is_Weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
            
            # Extract features on test
            parsed_test = pd.to_datetime(X_test[col], errors='coerce')
            X_test[f"{col}_Year"] = parsed_test.dt.year
            X_test[f"{col}_Month"] = parsed_test.dt.month
            X_test[f"{col}_Day"] = parsed_test.dt.day
            X_test[f"{col}_DayOfWeek"] = parsed_test.dt.dayofweek
            X_test[f"{col}_Is_Weekend"] = (parsed_test.dt.dayofweek >= 5).astype(int)
            
            state["datetime_cols"].append(col)
            state["dropped_cols"].append(col)
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
            continue
            
        # Try Smart Delimiter Splitting
        delimiters = ['/', '-', '_', '|']
        best_delimiter = None
        best_k = None
        
        for delim in delimiters:
            counts = X_train[col].dropna().astype(str).str.count(delim)
            if len(counts) > 0:
                mode_vals = counts.mode()
                if len(mode_vals) > 0:
                    k = mode_vals[0]
                    if k > 0 and (counts == k).mean() > 0.9:
                        best_delimiter = delim
                        best_k = k
                        break
        
        if best_delimiter is not None:
            new_cols = [f"{col}_part_{i+1}" for i in range(int(best_k) + 1)]
            
            # Split train
            split_train = X_train[col].astype(str).str.split(best_delimiter, expand=True)
            if split_train.shape[1] == len(new_cols):
                split_train.columns = new_cols
                X_train = pd.concat([X_train, split_train], axis=1)
                
                # Split test
                split_test = X_test[col].astype(str).str.split(best_delimiter, expand=True)
                for i, new_col in enumerate(new_cols):
                    X_test[new_col] = split_test[i] if i < split_test.shape[1] else np.nan
                
                X_test = X_test.drop(columns=[col])
                X_train = X_train.drop(columns=[col])
                
                state["split_cols"][col] = {
                    "delimiter": best_delimiter,
                    "new_cols": new_cols
                }
                state["dropped_cols"].append(col)
                
    return X_train, X_test, state
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
    report: dict = {
        "categorical_imputed": [], 
        "numeric_median": [], 
        "numeric_mean": [],
        "median_values": {},
        "mean_values": {}
    }

    # Categorical (including bool which might have become mixed with NaNs)
    for col in X_train.select_dtypes(["object", "category", "bool"]).columns:
        X_train[col] = X_train[col].fillna("None").astype(str)
        X_test[col] = X_test[col].fillna("None").astype(str)
        if col not in report["categorical_imputed"]:
            report["categorical_imputed"].append(col)

    # Numeric
    for col in X_train.select_dtypes(["int64", "float64"]).columns:
        # Guard: entirely NaN column — skew() and mean() would return NaN
        if X_train[col].isnull().all():
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
            report["numeric_median"].append(col)
            report["median_values"][col] = 0
            continue
        skew_val = X_train[col].skew()
        if pd.isna(skew_val) or abs(skew_val) > 1:
            val = X_train[col].median()
            report["numeric_median"].append(col)
            report["median_values"][col] = val
        else:
            val = X_train[col].mean()
            report["numeric_mean"].append(col)
            report["mean_values"][col] = val
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
        # Guard: drop NaN from y before computing correlation (NaN → cat.codes = -1)
        y_clean = y_train.dropna()
        X_aligned = X_train[numeric_cols].loc[y_clean.index]
        try:
            target_corr = X_aligned.corrwith(
                y_clean.astype(float) if task_type == "regression" else y_clean.astype("category").cat.codes
            ).abs().fillna(0)
        except Exception:
            target_corr = pd.Series(0, index=numeric_cols)

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
            try:
                groups = [
                    train_temp.loc[train_temp[col] == cat, "__target__"].dropna()
                    for cat in train_temp[col].unique()
                ]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) > 1:
                    _, p_val = stats.f_oneway(*groups)
                    # Guard: p_val can be NaN if groups have zero variance
                    if pd.notna(p_val) and p_val > 0.05:
                        dead_weight.append(col)
            except Exception:
                pass  # Skip if ANOVA fails (e.g. constant groups)
    else:
        report["method"] = "Chi-Square test"
        train_temp = pd.concat([X_train, y_train.rename("__target__")], axis=1)
        for col in cat_cols:
            try:
                contingency = pd.crosstab(train_temp[col], train_temp["__target__"])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue  # Skip degenerate contingency tables
                _, p_val, _, _ = stats.chi2_contingency(contingency)
                if pd.notna(p_val) and p_val > 0.05:
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
        "upper_bounds": {},
        "lower_bounds": {},
        "rare_categories": {}
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

        # Guard: IQR == 0 means the feature has no spread (binary/constant).
        # Clipping to [Q1, Q1] would collapse all values — skip instead.
        if IQR == 0:
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if col in vip_features:
            flag_name = f"Is_Massive_{col}"
            X_train[flag_name] = (X_train[col] > upper).astype(int)
            X_test[flag_name] = (X_test[col] > upper).astype(int)
            report["massive_flags_created"].append(flag_name)

        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)
        report["upper_bounds"][col] = upper
        report["lower_bounds"][col] = lower
        report["capped_features"] += 1

    # 2. Club rare categorical values
    for col in X_train.select_dtypes(["object", "category"]).columns:
        freq = X_train[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index.tolist()
        if len(rare) > 0:
            X_train[col] = X_train[col].replace(rare, "Other")
            X_test[col] = X_test[col].replace(rare, "Other")
            report["rare_cats_clubbed"].append(col)
            report["rare_categories"][col] = rare

    # 3. Log-transform skewed features (skip Is_Massive flags)
    numeric_cols_post = [
        c for c in X_train.select_dtypes(["int64", "float64"]).columns
        if not c.startswith("Is_Massive_")
    ]
    for col in numeric_cols_post:
        # Guard: skip zero-variance columns (all identical values)
        if X_train[col].nunique() <= 1:
            continue
        skew_val = X_train[col].skew()
        if pd.notna(skew_val) and abs(skew_val) > 1:
            # Ensure no negative values before log1p
            if X_train[col].min() >= 0 and X_test[col].min() >= 0:
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])
                report["log_transformed_features"].append(col)

    # 4. Log-transform target (regression only)
    if task_type == "regression" and log_transform_target:
        # Guard: check BOTH train and test for negative values
        if y_train.min() >= 0 and y_test.min() >= 0:
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
    report: dict = {
        "target_encoded": [], 
        "one_hot_encoded": [], 
        "numeric_passthrough": [],
        "target_encoder": None,
        "one_hot_encoder": None
    }

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
        # Use appropriate target_type: "binary" for 2-class, "continuous" otherwise
        n_classes = y_train.nunique()
        te_target_type = "binary" if n_classes == 2 else "continuous"
        te = TargetEncoder(target_type=te_target_type)
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
        report["target_encoder"] = te

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
        report["one_hot_encoder"] = ohe

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
    report: dict = {
        "scaled_cols": 0, 
        "zero_variance_dropped": [],
        "scale_means": {},
        "scale_stds": {}
    }

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
            report["scale_means"][col] = mean
            report["scale_stds"][col] = std

    return X_train, X_test, report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: RF IMPORTANCE SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def rf_importance_scan(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, task_type: str, top_n: int = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train a quick Random Forest, extract feature importances, and mathematically 
    detect the 'elbow' point to keep only the most predictive features.
    
    Returns (X_train_reduced, X_test_reduced, importance_df).
    """
    if X_train.shape[1] == 0:
        empty_imp = pd.DataFrame({"Feature": [], "Importance": []})
        return X_train, X_test, empty_imp

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

    n_features = len(importance_df)
    
    if n_features > 2:
        y = importance_df["Importance"].values
        y_min, y_max = y.min(), y.max()
        
        if y_max > y_min:
            # Normalize axes for accurate geometric distance calculation
            y_norm = (y - y_min) / (y_max - y_min)
            x_norm = np.linspace(0, 1, n_features)
            
            # Distance maximized when x+y is minimized (curvature point)
            elbow_index = np.argmin(x_norm + y_norm)
            elbow_n = max(1, elbow_index + 1)
            
            # Cumulative threshold fallback (min 95% information retained)
            cum_y = np.cumsum(y) / np.sum(y)
            cum_n = np.searchsorted(cum_y, 0.95) + 1
            
            # Take the safest upper bound to preserve predictive power
            auto_top_n = max(elbow_n, cum_n)
        else:
            auto_top_n = n_features
    else:
        auto_top_n = n_features

    # Fallback if user explicitly passed a top_n anyway
    if top_n is not None:
        auto_top_n = min(top_n, n_features)
        
    top_features = importance_df["Feature"].head(auto_top_n).tolist()

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

    # ── Phase 2.5: Feature Engineering ─────────────────────────────────────────
    _progress(3.5, "Feature Engineering")
    X_train, X_test, eng_state = smart_feature_engineering(X_train, X_test)
    results["engineering_state"] = eng_state
    _progress(3.5, "Feature Engineering", "end")

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
    eda_state = {
        "drop_cols": trim_report["high_missing_cols"] + trim_report["near_constant_cols"] + trim_report["id_cols_removed"] + trim_report["leakage_cols_removed"],
        "impute_medians": impute_report.get("median_values", {}),
        "impute_means": impute_report.get("mean_values", {}),
        "fs_dropped": fs_report["multicollinear_dropped"] + fs_report["weak_categorical_dropped"],
        "outlier_upper": outlier_report.get("upper_bounds", {}),
        "outlier_lower": outlier_report.get("lower_bounds", {}),
        "massive_flags": outlier_report["massive_flags_created"],
        "rare_categories": outlier_report.get("rare_categories", {}),
        "log_transformed": outlier_report["log_transformed_features"],
        "target_encoder": enc_report.get("target_encoder"),
        "one_hot_encoder": enc_report.get("one_hot_encoder"),
        "te_cols": enc_report.get("target_encoded", []),
        "ohe_cols": enc_report.get("one_hot_encoded", []),
        "scale_means": scale_report.get("scale_means", {}),
        "scale_stds": scale_report.get("scale_stds", {}),
        "scale_dropped": scale_report["zero_variance_dropped"],
        "top_features": X_train.columns.tolist(),
        "engineering_state": eng_state
    }
    
    results["eda_state"] = eda_state
    results["X_train"] = X_train
    results["X_test"] = X_test
    results["y_train"] = y_train
    results["y_test"] = y_test

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def apply_eda_pipeline(test_df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """
    Apply the exactly saved EDA mathematical transformations to an unseen dataset.
    Returns the fully transformed DataFrame ready for .predict().
    """
    df = test_df.copy()
    
    # 1. Basic Trim
    df = df.drop(columns=[c for c in state["drop_cols"] if c in df.columns], errors="ignore")
    
    # 1.5 Feature Engineering
    if "engineering_state" in state:
        eng_state = state["engineering_state"]
        
        # Datetime extraction
        for col in eng_state.get("datetime_cols", []):
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors='coerce')
                df[f"{col}_Year"] = parsed.dt.year
                df[f"{col}_Month"] = parsed.dt.month
                df[f"{col}_Day"] = parsed.dt.day
                df[f"{col}_DayOfWeek"] = parsed.dt.dayofweek
                df[f"{col}_Is_Weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
                
        # Smart delimiter splitting
        for col, split_info in eng_state.get("split_cols", {}).items():
            if col in df.columns:
                delim = split_info["delimiter"]
                new_cols = split_info["new_cols"]
                split_df = df[col].astype(str).str.split(delim, expand=True)
                for i, new_col in enumerate(new_cols):
                    df[new_col] = split_df[i] if i < split_df.shape[1] else np.nan
                    
        # Drop original feature-engineered columns
        df = df.drop(columns=[c for c in eng_state.get("dropped_cols", []) if c in df.columns], errors="ignore")
    
    # 2. Imputation
    for col in df.select_dtypes(["object", "category", "bool"]).columns:
        df[col] = df[col].fillna("None").astype(str)
    for col, val in state["impute_medians"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in state["impute_means"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
            
    # 3. Feature Selection Drops
    df = df.drop(columns=[c for c in state["fs_dropped"] if c in df.columns], errors="ignore")
    
    # 4. Outliers & Skewness
    for col, upper in state["outlier_upper"].items():
        if col in df.columns:
            flag_name = f"Is_Massive_{col}"
            if flag_name in state["massive_flags"]:
                df[flag_name] = (df[col] > upper).astype(int)
            lower = state["outlier_lower"].get(col, -float('inf'))
            df[col] = df[col].clip(lower=lower, upper=upper)
            
    for col, rare_list in state["rare_categories"].items():
        if col in df.columns:
            df[col] = df[col].replace(rare_list, "Other")
            
    for col in state["log_transformed"]:
        if col in df.columns:
            # Replicate np.log1p logic from training
            if df[col].min() >= 0:
                df[col] = np.log1p(df[col])
                
    # 5. Hybrid Encoding
    parts = []
    numeric_cols = df.select_dtypes(["int64", "float64"]).columns.tolist()
    if numeric_cols:
        parts.append(df[numeric_cols].reset_index(drop=True))
        
    if state["target_encoder"] is not None and state["te_cols"]:
        te = state["target_encoder"]
        # Filter te_cols that exist
        valid_te_cols = [c for c in state["te_cols"] if c in df.columns]
        if valid_te_cols:
            transformed_te = pd.DataFrame(
                te.transform(df[valid_te_cols]), columns=valid_te_cols
            )
            parts.append(transformed_te.reset_index(drop=True))
            
    if state["one_hot_encoder"] is not None and state["ohe_cols"]:
        ohe = state["one_hot_encoder"]
        valid_ohe_cols = [c for c in state["ohe_cols"] if c in df.columns]
        if valid_ohe_cols:
            ohe_feature_names = ohe.get_feature_names_out(valid_ohe_cols).tolist()
            transformed_ohe = pd.DataFrame(
                ohe.transform(df[valid_ohe_cols]), columns=ohe_feature_names
            )
            parts.append(transformed_ohe.reset_index(drop=True))
            
    if parts:
        df = pd.concat(parts, axis=1)
        
    # 6. Scaling
    df = df.drop(columns=[c for c in state["scale_dropped"] if c in df.columns], errors="ignore")
    for col, mean in state["scale_means"].items():
        if col in df.columns:
            std = state["scale_stds"].get(col, 1.0)
            if std > 0:
                df[col] = (df[col] - mean) / std
                
    # 7. Final Feature Selection (Top N)
    # Ensure missing columns are filled with 0 (e.g. OHE columns not present in test set)
    for col in state["top_features"]:
        if col not in df.columns:
            df[col] = 0
            
    return df[state["top_features"]]
