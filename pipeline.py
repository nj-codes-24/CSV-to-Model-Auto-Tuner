import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC  # ← O(N) replacement for SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# ── Tuning knobs ────────────────────────────────────────────────────────────────
# If the training split is larger than this, models that don't scale well
# (KNN, SVC-variant) will be trained on a random subsample instead.
_SAMPLE_THRESHOLD = 50_000   # rows
_SLOW_MODEL_SAMPLE = 20_000  # rows used for slow models when above threshold
# ────────────────────────────────────────────────────────────────────────────────


def auto_drop_columns(df: pd.DataFrame, target_var: str) -> dict:
    """
    Automatically detects and returns a dictionary of useless/noisy columns to drop, grouped by reason:
    1. All NaNs (Empty)
    2. Single unique value (Zero variance)
    3. High cardinality strings (IDs/Names/Hashes)
    4. Multicollinear numeric columns (>0.95 correlation)
    """
    drop_reasons = {
        "empty": [],
        "zero_variance": [],
        "high_cardinality": [],
        "multicollinear": []
    }

    for col in df.columns:
        if col == target_var:
            continue

        # 1. Empty Columns
        if df[col].isna().all():
            drop_reasons["empty"].append(col)
            continue

        # 2. Zero Variance
        if df[col].nunique(dropna=True) <= 1:
            drop_reasons["zero_variance"].append(col)
            continue

        # 3. High Cardinality (for categorical/string types)
        num_unique = df[col].nunique()
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            if num_unique > 100 or num_unique > len(df) * 0.5:
                drop_reasons["high_cardinality"].append(col)
                continue

        # 4. Numeric IDs (like RowNumber, CustomerId)
        if pd.api.types.is_numeric_dtype(df[col]):
            col_lower = str(col).lower()
            is_id_name = any(x in col_lower for x in ['id', 'row', 'index', 'uuid'])
            if num_unique == len(df) or (is_id_name and num_unique > len(df) * 0.5):
                drop_reasons["high_cardinality"].append(col)
                continue

    # 5. Multicollinearity
    dropped_so_far = set(
        drop_reasons["empty"] + drop_reasons["zero_variance"] + drop_reasons["high_cardinality"]
    )
    remaining_num_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c not in dropped_so_far and c != target_var
    ]

    if len(remaining_num_cols) > 1:
        corr_matrix = df[remaining_num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        drop_reasons["multicollinear"] = to_drop

    return drop_reasons


# ── Model registry ──────────────────────────────────────────────────────────────
# LinearSVC: same algorithm family as SVC but O(N) — safe on 200k rows.
# Wrapped in CalibratedClassifierCV so predict_proba works if needed downstream.
# n_jobs=-1 on tree ensembles uses all CPU cores without blocking the GIL for long.
MODELS = {
    "Decision Tree":      DecisionTreeClassifier(random_state=42),
    "Random Forest":      RandomForestClassifier(n_jobs=-1, random_state=42),
    "AdaBoost":           AdaBoostClassifier(random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(random_state=42),
    "XGBoost":            XGBClassifier(random_state=42, eval_metric="logloss",
                                        verbosity=0, n_jobs=-1, tree_method="hist"),
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "LinearSVC":          CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
    "KNN":                KNeighborsClassifier(n_jobs=-1),
}

# Models that still don't scale to huge N — they will be trained on a subsample
_SLOW_MODELS = {"KNN", "LinearSVC"}

PARAM_GRIDS = {
    "Decision Tree":      {
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
    },
    "Random Forest":      {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10],
    },
    "AdaBoost":           {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 1.0],
    },
    "Gradient Boosting":  {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
    "XGBoost":            {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "LinearSVC":          {
        "model__estimator__C": [0.1, 1, 10],
    },
    "KNN":                {
        "model__n_neighbors": [3, 5, 7],
        "model__weights": ["uniform", "distance"],
    },
}
# ────────────────────────────────────────────────────────────────────────────────


def build_preprocessor(X: pd.DataFrame):
    """Infer numeric and categorical columns, return a ColumnTransformer."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])

    # sparse_output=True (the default) keeps the OHE matrix sparse → much less RAM
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def _maybe_subsample(X_train, y_train, model_name: str):
    """
    Return a (possibly subsampled) view of training data.
    Slow models get capped at _SLOW_MODEL_SAMPLE rows when the dataset is large.
    Fast models always see the full training set.
    """
    n = len(X_train)
    if model_name in _SLOW_MODELS and n > _SAMPLE_THRESHOLD:
        idx = np.random.RandomState(42).choice(n, size=_SLOW_MODEL_SAMPLE, replace=False)
        if isinstance(X_train, pd.DataFrame):
            return X_train.iloc[idx], y_train.iloc[idx]
        return X_train[idx], y_train[idx]
    return X_train, y_train


def run_baseline(X_train, X_test, y_train, y_test, preprocessor, progress_callback=None):
    """
    Train all models and return a dict of {model_name: accuracy}.
    progress_callback(idx, total, name) is called after each model finishes.

    Large-dataset safety:
    - Slow models (KNN, LinearSVC) are trained on a subsample if N > _SAMPLE_THRESHOLD.
    - Sparse OHE matrices are kept sparse throughout.
    - n_jobs=-1 on compatible models lets scikit-learn parallelise without blocking.
    """
    results = {}
    best_accuracy = 0
    best_model_name = ""

    for idx, (name, model) in enumerate(MODELS.items()):
        X_tr, y_tr = _maybe_subsample(X_train, y_train, name)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_tr, y_tr)
        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        results[name] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

        if progress_callback:
            progress_callback(idx + 1, len(MODELS), name)

    return results, best_model_name, best_accuracy


def run_tuning(X_train, X_test, y_train, y_test, preprocessor, best_model_name, cv_folds):
    """
    Run RandomizedSearchCV on the winning model and return tuning results.

    Large-dataset safety:
    - Slow models are tuned on a subsample (same cap as baseline).
    - The final winner is retrained on the full training set.
    """
    winning_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", MODELS[best_model_name]),
    ])

    X_tune, y_tune = _maybe_subsample(X_train, y_train, best_model_name)

    grid_search = RandomizedSearchCV(
        winning_pipeline,
        PARAM_GRIDS[best_model_name],
        cv=cv_folds,
        n_iter=10,
        random_state=42,
        n_jobs=1,          # keep 1 here — outer parallelism can OOM inside CV
        scoring="accuracy",
        verbose=3,
    )

    grid_search.fit(X_tune, y_tune)

    final_model = grid_search.best_estimator_

    # Retrain winning architecture on the FULL training set
    final_model.fit(X_train, y_train)

    y_pred_final = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)

    best_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}
    report = classification_report(y_test, y_pred_final, output_dict=True)

    return final_accuracy, best_params, report, final_model