"""
pipeline.py — ML Engine (Classification + Regression)
=====================================================
Model registries, hyperparameter grids, baseline benchmarking, and
hyperparameter tuning for both classification and regression tasks.

This module receives pre-cleaned, pre-encoded, feature-selected data
from eda_engine.py and focuses purely on model training and evaluation.
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ── Classification imports ───────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# ── Regression imports ───────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
)
from xgboost import XGBRegressor

# ── Shared imports ───────────────────────────────────────────────────────────
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
)

# ── Tuning knobs ─────────────────────────────────────────────────────────────
_SAMPLE_THRESHOLD = 50_000
_SLOW_MODEL_SAMPLE = 20_000


# =============================================================================
# CLASSIFICATION REGISTRY
# =============================================================================

def _build_classification_models(use_class_weight: bool = False):
    """Build the classification model registry, optionally with balanced class weights."""
    cw = "balanced" if use_class_weight else None
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, n_jobs=-1, class_weight=cw,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight=cw,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42, class_weight=cw),
        ),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_jobs=-1, random_state=42, class_weight=cw,
        ),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            random_state=42, eval_metric="logloss",
            verbosity=0, n_jobs=-1, tree_method="hist",
        ),
    }


CLASSIFICATION_PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
    },
    "LinearSVC": {"estimator__C": [0.01, 0.1, 1, 10]},
    "KNN": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
}

_SLOW_MODELS = {"KNN", "LinearSVC"}


# =============================================================================
# REGRESSION REGISTRY
# =============================================================================

def _build_regression_models():
    """Build the regression model registry with fresh instances each time."""
    return {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_jobs=-1, random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            random_state=42, verbosity=0, n_jobs=-1, tree_method="hist",
        ),
    }

REGRESSION_PARAM_GRIDS = {
    "Linear Regression": {},  # No hyperparameters
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
    "Decision Tree": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
}


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _maybe_subsample(X_train, y_train, model_name: str):
    """
    Return a (possibly subsampled) view of training data.
    Slow models get capped at _SLOW_MODEL_SAMPLE rows when the dataset is large.
    """
    n = len(X_train)
    if model_name in _SLOW_MODELS and n > _SAMPLE_THRESHOLD:
        idx = np.random.RandomState(42).choice(n, size=_SLOW_MODEL_SAMPLE, replace=False)
        if isinstance(X_train, pd.DataFrame):
            X_sub = X_train.iloc[idx]
        else:
            X_sub = X_train[idx]
        # Guard: handle both pandas Series and numpy arrays for y_train
        if isinstance(y_train, pd.Series):
            y_sub = y_train.iloc[idx]
        else:
            y_sub = y_train[idx]
        return X_sub, y_sub
    return X_train, y_train


def _sklearn_scoring(metric: str) -> str:
    """Map our metric names to sklearn scoring strings."""
    mapping = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    return mapping.get(metric, metric)


# =============================================================================
# CLASSIFICATION BASELINE & TUNING
# =============================================================================

def run_baseline_classification(
    X_train, X_test, y_train, y_test,
    scoring_metric: str = "accuracy",
    use_class_weight: bool = False,
    progress_callback=None,
):
    """
    Train all classification models and return results ranked by the chosen metric.

    Returns (results_dict, best_model_name, best_score, best_fitted_model).
    results_dict maps model_name → {accuracy, f1_weighted}.
    """
    models = _build_classification_models(use_class_weight)
    results = {}
    best_score = -1
    best_model_name = ""
    best_fitted_model = None

    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx, len(models), name, status="start")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Initializing {name}...")

        try:
            X_tr, y_tr = _maybe_subsample(X_train, y_train, name)
            if len(X_tr) < len(X_train):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Subsampling to {len(X_tr):,} rows...")

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Fitting model...")
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            results[name] = {"Accuracy": round(acc, 4), "F1 (weighted)": round(f1, 4)}

            score = acc if scoring_metric == "accuracy" else f1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {name} — Accuracy: {acc:.4f}, F1: {f1:.4f}\n")

            if score > best_score:
                best_score = score
                best_model_name = name
                best_fitted_model = model
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {name} failed: {e}\n")
            results[name] = {"Accuracy": 0.0, "F1 (weighted)": 0.0}

        if progress_callback:
            progress_callback(idx + 1, len(models), name, status="end")

    # Guard: no model succeeded
    if not best_model_name:
        raise RuntimeError(
            "All classification models failed to train. "
            "Please check your data for issues (e.g., NaN values, incompatible dtypes)."
        )

    # Retrain best model on full (non-subsampled) training data if it was subsampled
    if best_model_name in _SLOW_MODELS and len(X_train) > _SAMPLE_THRESHOLD:
        best_fitted_model.fit(X_train, y_train)

    return results, best_model_name, best_score, best_fitted_model


def run_tuning_classification(
    X_train, X_test, y_train, y_test,
    best_model_name: str, cv_folds: int,
    scoring_metric: str = "accuracy",
    use_class_weight: bool = False,
):
    """
    Run RandomizedSearchCV on the winning classification model.

    Returns (final_accuracy, final_f1, best_params, report_dict, final_model).
    """
    models = _build_classification_models(use_class_weight)
    model = models[best_model_name]

    X_tune, y_tune = _maybe_subsample(X_train, y_train, best_model_name)

    param_grid = CLASSIFICATION_PARAM_GRIDS.get(best_model_name, {})
    if not param_grid:
        # No hyperparameters to tune — just refit on full data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)
        return acc, f1, {}, report, model

    # Guard: cap cv_folds to the smallest class count to avoid StratifiedKFold crash
    if hasattr(y_tune, 'value_counts'):
        min_class_count = y_tune.value_counts().min()
    else:
        _, counts = np.unique(y_tune, return_counts=True)
        min_class_count = counts.min() if len(counts) > 0 else 2
    safe_cv = min(cv_folds, max(2, int(min_class_count)))

    search = RandomizedSearchCV(
        model, param_grid,
        cv=safe_cv, n_iter=min(10, _grid_combinations(param_grid)),
        random_state=42, n_jobs=1,
        scoring=_sklearn_scoring(scoring_metric),
        verbose=3,
    )
    search.fit(X_tune, y_tune)

    final_model = search.best_estimator_
    final_model.fit(X_train, y_train)  # Retrain on full train set

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    best_params = search.best_params_

    return acc, f1, best_params, report, final_model


# =============================================================================
# REGRESSION BASELINE & TUNING
# =============================================================================

def run_baseline_regression(
    X_train, X_test, y_train, y_test,
    progress_callback=None,
):
    """
    Train all regression models and return results ranked by R².

    Returns (results_dict, best_model_name, best_r2, best_fitted_model).
    results_dict maps model_name → {R², MAE, RMSE}.
    """
    models = _build_regression_models()
    results = {}
    best_r2 = -float("inf")
    best_model_name = ""
    best_fitted_model = None

    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx, len(models), name, status="start")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Initializing {name}...")

        try:
            X_tr, y_tr = _maybe_subsample(X_train, y_train, name)
            if len(X_tr) < len(X_train):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Subsampling to {len(X_tr):,} rows...")

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Fitting model...")
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                "R²": round(r2, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
            }
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {name} — R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}\n")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_fitted_model = model
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {name} failed: {e}\n")
            results[name] = {"R²": 0.0, "MAE": 0.0, "RMSE": 0.0}

        if progress_callback:
            progress_callback(idx + 1, len(models), name, status="end")

    # Guard: no model succeeded
    if not best_model_name:
        raise RuntimeError(
            "All regression models failed to train. "
            "Please check your data for issues (e.g., NaN values, incompatible dtypes)."
        )

    # Retrain best model on full (non-subsampled) training data if it was subsampled
    if best_model_name in _SLOW_MODELS and len(X_train) > _SAMPLE_THRESHOLD:
        best_fitted_model.fit(X_train, y_train)

    return results, best_model_name, best_r2, best_fitted_model


def run_tuning_regression(
    X_train, X_test, y_train, y_test,
    best_model_name: str, cv_folds: int,
):
    """
    Run RandomizedSearchCV on the winning regression model.

    Returns (final_r2, final_mae, final_rmse, best_params, final_model).
    """
    models = _build_regression_models()
    model = models[best_model_name]

    X_tune, y_tune = _maybe_subsample(X_train, y_train, best_model_name)

    param_grid = REGRESSION_PARAM_GRIDS.get(best_model_name, {})
    if not param_grid:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, mae, rmse, {}, model

    # Guard: cap cv_folds to sample count to avoid KFold crash
    safe_cv = min(cv_folds, max(2, len(X_tune)))

    search = RandomizedSearchCV(
        model, param_grid,
        cv=safe_cv, n_iter=min(10, _grid_combinations(param_grid)),
        random_state=42, n_jobs=1,
        scoring="r2",
        verbose=3,
    )
    search.fit(X_tune, y_tune)

    final_model = search.best_estimator_
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, mae, rmse, search.best_params_, final_model


def _grid_combinations(param_grid: dict) -> int:
    """Count total combinations in a param grid."""
    n = 1
    for v in param_grid.values():
        if isinstance(v, list):
            n *= len(v)
    return n