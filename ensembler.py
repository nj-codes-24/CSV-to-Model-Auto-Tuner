"""
ensembler.py — Advanced Ensembling Module
=========================================
Builds Voting and Stacking ensembles from the top N models
produced during the baseline stage.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.base import clone

# ── Classification imports ───────────────────────────────────────────────────
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Regression imports ───────────────────────────────────────────────────────
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def get_top_n_models(mdl_results: dict, all_fitted_models: dict, sort_col: str, n: int = 3):
    """
    Extract the top N distinct models based on their score.
    Returns a list of (name, cloned_unfitted_model) tuples.
    """
    sorted_names = sorted(
        mdl_results.keys(),
        key=lambda k: mdl_results[k].get(sort_col, -float('inf')),
        reverse=True
    )
    
    top_models = []
    for name in sorted_names:
        if name in all_fitted_models and all_fitted_models[name] is not None:
            # We clone the model to get a fresh, unfitted instance 
            # for the Ensembles to use safely during CV
            fresh_model = clone(all_fitted_models[name])
            top_models.append((name, fresh_model))
            if len(top_models) == n:
                break
                
    return top_models


def _ensure_predict_proba(models: list):
    """
    Wraps classification models that lack predict_proba natively
    (e.g., LinearSVC) in a CalibratedClassifierCV so soft voting works.
    """
    calibrated_models = []
    for name, model in models:
        # Check if the base class has predict_proba
        if not hasattr(model, "predict_proba"):
            calibrated = CalibratedClassifierCV(estimator=model, cv=3)
            calibrated_models.append((name, calibrated))
        else:
            calibrated_models.append((name, model))
    return calibrated_models


def build_classification_ensembles(
    X_train, X_test, y_train, y_test, top_models: list, scoring_metric: str, progress_callback=None
):
    """
    Builds and evaluates a VotingClassifier and StackingClassifier.
    Returns (results_dict, best_ensemble_name, best_score, best_fitted_ensemble).
    """
    results = {}
    best_score = -1
    best_name = ""
    best_ensemble = None
    
    # 1. Voting Classifier (Soft)
    if progress_callback:
        progress_callback(0, 2, "Voting Classifier (Soft)", status="start")
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Building VotingClassifier...")
    try:
        calibrated_models = _ensure_predict_proba(top_models)
        voting_clf = VotingClassifier(estimators=calibrated_models, voting='soft')
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Fitting VotingClassifier...")
        voting_clf.fit(X_train, y_train)
        
        y_pred = voting_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        results["Voting Classifier"] = {"Accuracy": round(acc, 4), "F1 (weighted)": round(f1, 4)}
        score = acc if scoring_metric == "accuracy" else f1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Voting Classifier — Accuracy: {acc:.4f}, F1: {f1:.4f}\n")
        
        if score > best_score:
            best_score = score
            best_name = "Voting Classifier"
            best_ensemble = voting_clf
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Voting Classifier failed: {e}\n")
        results["Voting Classifier"] = {"Accuracy": 0.0, "F1 (weighted)": 0.0}
        
    if progress_callback:
        progress_callback(1, 2, "Voting Classifier (Soft)", status="end")


    # 2. Stacking Classifier
    if progress_callback:
        progress_callback(1, 2, "Stacking Classifier", status="start")
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Building StackingClassifier...")
    try:
        # LogisticRegression is a robust meta-learner
        meta_learner = LogisticRegression(max_iter=1000)
        
        stacking_clf = StackingClassifier(
            estimators=top_models,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=1
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Fitting StackingClassifier...")
        stacking_clf.fit(X_train, y_train)
        
        y_pred = stacking_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        results["Stacking Classifier"] = {"Accuracy": round(acc, 4), "F1 (weighted)": round(f1, 4)}
        score = acc if scoring_metric == "accuracy" else f1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Stacking Classifier — Accuracy: {acc:.4f}, F1: {f1:.4f}\n")
        
        if score > best_score:
            best_score = score
            best_name = "Stacking Classifier"
            best_ensemble = stacking_clf
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Stacking Classifier failed: {e}\n")
        results["Stacking Classifier"] = {"Accuracy": 0.0, "F1 (weighted)": 0.0}
        
    if progress_callback:
        progress_callback(2, 2, "Stacking Classifier", status="end")

    return results, best_name, best_score, best_ensemble


def build_regression_ensembles(
    X_train, X_test, y_train, y_test, top_models: list, progress_callback=None
):
    """
    Builds and evaluates a VotingRegressor and StackingRegressor.
    Returns (results_dict, best_ensemble_name, best_score, best_fitted_ensemble).
    """
    results = {}
    best_r2 = -float("inf")
    best_name = ""
    best_ensemble = None
    
    # 1. Voting Regressor
    if progress_callback:
        progress_callback(0, 2, "Voting Regressor", status="start")
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Building VotingRegressor...")
    try:
        voting_reg = VotingRegressor(estimators=top_models)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Fitting VotingRegressor...")
        voting_reg.fit(X_train, y_train)
        
        y_pred = voting_reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results["Voting Regressor"] = {"R²": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Voting Regressor — R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = "Voting Regressor"
            best_ensemble = voting_reg
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Voting Regressor failed: {e}\n")
        results["Voting Regressor"] = {"R²": 0.0, "MAE": 0.0, "RMSE": 0.0}
        
    if progress_callback:
        progress_callback(1, 2, "Voting Regressor", status="end")


    # 2. Stacking Regressor
    if progress_callback:
        progress_callback(1, 2, "Stacking Regressor", status="start")
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Building StackingRegressor...")
    try:
        # Ridge is a robust meta-learner for regression
        meta_learner = Ridge()
        
        stacking_reg = StackingRegressor(
            estimators=top_models,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=1
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Fitting StackingRegressor...")
        stacking_reg.fit(X_train, y_train)
        
        y_pred = stacking_reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results["Stacking Regressor"] = {"R²": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Stacking Regressor — R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = "Stacking Regressor"
            best_ensemble = stacking_reg
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Stacking Regressor failed: {e}\n")
        results["Stacking Regressor"] = {"R²": 0.0, "MAE": 0.0, "RMSE": 0.0}
        
    if progress_callback:
        progress_callback(2, 2, "Stacking Regressor", status="end")

    return results, best_name, best_r2, best_ensemble
