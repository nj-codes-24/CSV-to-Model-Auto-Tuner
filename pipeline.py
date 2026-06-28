import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


MODELS = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier(),
}

PARAM_GRIDS = {
    "Naive Bayes": {"model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    "Decision Tree": {
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10],
    },
    "AdaBoost": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 1.0],
    },
    "Gradient Boosting": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
    "XGBoost": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "SVC": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
    "KNN": {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]},
}


def build_preprocessor(X: pd.DataFrame):
    """Infer numeric and categorical columns, return a fitted ColumnTransformer."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def run_baseline(X_train, X_test, y_train, y_test, preprocessor, progress_callback=None):
    """
    Train all models and return a dict of {model_name: accuracy}.
    progress_callback(idx, total, name) is called after each model finishes.
    """
    results = {}
    best_accuracy = 0
    best_model_name = ""

    for idx, (name, model) in enumerate(MODELS.items()):
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
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
    Run GridSearchCV on the winning model and return tuning results.
    """
    winning_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", MODELS[best_model_name]),
    ])

    grid_search = GridSearchCV(
        winning_pipeline,
        PARAM_GRIDS[best_model_name],
        cv=cv_folds,
        n_jobs=-1,
        scoring="accuracy",
    )
    grid_search.fit(X_train, y_train)

    final_model = grid_search.best_estimator_
    y_pred_final = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)

    best_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}
    report = classification_report(y_test, y_pred_final, output_dict=True)

    return final_accuracy, best_params, report, final_model