# ⚙️ CSV-to-Model Auto-Tuner — End-to-End AutoML Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://csv-to-model-auto-tuner.streamlit.app/)

> Upload any classification CSV. Pick your target column. Click Execute. The engine automatically cleans your data, benchmarks 9 algorithms head-to-head, and fine-tunes the winner — with zero training code required.

**Live Demo:** [https://csv-to-model-auto-tuner.streamlit.app/](https://csv-to-model-auto-tuner.streamlit.app/)

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [The Problem AutoML Solves](#the-problem-automl-solves)
3. [Architecture Overview](#architecture-overview)
4. [Data Cleaning (Auto-Cleaner) — Safe by Design](#data-cleaning-auto-cleaner--safe-by-design)
5. [Dynamic Data Ingestion — Working on ANY Dataset](#dynamic-data-ingestion--working-on-any-dataset)
6. [The Preprocessing Pipeline](#the-preprocessing-pipeline)
7. [Model Benchmarking — The Algorithm Suite](#model-benchmarking--the-algorithm-suite)
8. [Automated Hyperparameter Tuning with RandomizedSearchCV](#automated-hyperparameter-tuning-with-randomizedsearchcv)
9. [The Classification Report — Reading the Full Picture](#the-classification-report--reading-the-full-picture)
10. [The Streamlit App — UI as Observability Layer](#the-streamlit-app--ui-as-observability-layer)
11. [Project File Structure](#project-file-structure)
12. [How to Run This Locally](#how-to-run-this-locally)
13. [Key Concepts Cheat Sheet](#key-concepts-cheat-sheet)

---

## What This Project Does

Traditional ML workflows require a data scientist to manually clean data, write preprocessing code, try different models, tune hyperparameters, and compare results. This is repetitive, time-consuming, and error-prone — especially for someone who just wants to know "which algorithm works best on my data?"

This project is an **AutoML (Automated Machine Learning) engine** that collapses that entire workflow into a single button click. It works on *any* classification dataset, not just one it was hardcoded for. The full pipeline is:

```text
User uploads any CSV
         ↓
User selects target variable
         ↓
Stage 1: Auto-Cleaner drops useless/noisy columns (IDs, empty, zero variance, multicollinear)
         ↓
Data is split → preprocessing pipeline is built dynamically (infers numeric vs categorical)
         ↓
Stage 2: All 9 algorithms are benchmarked in parallel pipelines
         ↓
Best baseline model is identified
         ↓
Stage 3: RandomizedSearchCV fine-tunes the winner
         ↓
Stage 4: Dashboard displays recommended optimal hyperparameters and full classification report
```

---

## The Problem AutoML Solves

When you get a new dataset and want to train a classifier, you face manual steps that this engine automates:

**Step 1 — Is my data clean?** Models struggle with useless columns like IDs, high-cardinality strings, or highly correlated features. Normally you have to find and drop these manually. The engine uses a programmatic auto-cleaner to do this for you.

**Step 2 — What type are my columns?** You need to know which columns are numeric (scale them) and which are categorical (encode them). Normally you inspect the data and write this by hand. The engine infers this automatically from dtypes.

**Step 3 — Which algorithm is best?** There's no way to know in advance whether Logistic Regression, Random Forest, or XGBoost will perform best on a given dataset. The engine runs all of them and ranks them.

**Step 4 — What are the optimal hyperparameters?** Every algorithm has knobs that dramatically affect accuracy. Manually searching this space is tedious. The engine automates this with RandomizedSearchCV.

---

## Architecture Overview

The project is split into two files with a clean separation of concerns:

**`pipeline.py`** — the pure ML engine. No UI code. Contains the data cleaner, preprocessing builder, the model registry, the hyperparameter grid registry, and the core functions: `auto_drop_columns()`, `run_baseline()`, and `run_tuning()`. This is the brain.

**`app.py`** — the Streamlit UI layer. Handles file upload, user configuration, progress display, and rendering of results. It calls functions from `pipeline.py` but contains no ML logic itself. This is the face.

This separation is a professional pattern: your ML logic should be independently testable without needing to run a UI.

---

## Data Cleaning (Auto-Cleaner) — Safe by Design

Before any ML happens, the engine automatically purges noisy or useless columns using the `auto_drop_columns` function. It safely identifies:

1. **Empty Columns:** Columns with 100% NaN values.
2. **Zero Variance:** Columns with only a single unique value (constant).
3. **High Cardinality:** Categorical strings with too many unique values (like names or hashes) and numeric IDs (like RowNumber, CustomerId).
4. **Multicollinearity:** Numeric features that have >0.95 correlation with each other, dropping the redundant ones to prevent model confusion.

This prevents common errors and improves baseline model performance without manual intervention.

---

## Dynamic Data Ingestion — Working on ANY Dataset

The most important design decision in this project is the `build_preprocessor()` function:

```python
def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    ...
```

**`select_dtypes()`** is the key. Instead of hardcoding column names, the function asks pandas: "which columns in this DataFrame are numeric types?" and "which are string/categorical types?" It returns those column names dynamically.

This means the engine doesn't care whether you upload a customer churn dataset or a medical diagnosis dataset. The engine figures out the schema at runtime.

---

## The Preprocessing Pipeline

```python
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
```

### Why each piece exists

**`SimpleImputer`** — handles missing values by filling gaps with the column mean (numeric) or most frequent value (categorical).
**`StandardScaler`** — normalises numeric features to mean=0, std=1. Critical for distance-based models (KNN, SVC).
**`OneHotEncoder`** — converts categorical string columns into binary numeric columns.
**`ColumnTransformer`** — applies `numeric_transformer` to numeric columns and `categorical_transformer` to categorical columns, then concatenates them.

### The Golden Rule: Fit on Train Only

Inside each model pipeline, the preprocessor is included as the first step:
```python
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model),
])
pipeline.fit(X_train, y_train)
```
When you call `pipeline.fit(X_train, y_train)`, sklearn fits the preprocessor *on the training data only*. When you later predict on the test set, the preprocessor uses those learned training statistics. This prevents **data leakage**.

---

## Model Benchmarking — The Algorithm Suite

The engine tests 9 algorithms:

- **Naive Bayes (`GaussianNB`)** — Fast baseline assuming feature independence.
- **Decision Tree** — Interpretable tree splits, prone to overfitting if deep.
- **Random Forest** — Ensemble of trees using random subsets (Bagging). Highly reliable.
- **AdaBoost** — Boosting algorithm training weak classifiers on previous errors.
- **Gradient Boosting** — Fits new trees to the residual errors of the previous ensemble.
- **XGBoost** — Optimised, regularised, parallelised Gradient Boosting. Often the top performer.
- **Logistic Regression** — Linear model for classification predicting log-odds.
- **SVC** — Finds hyperplane maximising margin between classes.
- **KNN** — Classifies based on K nearest neighbors in feature space.

---

## Automated Hyperparameter Tuning with RandomizedSearchCV

Once the baseline winner is identified, it gets promoted to tuning:

```python
def run_tuning(X_train, X_test, y_train, y_test, preprocessor, best_model_name, cv_folds):
    ...
    grid_search = RandomizedSearchCV(
        winning_pipeline,
        PARAM_GRIDS[best_model_name],
        cv=cv_folds,
        n_iter=10,
        random_state=42,
        ...
    )
```

**RandomizedSearchCV** randomly samples `n_iter` combinations from the parameter space. It's much faster than exhaustive `GridSearchCV` while finding near-optimal parameters. It uses **k-fold cross-validation** to evaluate hyperparameters without touching the test set.

---

## The Classification Report — Reading the Full Picture

After tuning, the engine displays a full `classification_report`. Accuracy alone is misleading for imbalanced data. The report includes:

- **Precision:** Of all predicted class X, what fraction were actually class X? (Minimises false positives)
- **Recall:** Of all actual class X, what fraction were identified? (Minimises false negatives)
- **F1-Score:** Harmonic mean of Precision and Recall.
- **Support:** Number of actual samples in each class.

---

## The Streamlit App — UI as Observability Layer

`app.py` delegates computation to `pipeline.py` and focuses on observability:

- **Sidebar configuration:** Upload CSV, pick target variable, set test split and CV folds.
- **Data Cleaning Insights:** Shows exactly which columns were automatically dropped and why.
- **Progress Tracking:** Live progress bar during benchmarking via callbacks.
- **Results Layout:** Baseline results shown as both a sortable table and bar chart.
- **Tuning Delta:** Shows baseline vs tuned accuracy to answer "did tuning help?".

---

## Project File Structure

```text
CSV-to-Model-Auto-Tuner/
├── app.py              # Streamlit UI layer — user interaction and rendering
├── pipeline.py         # ML engine — data cleaner, benchmarking, tuning (no UI code)
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## How to Run This Locally

**1. Clone the repository**
```bash
git clone https://github.com/nj-codes-24/CSV-to-Model-Auto-Tuner.git
cd CSV-to-Model-Auto-Tuner
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the app**
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`

---

## Key Concepts Cheat Sheet

| Concept | One-line explanation |
|---|---|
| **AutoML** | Automating model selection, preprocessing, and tuning |
| **Type inference** | Detecting numeric vs categorical columns at runtime |
| **Data leakage** | Contaminating training with test info — prevented by fitting preprocessor only on train |
| **Ensemble methods** | Combining many weak learners into a stronger one (Random Forest, Boosting) |
| **RandomizedSearchCV** | Sampling random hyperparameter combinations efficiently |
| **k-fold CV** | Splitting train data into k folds to evaluate models robustly |
| **F1-Score** | Harmonic mean of precision and recall |
