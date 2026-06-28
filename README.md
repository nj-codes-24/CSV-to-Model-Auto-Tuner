# ⚙️ CSV-to-Model Auto-Tuner — End-to-End AutoML Engine

> Upload any classification CSV. Pick your target column. Click Execute. The engine automatically cleans your data, benchmarks 9 algorithms head-to-head, and fine-tunes the winner — with zero training code required.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [The Problem AutoML Solves](#the-problem-automl-solves)
3. [Architecture Overview](#architecture-overview)
4. [Dynamic Data Ingestion — Working on ANY Dataset](#dynamic-data-ingestion--working-on-any-dataset)
5. [The Preprocessing Pipeline — Safe by Design](#the-preprocessing-pipeline--safe-by-design)
6. [Model Benchmarking — The Algorithm Suite](#model-benchmarking--the-algorithm-suite)
7. [Automated Hyperparameter Tuning with RandomizedSearchCV](#automated-hyperparameter-tuning-with-randomizedsearchcv)
8. [The Classification Report — Reading the Full Picture](#the-classification-report--reading-the-full-picture)
9. [The Streamlit App — UI as Observability Layer](#the-streamlit-app--ui-as-observability-layer)
10. [Project File Structure](#project-file-structure)
11. [How to Run This Locally](#how-to-run-this-locally)
12. [Key Concepts Cheat Sheet](#key-concepts-cheat-sheet)

---

## What This Project Does

Traditional ML workflows require a data scientist to manually clean data, write preprocessing code, try different models, tune hyperparameters, and compare results. This is repetitive, time-consuming, and error-prone — especially for someone who just wants to know "which algorithm works best on my data?"

This project is an **AutoML (Automated Machine Learning) engine** that collapses that entire workflow into a single button click. It works on *any* classification dataset, not just one it was hardcoded for. The full pipeline is:

```
User uploads any CSV
         ↓
App detects numeric vs categorical columns automatically
         ↓
User selects target column + columns to drop (IDs, names)
         ↓
Data is split → preprocessing pipeline is built dynamically
         ↓
Stage 1: All 9 algorithms are benchmarked in parallel pipelines
         ↓
Best baseline model is identified
         ↓
Stage 2: RandomizedSearchCV fine-tunes the winner
         ↓
Dashboard displays accuracy comparison, best params, full classification report
```

---

## The Problem AutoML Solves

When you get a new dataset and want to train a classifier, you face three manual steps that this engine automates:

**Step 1 — What type are my columns?** You need to know which columns are numeric (scale them) and which are categorical (encode them). Normally you inspect the data and write this by hand. The engine infers this automatically from dtypes.

**Step 2 — Which algorithm is best?** There's no way to know in advance whether Logistic Regression, Random Forest, or XGBoost will perform best on a given dataset. The data distribution, feature interactions, and class balance all affect this. Normally you'd try each one manually. The engine runs all of them and ranks them.

**Step 3 — What are the optimal hyperparameters?** Every algorithm has knobs (depth of trees, number of neighbors, regularisation strength, etc.) that dramatically affect accuracy. Manually searching this space is called hyperparameter tuning and is tedious. The engine automates this with RandomizedSearchCV.

---

## Architecture Overview

The project is split into two files with a clean separation of concerns:

**`pipeline.py`** — the pure ML engine. No UI code. Contains the preprocessing builder, the model registry, the hyperparameter grid registry, and the two core functions: `run_baseline()` and `run_tuning()`. This is the brain.

**`app.py`** — the Streamlit UI layer. Handles file upload, user configuration, progress display, and rendering of results. It calls functions from `pipeline.py` but contains no ML logic itself. This is the face.

This separation is a professional pattern: your ML logic should be independently testable without needing to run a UI. If you wanted to add a CLI or an API endpoint later, you'd just call `pipeline.py` functions directly.

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

This means the engine doesn't care whether you upload a customer churn dataset, a medical diagnosis dataset, or a loan approval dataset. As long as it's a classification problem, the engine figures out the schema at runtime and builds the appropriate preprocessing pipeline for it.

The function returns three things: the `preprocessor` object (which encodes and scales data), `numeric_cols` (a list of the numeric column names found), and `categorical_cols` (a list of the categorical column names found). The UI displays these in the Feature Schema expander so the user can verify the inference was correct.

---

## The Preprocessing Pipeline — Safe by Design

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

**`SimpleImputer`** — handles missing values. Real-world datasets frequently have `NaN` entries. For numeric columns, we fill gaps with the column mean (a reasonable neutral estimate). For categorical columns, we fill with the most frequent value. Without imputation, most sklearn models throw an error when they encounter `NaN`.

**`StandardScaler`** — normalises numeric features to mean=0, std=1 using `z = (x - mean) / std`. This matters critically for distance-based models (KNN, SVC) where a feature with values in the thousands would completely dominate a feature with values between 0 and 1. Tree-based models (Decision Tree, Random Forest) are not sensitive to scale, but standardising doesn't hurt them either, so it's always safe to apply.

**`OneHotEncoder`** — converts categorical string columns into binary numeric columns. For example a `Color` column with values `{Red, Green, Blue}` becomes three columns: `Color_Red`, `Color_Green`, `Color_Blue`. Each row gets a 1 in one column and 0 in the others. The `drop="if_binary"` argument handles columns that only have two values (like `Yes/No`) by dropping one of the two resulting columns to avoid perfect multicollinearity. `handle_unknown="ignore"` ensures the encoder doesn't crash if a test row contains a category value that wasn't in the training data — it just produces all zeros for that category.

**`ColumnTransformer`** — the orchestrator. It applies `numeric_transformer` to the numeric columns and `categorical_transformer` to the categorical columns, then horizontally concatenates the outputs into a single processed matrix. This is what allows different transformations to be applied to different column types simultaneously.

### The Golden Rule: Fit on Train Only

Inside each model pipeline, the preprocessor is included as the first step:

```python
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model),
])
pipeline.fit(X_train, y_train)
```

When you call `pipeline.fit(X_train, y_train)`, sklearn fits the preprocessor *on the training data only*, learning the training set's mean, std, and category vocabularies. When you later call `pipeline.predict(X_test)`, the preprocessor transforms the test data using those learned training statistics — it does not refit. This is called **preventing data leakage**, and it's the reason we use a Pipeline instead of manually calling `scaler.fit(X)` on the full dataset before splitting.

---

## Model Benchmarking — The Algorithm Suite

The engine tests 9 algorithms. Here's what each one is and why it's included:

```python
MODELS = {
    "Naive Bayes":          GaussianNB(),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(random_state=42),
    "AdaBoost":             AdaBoostClassifier(random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
    "XGBoost":              XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
    "Logistic Regression":  LogisticRegression(max_iter=1000),
    "SVC":                  SVC(),
    "KNN":                  KNeighborsClassifier(),
}
```

**Naive Bayes (`GaussianNB`)** — assumes all features are independent of each other and follow a Gaussian distribution. This is almost never true in reality, which is why it's called "naive." Despite this, it performs surprisingly well on many problems and is extremely fast. It's a useful baseline to see how much more complex models actually gain.

**Decision Tree** — learns a series of if/else rules by splitting the data on feature thresholds. Very interpretable (you can literally visualise the tree), but prone to overfitting on deep trees. `random_state=42` makes results reproducible.

**Random Forest** — an ensemble of many decision trees, each trained on a random subset of rows and features, then combined by majority vote. The randomness prevents the trees from all making the same mistakes and typically outperforms a single tree. One of the most reliable algorithms for tabular data.

**AdaBoost (Adaptive Boosting)** — a boosting algorithm that trains a sequence of weak classifiers (typically shallow trees), where each one focuses more on the examples the previous ones got wrong. The final prediction is a weighted vote of all weak classifiers.

**Gradient Boosting** — also builds trees sequentially, but instead of reweighting misclassified examples, it fits each new tree to the *residual errors* of the previous ensemble. This minimises a loss function via gradient descent in function space. Generally more accurate than AdaBoost but slower.

**XGBoost** — an optimised, regularised, and parallelised implementation of Gradient Boosting. It adds L1/L2 regularisation to prevent overfitting, handles missing values natively, and runs faster than sklearn's GradientBoostingClassifier. Often wins Kaggle competitions. `eval_metric="logloss"` sets the evaluation metric for binary classification; `verbosity=0` suppresses training logs.

**Logistic Regression** — despite the name, this is a classification algorithm. It models the log-odds of the target class as a linear combination of features, then passes this through a sigmoid function to get probabilities. `max_iter=1000` gives the optimiser enough iterations to converge on scaled data.

**SVC (Support Vector Classifier)** — finds the hyperplane in feature space that maximises the margin between classes. Points near the boundary (support vectors) are the only ones that matter for defining the boundary. Very powerful for high-dimensional data but sensitive to feature scale (which is why StandardScaler is important here).

**KNN (K-Nearest Neighbors)** — classifies a new point by looking at the K most similar points in the training set and taking a majority vote. "Similar" means closest in Euclidean distance. Sensitive to scale and slow at prediction time (must compute distances to all training points), but sometimes surprisingly accurate.

### The Benchmarking Loop

```python
def run_baseline(X_train, X_test, y_train, y_test, preprocessor, progress_callback=None):
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
```

Notice the `progress_callback` parameter. This is a function passed in from the UI layer that updates the Streamlit progress bar after each model finishes. This is a clean design pattern: the ML function doesn't know anything about Streamlit — it just calls a callback with progress info. The UI decides what to do with that info. This makes `run_baseline()` reusable outside of Streamlit.

---

## Automated Hyperparameter Tuning with RandomizedSearchCV

Once the baseline winner is identified, it gets promoted to tuning:

```python
def run_tuning(X_train, X_test, y_train, y_test, preprocessor, best_model_name, cv_folds):
    winning_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", MODELS[best_model_name]),
    ])

    grid_search = RandomizedSearchCV(
        winning_pipeline,
        PARAM_GRIDS[best_model_name],
        cv=cv_folds,
        n_iter=5,
        random_state=42,
        n_jobs=1,
        scoring="accuracy",
    )

    grid_search.fit(X_train, y_train)
    ...
```

### GridSearchCV vs RandomizedSearchCV

**GridSearchCV** tries every single combination of hyperparameters in the grid. If you have 3 parameters each with 4 values, that's 4³ = 64 combinations, each trained with k-fold cross-validation. This is exhaustive but slow.

**RandomizedSearchCV** randomly samples `n_iter` combinations from the parameter space. With `n_iter=5` and `cv_folds` cross-validation folds, it runs `5 × cv_folds` model fits. Much faster, and research shows it finds near-optimal parameters almost as reliably as exhaustive search.

### What is k-fold Cross-Validation?

Cross-validation solves a fundamental problem: you need a validation set to evaluate hyperparameters, but if you use your test set for this, you've leaked it into model selection. k-fold CV works like this:

1. Split the training data into k equal chunks (folds)
2. For each combination of hyperparameters, train on k-1 folds and evaluate on the remaining fold
3. Do this k times (each fold gets to be the validation fold once)
4. Average the k accuracy scores to get a robust estimate for that hyperparameter combination
5. Pick the combination with the best average score

The test set is never touched during this process. It's only used once at the very end to report the final tuned accuracy.

### The Parameter Grid

Each model has its own search space defined in `PARAM_GRIDS`. Note the `model__` prefix:

```python
PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10],
    },
    "SVC": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
    },
    ...
}
```

The `model__` prefix is how sklearn's `RandomizedSearchCV` penetrates into a `Pipeline` to address parameters of a specific step. Since our pipeline has a step named `"model"`, the parameter `"model__n_estimators"` targets the `n_estimators` argument of the model inside that step. Without this prefix, sklearn wouldn't know which pipeline step owns the parameter.

### Cleaning Up the Output

```python
best_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}
```

`grid_search.best_params_` returns parameters with the `model__` prefix (e.g. `{"model__n_estimators": 100}`). This one-liner strips the prefix so the displayed output is clean and readable (e.g. `{"n_estimators": 100}`).

---

## The Classification Report — Reading the Full Picture

After tuning, the engine displays a full `classification_report`. Accuracy alone is a misleading metric — a model that predicts the majority class every time can have 95% accuracy on an imbalanced dataset while being completely useless.

The report breaks down performance per class:

| Metric | What It Measures |
|---|---|
| **Precision** | Of all predictions the model labelled as class X, what fraction were actually class X? Measures false positives. |
| **Recall** | Of all actual class X samples, what fraction did the model correctly identify? Measures false negatives. |
| **F1-Score** | Harmonic mean of Precision and Recall: `2 × (P × R) / (P + R)`. Balances both concerns. |
| **Support** | How many actual samples belong to this class in the test set. |
| **Macro avg** | Average of the per-class metrics, treating all classes equally regardless of sample count. |
| **Weighted avg** | Average of per-class metrics, weighted by class support. Accounts for class imbalance. |

When your classes are imbalanced (e.g. 90% class 0, 10% class 1), pay attention to the per-class recall and F1 for the minority class — that's where most models fail silently while showing high overall accuracy.

---

## The Streamlit App — UI as Observability Layer

`app.py` is intentionally thin. It delegates all computation to `pipeline.py` and focuses on making the execution process *visible* to the user.

**Sidebar configuration** — the user selects the target variable from a dropdown populated with the actual column names from their CSV. They can multi-select columns to drop (for IDs, names, or high-cardinality columns that would add noise). Test split ratio and CV folds are configurable sliders.

**Feature Schema expander** — immediately after execution begins, the app shows the user exactly which columns were detected as numeric vs categorical. This is the observability layer: the user can verify the engine inferred their data correctly before looking at model results.

**Progress bar with callback** — the `progress_callback` function is defined in `app.py` and passed into `run_baseline()`. As each model finishes training, the callback fires and updates `st.progress()` and a status caption in real time. This gives the user live feedback instead of a frozen screen.

**Two-column results layout** — the baseline results are displayed as both a sortable table (with the top row highlighted) and a bar chart side-by-side. Different users extract information differently; showing both serves everyone.

**Delta metric for tuning** — the tuned accuracy is shown as a `st.metric` with a delta compared to the baseline. This directly answers the user's question: "did tuning actually help, and by how much?"

---

## Project File Structure

```
CSV-to-Model-Auto-Tuner/
├── app.py              # Streamlit UI layer — handles all user interaction and rendering
├── pipeline.py         # ML engine — preprocessing, benchmarking, tuning (no UI code)
├── requirements.txt    # Python dependencies
└── .gitignore
```

The two-file structure is intentional. `pipeline.py` is a pure Python module that could be imported from a CLI, a Flask API, or a test suite. `app.py` is the delivery mechanism for this particular deployment. Keeping them separate means the ML logic is independently testable.

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

**4. Try it with any classification CSV**

Upload a dataset, select your target column (the thing you want to predict), drop any ID or name columns, and click Execute. The engine handles everything else.

---

## Key Concepts Cheat Sheet

| Concept | One-line explanation |
|---|---|
| **AutoML** | Automating the manual steps of model selection, preprocessing, and tuning |
| **Type inference** | Using `select_dtypes()` to detect numeric vs categorical columns at runtime without hardcoding |
| **ColumnTransformer** | Applies different transformations to different column subsets, then merges them |
| **Pipeline** | Chains preprocessing + model into one object so fit/predict are always consistent |
| **Data leakage** | When test set information contaminates training — prevented by fitting preprocessor only on train data |
| **SimpleImputer** | Fills missing values with mean (numeric) or most frequent value (categorical) |
| **StandardScaler** | Normalises features to mean=0, std=1; critical for distance-based models |
| **OneHotEncoder** | Converts categorical string columns into binary columns, one per category |
| **`drop="if_binary"`** | Drops one redundant column when a category only has two values |
| **`handle_unknown="ignore"`** | Prevents encoder from crashing on unseen category values at inference time |
| **Ensemble methods** | Models like Random Forest and Boosting that combine many weak learners into a stronger one |
| **Boosting** | Sequentially training models where each corrects the errors of the previous one (AdaBoost, GBM, XGBoost) |
| **Bagging** | Training many models on random data subsets and averaging their predictions (Random Forest) |
| **SVC** | Finds the maximum-margin hyperplane separating classes in feature space |
| **KNN** | Classifies by majority vote among K nearest training points in feature space |
| **RandomizedSearchCV** | Samples random hyperparameter combinations from a grid to find near-optimal settings efficiently |
| **GridSearchCV** | Exhaustively tries every combination in the hyperparameter grid |
| **k-fold CV** | Splits training data into k folds; each fold serves as validation once — averages k accuracy scores |
| **`model__param`** | Prefix syntax to address parameters of a named step inside a sklearn Pipeline |
| **Accuracy** | Fraction of predictions that are correct — misleading for imbalanced datasets |
| **Precision** | Of predicted positives, what fraction are truly positive — penalises false positives |
| **Recall** | Of actual positives, what fraction were caught — penalises false negatives |
| **F1-Score** | Harmonic mean of precision and recall — balanced single metric for classification |
| **`n_jobs=1`** | Run processes sequentially to prevent out-of-memory errors on cloud platforms |
| **`progress_callback`** | A function passed into ML code so the UI can update without ML code knowing about the UI |
