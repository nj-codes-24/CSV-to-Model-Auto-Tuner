# ⚙️ CSV-to-Model Auto-Tuner (AutoML Engine)

An end-to-end Machine Learning pipeline and interactive web application that transforms raw, messy CSV datasets into production-ready, hyperparameter-tuned machine learning models in a single click.

Built with **Python**, **Streamlit**, and **Scikit-Learn**, this tool bridges the gap between raw data and deployment by automating rigorous, industry-standard Exploratory Data Analysis (EDA) and model benchmarking.

---

## ✨ Key Features

### 🧪 1. Leakage-Free, Industry-Standard EDA
The engine automatically detects whether your dataset requires **Classification** or **Regression** and executes an 8-phase data preparation pipeline. All transformations are fitted strictly on the training set to prevent data leakage.
- **Smart Imputation:** Automatically median-fills skewed numeric data, mean-fills normal distributions, and handles categorical nulls.
- **Statistical Feature Selection:** Drops multicollinear features (>0.80 correlation) and utilizes ANOVA (Regression) or Chi-Square (Classification) F-tests to drop statistically insignificant columns.
- **Outlier & Skewness Handling:** Caps outliers using IQR boundaries and applies `log1p` transformations to highly skewed target variables to stabilize variance.
- **Hybrid Encoding:** Dynamically routes low-cardinality features (≤10 unique) to One-Hot Encoding and high-cardinality features to Target Encoding.

### 🌲 2. RF Signal Extraction
Before training, the engine runs a Random Forest Feature Importance Scan to rank features, stripping away noise and isolating only the top *N* strongest predictive signals.

### 📊 3. Automated Benchmarking & Tuning
The cleaned data is evaluated against a gauntlet of 8 algorithms (including XGBoost, Gradient Boosting, Random Forest, Ridge/Lasso, etc.). 
- **Smart Metrics:** The engine dynamically selects the right evaluation metric (e.g., switching to F1-Weighted if it detects severe class imbalance, or R² for regression).
- **Hyperparameter Tuning:** The winning baseline model is passed through a `RandomizedSearchCV` to find optimal hyperparameters via Cross-Validation.

### 💻 4. Premium "Glassmorphic" UI & Handoff
Ditching standard dashboards, the app features a sleek, dark-mode glassmorphic interface with a linear "Wizard" execution flow.
- **Executive Log:** Watch the engine make decisions in real-time. The UI prints exactly *why* a column was dropped or *how* a value was imputed.
- **Single-Click Deployment:** Download the final tuned model as a `.pkl` file (ready for production) alongside the fully cleaned and encoded dataset as a CSV.

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.9+ installed.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/nj-codes-24/CSV-to-Model-Auto-Tuner.git
   cd CSV-to-Model-Auto-Tuner
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the AutoML Engine:
   ```bash
   streamlit run app.py
   ```

### Usage Instructions
1. Open the local Streamlit URL in your browser (usually `http://localhost:8501`).
2. Drop any CSV dataset (e.g., housing prices, customer churn, medical records) into the upload zone.
3. Select your **Target Variable** from the dropdown menu.
4. *(Optional)* Open the "Advanced Parameters" tab to adjust the test split ratio, Cross-Validation folds, top N features, or flag specific leakage columns to be ignored.
5. Click **Launch Engine** and watch the automated pipeline do the heavy lifting!

---

## 📂 Project Structure

```text
CSV-to-Model-Auto-Tuner/
├── app.py                  # The Streamlit UI (Wizard flow, Executive log, Downloads)
├── eda_engine.py           # Core logic: 8-phase automated data cleaning & transformation
├── pipeline.py             # ML engine: Model registries, benchmarking, and CV tuning
├── EDA_master_class.ipynb  # Reference notebook detailing the manual EDA workflow
├── requirements.txt        # Project dependencies
└── .streamlit/
    └── config.toml         # Custom dark-mode glassmorphic theme configuration
```

---

## 🛠️ Technologies Used
- **UI Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, SciPy
- **Model Serialization:** Joblib
