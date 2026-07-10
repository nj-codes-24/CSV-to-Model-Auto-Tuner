<div align="center">
  <h1>⚙️ AutoML Engine</h1>
  <p>
    <strong>An end-to-end Machine Learning pipeline that transforms raw, messy CSV datasets into production-ready models in a single click.</strong>
  </p>
  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
    <a href="https://github.com/nj-codes-24/CSV-to-Model-Auto-Tuner/issues"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions Welcome"></a>
  </p>
</div>

<hr>

## 📖 Overview

The **AutoML Engine** (CSV-to-Model Auto-Tuner) bridges the gap between messy real-world data and production deployment. Built for data scientists and developers alike, it fully automates rigorous, industry-standard Exploratory Data Analysis (EDA), statistical feature selection, algorithm benchmarking, and hyperparameter tuning.

By abstracting away the boilerplate of data preparation and model selection, this engine allows you to go from a raw `.csv` file to a deployable `.pkl` model in under a minute, all wrapped in a sleek, glassmorphic UI.

---

## ✨ Features

### 🧪 1. Leakage-Free, Industry-Standard EDA
The engine executes a rigorous data preparation pipeline. All transformations are mathematically persisted and fitted **strictly on the training set** to prevent data leakage.
- **Smart Feature Engineering:** Intelligently shreds text columns, automatically extracting Date/Time metrics (Year, Month, Day, Is_Weekend) and splitting complex structural IDs (like `Cabin: F/34/P`) based on consistent delimiters.
- **Smart Imputation:** Dynamically median-fills skewed numeric data, mean-fills normal distributions, and handles categorical nulls.
- **Statistical Feature Selection:** Eliminates multicollinear features (>0.80 correlation) and utilizes ANOVA (Regression) or Chi-Square (Classification) F-tests to drop statistically insignificant columns.
- **Outlier & Skewness Handling:** Caps outliers using IQR boundaries and applies `log1p` transformations to highly skewed target variables to stabilize variance.
- **Hybrid Encoding:** Routes low-cardinality features (≤10 unique) to One-Hot Encoding and high-cardinality features to Target Encoding.
- **Automated Data Balancing (SMOTE):** Synthetically oversamples the minority class automatically when severe target imbalance (> 10x ratio) is detected, ensuring algorithms can learn rare events.

### 🌲 2. Random Forest Signal Extraction
Before training, the engine runs a Random Forest Feature Importance Scan to rank features. It mathematically detects the "elbow" point of maximum curvature in the cumulative importance graph, slicing the features autonomously to maximize signal and drop noise.

### 📊 3. Automated Benchmarking & Ensembling
The cleaned data is evaluated against a gauntlet of 8 state-of-the-art algorithms (including XGBoost, Gradient Boosting, Random Forest, AdaBoost, Ridge/Lasso, and SVM). 
- **Smart Metrics:** The engine dynamically selects the right evaluation metric (e.g., automatically switching to F1-Weighted if it detects severe class imbalance).
- **Hyperparameter Tuning:** The top 3 winning baseline models are passed through a `RandomizedSearchCV` to find optimal hyperparameters via Cross-Validation.
- **Voting Ensembles:** Combines the top 3 tuned models into a highly robust Voting Regressor/Classifier for maximum performance.

### 💻 4. Resilient & Premium "Glassmorphic" UI
- **Context-Aware Setup:** Guides users through dataset setup by asking simple, plain-English context questions (e.g., identifying ID-like columns for exact deduplication and flagging potential data leakage variables).
- **Big Data Sampler:** Automatically guards against memory crashes by deploying stratified downsampling on datasets larger than 100,000 rows, preserving exact class distributions for scalable learning.
- **Smart Circuit Breakers:** Evaluates baseline algorithmic performance and aborts the pipeline early (skipping tuning) if a dataset lacks any learnable signal or if it achieves perfect accuracy (signaling severe data leakage/overfitting).
- **Executive Log:** Watch the engine make decisions in real-time. The UI prints exactly *why* a column was dropped or *how* a value was imputed.
- **Kaggle Inference Support:** After training, immediately upload an unseen `test.csv` (like in Kaggle competitions). The engine will perfectly re-apply the exact Feature Engineering, Imputation, and Encoding logic to generate perfect predictions with ID columns preserved!
- **Single-Click Deployment:** Instantly download the final tuned model as a serialized `.pkl` file (ready for production integration) alongside the fully cleaned and encoded dataset as a CSV.

---

## 🚀 Installation & Quick Start

### Prerequisites
- Python 3.9 or higher
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nj-codes-24/CSV-to-Model-Auto-Tuner.git
   cd CSV-to-Model-Auto-Tuner
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Engine:**
   ```bash
   streamlit run app.py
   ```

---

## 🎯 Usage Guide

1. **Upload:** Open the local Streamlit URL (`http://localhost:8501`) and drop any raw CSV dataset into the upload zone.
2. **Configure:** Select your **Target Variable** from the dropdown. Then, answer the natural language **Context Questions** to help the engine deduplicate exactly (using ID columns) and prevent data leakage. *(Optional: Open "Advanced Parameters" to adjust the train/test split or CV folds).*
3. **Launch:** Click **⚡ Launch Engine**.
4. **Monitor:** Watch the progressively rendering **Executive Log** as the engine cleans your data, followed by the background model training logs.
5. **Kaggle Inference:** Upload an unseen test dataset (like `test.csv` from Kaggle), select your ID column, and instantly generate an automated prediction file!
6. **Deploy:** Review the final benchmarks and download your clean dataset, test predictions, and trained `.pkl` model directly from the UI.

---

## 🏗️ Architecture

```text
CSV-to-Model-Auto-Tuner/
├── app.py                  # Frontend: Streamlit Wizard UI, Progressive Rendering, State Management
├── eda_engine.py           # Core backend: Automated 8-phase data cleaning & transformation logic
├── pipeline.py             # ML backend: Model registries, CV benchmarking, and parameter grids
├── EDA_master_class.ipynb  # Jupyter Notebook detailing the manual EDA theory behind the engine
├── requirements.txt        # Project dependencies
└── .streamlit/
    └── config.toml         # Custom dark-mode glassmorphic theme styling configuration
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">
  <p>Built with ❤️ by Nishchal Jain</p>
</div>
