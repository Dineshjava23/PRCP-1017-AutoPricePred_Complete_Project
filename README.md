# PRCP-1017: Automobile Import Price Prediction (`AutoPricePred`)

An end-to-end machine learning regression pipeline and interactive web application that predicts automobile import prices based on vehicle specifications, engine attributes, body style, and market indicators.

## Key Results & Highlights
- **High Predictive Accuracy (R² = 0.92):** Tuned **XGBoost** regression model outperformed linear and tree-based baselines while minimizing prediction error across diverse vehicle segments.
- **Complete Preprocessing Pipeline:** Handled missing values, outlier treatment, categorical encoding, and feature scaling inside a reproducible Scikit-learn pipeline.
- **Interactive Cloud Deployment:** Built a user-friendly **Streamlit** web application for real-time price estimation and deployed it on **AWS**.

---

## Tech Stack
- **Language:** Python 3.10+
- **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning & Tuning:** Scikit-learn, XGBoost
- **Web App & Deployment:** Streamlit, AWS (EC2)

---

## Project Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed price distributions across manufacturers, fuel types, aspiration, body styles, engine sizes, horsepower, and highway/city MPG.
2. **Data Cleaning & Feature Engineering:**
   - Imputed missing numerical and categorical entries (`num-of-doors`, `normalized-losses`, etc.).
   - Standardized brand names and engineered domain ratios (e.g., power-to-weight and fuel-efficiency indicators).
   - Applied One-Hot / Ordinal encoding and feature scaling.
3. **Model Benchmarking & Hyperparameter Tuning:**
   - Compared Linear Regression, Ridge/Lasso, Decision Tree, Random Forest, Gradient Boosting, and **XGBoost**.
   - Evaluated using **R² Score**, **RMSE**, and **MAE** with k-fold cross-validation.
4. **Web Application & Cloud Hosting:** Serialized the final pipeline and built a **Streamlit** UI hosted on **AWS** for instant price inference.

---

## Repository Structure
```text
PRCP-1017-AutoPricePred/
├── data/                   # Automobile dataset
├── notebooks/              # EDA, feature engineering, and model comparison
├── models/                 # Saved model and preprocessing pipeline (.pkl / .joblib)
├── app.py                  # Streamlit web application
├── train.py                # End-to-end training and evaluation script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com<your-username>/PRCP-1017-AutoPricePred.git
cd PRCP-1017-AutoPricePred
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App Locally
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to enter vehicle specifications and generate predicted prices.
