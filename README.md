

# 🌾 Agricultural Crop Yield Prediction

Predicting crop yield (production per hectare) across Indian states and districts using machine learning, trained on two decades of government crop production records.

## 🚀 Live Dashboard:  👉 [**Open the Live App**](https://agriculture--crop-yield-prediction.streamlit.app/)
---

## 📌 Overview

This project takes raw crop production data (state, district, year, season, crop, and cultivated area) and:

1. Cleans and engineers a `Yield` feature (`Production / Area`)
2. Trains and compares **6 regression models**
3. Serves live predictions and model performance charts through an interactive Streamlit dashboard

Enter a state, district, crop, season, year, and cultivated area — the app predicts expected yield and shows how each model performs.

## 📊 Model Performance

All models evaluated on the same held-out 20% test split:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost** ⭐ | 19.33 | 279.89 | **0.855** |
| Random Forest | 11.11 | 308.12 | 0.825 |
| Random Forest (tuned) | 13.48 | 330.16 | 0.799 |
| Decision Tree | 13.77 | 452.21 | 0.622 |
| Gradient Boosting | 52.91 | 536.86 | 0.467 |
| AdaBoost | 90.81 | 638.12 | 0.247 |
| Linear Regression | 93.61 | 732.44 | 0.008 |

XGBoost gives the strongest fit, explaining ~85.5% of the variance in crop yield.

## 🗂️ Dataset

- **Source:** Historical crop production records, 2000–2015
- **Rows:** 246,091 (post-cleaning)
- **Features used:** `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`, `Area`
- **Target:** `Yield` (derived as `Production / Area`)

Categorical columns (`State_Name`, `District_Name`, `Season`, `Crop`) are label-encoded before training; the fitted encoders are saved alongside the models so the app can decode user input consistently.

## 🧠 Models Trained

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor (base + hyperparameter-tuned via `RandomizedSearchCV`)
- Gradient Boosting Regressor
- XGBoost Regressor
- AdaBoost Regressor

## 🖥️ Dashboard Features

- Model selector — switch between all 6 trained models
- Live yield prediction from user-entered state, district, crop, season, year, and area
- R² / RMSE / MAE for the selected model
- Model comparison charts (R² ranking, and combined MAE/RMSE/R² view)
- Actual vs. predicted scatter plot for the selected model
- Feature importance chart for tree-based models

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** scikit-learn, XGBoost
- **App/Dashboard:** Streamlit
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Community Cloud

## 📁 Project Structure

```
├── app.py                              # Streamlit dashboard
├── crop_production.csv                 # Source dataset
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # (legacy — Python version now set via Streamlit Cloud Advanced settings)
├── *_model.pkl                         # Trained model files
├── *_encoder.pkl                       # Fitted label encoders
├── Agricultural_Crop_Yield_Prediction_Business_pr...  # Business problem notebook
├── Project_Steps.pdf                   # Project write-up
└── all_plots.pdf                       # Exported EDA & evaluation charts
```

## ▶️ Run Locally

```bash
git clone https://github.com/patelankit101-max/Agriculture_Crop_Yield_Prediction.git
cd Agriculture_Crop_Yield_Prediction
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## ☁️ Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io), which builds and redeploys automatically from this repository on every push to `main`.

## 👤 Author

**Ankit Patel**
[GitHub](https://github.com/patelankit101-max)
