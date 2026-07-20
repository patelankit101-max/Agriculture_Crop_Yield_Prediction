import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor

st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

st.title("🌾 Agricultural Crop Yield Prediction")

st.write("Predict crop yield using Machine Learning")

# ------------------------
# Data Loading and Preprocessing (required for plots)
# ------------------------

@st.cache_data # Cache data loading for performance
def load_and_preprocess_data():
    # df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Sample Data/crop_production.csv")
    df = pd.read_csv("crop_production.csv")
    df.drop_duplicates(inplace=True)
    df = df[df['Area'] > 0]
    df['Yield'] = df['Production'] / df['Area']
    df.dropna(subset=['Yield'], inplace=True)
    return df

df = load_and_preprocess_data()

# Load encoders
state_encoder = joblib.load("state_encoder.pkl")
district_encoder = joblib.load("district_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

# Create encoded columns (original columns remain unchanged)
df['State_Name_Encoded'] = state_encoder.transform(df['State_Name'])
df['District_Name_Encoded'] = district_encoder.transform(df['District_Name'])
df['Season_Encoded'] = season_encoder.transform(df['Season'])
df['Crop_Encoded'] = crop_encoder.transform(df['Crop'])

# Features for training
X = df[['State_Name_Encoded', 'District_Name_Encoded', 'Crop_Year', 'Season_Encoded', 'Crop_Encoded', 'Area']]
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Load Models and Calculate Metrics
# ------------------------

@st.cache_resource # Cache models to avoid reloading on each rerun
def load_models():
    lr_model = joblib.load("linear_regression_model.pkl")
    dt_model = joblib.load("decision_tree_model.pkl")
    rf_model = joblib.load("crop_yield_model.pkl") # Tuned Random Forest (best_model)
    gb_model = joblib.load("gradient_boosting_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl") # Load XGBoost model
    adaboost_model = joblib.load("adaboost_model.pkl") # Load AdaBoost model

    return {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model, # Add XGBoost
        "AdaBoost": adaboost_model # Add AdaBoost
    }

loaded_models = load_models()

# Evaluate all models to get their metrics for the comparison chart and display
results = []
for name, model in loaded_models.items():
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results.append([name, mae, rmse, r2])

model_performance_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
model_performance_df.set_index("Model", inplace=True)

# Define performance metrics for each model based on current evaluation
model_performance = model_performance_df.to_dict(orient='index')


# ------------------------
# Sidebar
# ------------------------

st.sidebar.header("Enter Crop Details")

# Model Selection Dropdown
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(loaded_models.keys())
)

# Set the current model based on selection
model = loaded_models[selected_model_name]

# Using original classes for selectbox to display meaningful names
selected_state = st.sidebar.selectbox(
    "State",
    state_encoder.classes_
)

selected_district = st.sidebar.selectbox(
    "District",
    district_encoder.classes_
)

selected_crop = st.sidebar.selectbox(
    "Crop",
    crop_encoder.classes_
)

slected_season = st.sidebar.selectbox(
    "Season",
    season_encoder.classes_
)

year = st.sidebar.number_input(
    "Crop Year",
    min_value=1997,
    max_value=2035,
    value=2023
)

area = st.sidebar.number_input(
    "Cultivated Area (Hectare)",
    min_value=0.1,
    value=1.0
)

# Encode selected values for model prediction
state_encoded = state_encoder.transform([selected_state])[0]
district_encoded = district_encoder.transform([selected_district])[0]
crop_encoded = crop_encoder.transform([selected_crop])[0]
season_encoded = season_encoder.transform([slected_season])[0]

input_data = pd.DataFrame({
    "State_Name_Encoded":[state_encoded],
    "District_Name_Encoded":[district_encoded],
    "Crop_Year":[year],
    "Season_Encoded":[season_encoded],
    "Crop_Encoded":[crop_encoded],
    "Area":[area]
})

if st.sidebar.button("Predict Yield"): # Changed to sidebar button
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Crop Yield : {prediction:.2f}")

# ------------------------
# Main Content - Model Performance and Plots
# ------------------------

st.subheader(f"Model Performance ({selected_model_name})")
current_metrics = model_performance[selected_model_name]
st.metric("R² Score", f"{current_metrics['R2']:.2f}")
st.metric("RMSE", f"{current_metrics['RMSE']:.2f}")
st.metric("MAE", f"{current_metrics['MAE']:.2f}")

st.markdown("### Model Comparison by R2 Score")
fig_r2_comparison = plt.figure(figsize=(10, 6))
sns.barplot(x=model_performance_df.index, y='R2', data=model_performance_df.sort_values(by='R2', ascending=False), palette='viridis') # Use consistent palette
plt.title('Model Comparison by R2 Score')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid as requested
plt.tight_layout()
st.pyplot(fig_r2_comparison)

st.markdown("### Comprehensive Model Comparison (MAE, RMSE, R2)")
results_melted = model_performance_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')

fig_all_metrics_comparison = plt.figure(figsize=(14, 7))
sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted, palette='muted') # Use a different palette for this plot
plt.title('Comprehensive Model Comparison (MAE, RMSE, R2)')
plt.xlabel('Model')
plt.ylabel('Score Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid
plt.tight_layout()
st.pyplot(fig_all_metrics_comparison)

st.markdown(f"### Actual vs Predicted Yield ({selected_model_name})")
fig_actual_predicted = plt.figure(figsize=(8, 6))
pred_selected = model.predict(X_test)
sns.scatterplot(x=y_test, y=pred_selected, alpha=0.6, color='skyblue') # Consistent color and alpha
# Add a perfect prediction line (y=x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title(f"Actual vs Predicted Yield for {selected_model_name}")
plt.grid(True, linestyle='--', alpha=0.7) # Add grid
st.pyplot(fig_actual_predicted)

# Feature Importance for tree-based models including XGBoost and AdaBoost
if selected_model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "AdaBoost"]:
    st.markdown(f"### Feature Importance ({selected_model_name})")
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        })
        importance = importance.sort_values('Importance', ascending=False)

        fig_feature_importance = plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance, palette='magma') # Use 'magma' palette
        plt.title(f'Feature Importance for {selected_model_name}')
        plt.grid(axis='x', linestyle='--', alpha=0.7) # Add grid
        plt.tight_layout()
        st.pyplot(fig_feature_importance)
    else:
        st.write("Feature importance is not directly available for this model type.")
