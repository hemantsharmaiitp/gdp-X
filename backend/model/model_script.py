import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# === Configuration ===
BASE_DIR = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model'
DATA_PATH = os.path.join(BASE_DIR, 'india_gdp', 'gdp.csv')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'gdp_model.pkl')

# Ensure model directory exists
os.makedirs(BASE_DIR, exist_ok=True)

try:
    # === Load Dataset ===
    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH, skiprows=4)
    print("Dataset loaded successfully!")

    # === Filter for GDP data only ===
    gdp_data = data[data['Indicator Name'] == 'GDP (current US$)']
    print(f"GDP Data filtered. Shape: {gdp_data.shape}")

    # === Clean and Reshape ===
    gdp_data = gdp_data.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], errors='ignore')
    gdp_data = gdp_data.melt(var_name='year', value_name='gdp')
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')
    gdp_data['gdp'] = pd.to_numeric(gdp_data['gdp'], errors='coerce')
    gdp_data = gdp_data.dropna()
    print("Data cleaned and reshaped successfully!")
    print(gdp_data.head())

    # === Feature Scaling ===
    features = gdp_data[['year']]
    target = gdp_data['gdp']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    print("Feature scaling completed.")

    # === Save Scaler ===
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved at: {SCALER_PATH}")

    # === Train-Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
    print(f"Train-Test split done: {len(X_train)} train, {len(X_test)} test")

    # === Model Training ===
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # === Evaluate Model ===
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R^2: {train_score:.2f}")
    print(f"Testing R^2: {test_score:.2f}")

    # === Cross Validation ===
    cv_scores = cross_val_score(model, X_scaled, target, cv=5, scoring='r2')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV R^2: {cv_scores.mean():.2f}")

    # === Save Model ===
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved at: {MODEL_PATH}")

except FileNotFoundError:
    print(f"File not found: {DATA_PATH}")
except KeyError as e:
    print(f"Missing column: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
