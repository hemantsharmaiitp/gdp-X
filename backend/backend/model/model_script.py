import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# File path for the dataset
file_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/india_gdp/gdp.csv'

try:
    # Load the dataset (skip first 4 rows as they contain metadata)
    data = pd.read_csv(file_path, skiprows=4)
    print("Dataset loaded successfully!")

    # Filter for India's GDP (current US$)
    gdp_data = data[
        (data['Country Name'] == 'India') &
        (data['Indicator Name'] == 'GDP (current US$)')
    ]
    print(f"GDP Data filtered. Shape: {gdp_data.shape}")

    # Drop unnecessary columns
    gdp_data = gdp_data.drop(columns=[
        'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'
    ], errors='ignore')

    # Reshape data to year-GDP format
    gdp_data = gdp_data.melt(var_name='year', value_name='gdp')
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')
    gdp_data['gdp'] = pd.to_numeric(gdp_data['gdp'], errors='coerce')
    gdp_data = gdp_data.dropna()
    print("Data cleaned and reshaped successfully!")
    print(gdp_data.head())

    # Prepare features and target
    X = gdp_data[['year']]
    y = gdp_data['gdp']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature scaling completed.")

    # Save scaler
    scaler_save_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/scaler.pkl'
    with open(scaler_save_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"Scaler saved successfully at: {scaler_save_path}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Train-Test Split completed: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print(f"Cross-validation R² Scores: {cv_scores}")
    print(f"Mean CV R² Score: {cv_scores.mean():.4f}")

    # Save model
    model_save_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/gdp_model.pkl'
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved successfully at: {model_save_path}")

except FileNotFoundError:
    print(f"File not found! Ensure the file exists at: {file_path}")
except KeyError as e:
    print(f"KeyError: Missing column in dataset: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
