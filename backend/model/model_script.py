import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# File path for the GDP CSV dataset
file_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/india_gdp/gdp.csv'

try:
    # Load the dataset and skip metadata rows
    data = pd.read_csv(file_path, skiprows=4)
    print("✅ Dataset loaded successfully!")

    # Filter data only for GDP (current US$) indicator
    gdp_data = data[data['Indicator Name'] == 'GDP (current US$)']
    print(f"✅ GDP data filtered. Shape: {gdp_data.shape}")

    # Remove irrelevant columns and reshape the data
    gdp_data = gdp_data.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], errors='ignore')
    gdp_data = gdp_data.melt(var_name='year', value_name='gdp')
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')
    gdp_data['gdp'] = pd.to_numeric(gdp_data['gdp'], errors='coerce')
    gdp_data = gdp_data.dropna()
    print("✅ Data cleaned and reshaped successfully!")
    print(gdp_data.head())

    # Prepare features and target
    X = gdp_data[['year']]
    y = gdp_data['gdp']

    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Feature scaling completed.")

    # Save the scaler for use in prediction
    scaler_save_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/scaler.pkl'
    with open(scaler_save_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"✅ Scaler saved at: {scaler_save_path}")

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_scaled, y)
    print("✅ Linear Regression model trained.")

    # Evaluate the model
    r2_score = model.score(X_scaled, y)
    print(f"📊 Model R^2 score: {r2_score:.4f}")

    # Save the trained model
    model_save_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/gdp_model.pkl'
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"✅ Model saved at: {model_save_path}")

except FileNotFoundError:
    print(f"❌ File not found! Check the path: {file_path}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
