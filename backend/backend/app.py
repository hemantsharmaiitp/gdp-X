from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained GDP prediction model
model_path = os.path.join('model', 'gdp_model.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Model or scaler file not found: {e}")
    raise
except Exception as e:
    print(f"An error occurred while loading model/scaler: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        year = data.get('year')

        if year is None:
            return jsonify({'error': 'Missing year in request'}), 400

        # Convert year to numpy array and reshape
        input_data = np.array([[year]])

        # Scale the year input using the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # Predict GDP
        prediction = model.predict(input_scaled)[0]

        return jsonify({'year': year, 'predicted_gdp': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
