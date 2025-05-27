from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the scaler and model
scaler_path = os.path.join('model', 'scaler.pkl')
model_path = os.path.join('model', 'model.pkl')  # Make sure this exists

# Load the scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the trained regression model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input values
        values = [float(request.form.get(f'f{i+1}')) for i in range(4)]
        input_data = np.array(values).reshape(1, -1)

        # Scale the input values
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Plot the input features
        features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
        plt.figure(figsize=(6, 4))
        plt.bar(features, values, color='skyblue')
        plt.title('Input Feature Values')
        plt.ylabel('Value')
        plt.tight_layout()

        # Save plot image
        plot_path = os.path.join('static', 'images', 'input_plot.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return render_template('result.html', prediction=round(prediction, 2), plot_url=plot_path)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
