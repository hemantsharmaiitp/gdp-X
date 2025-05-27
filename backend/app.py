from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model and scaler
model_path = os.path.join('model', 'scaler.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        values = [float(request.form.get(f'f{i+1}')) for i in range(4)]

        # Convert input to numpy array and reshape
        input_data = np.array(values).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Plot the input values as a bar chart
        features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
        plt.figure(figsize=(6,4))
        plt.bar(features, values, color='skyblue')
        plt.title('Input Feature Values')
        plt.ylabel('Value')
        plt.tight_layout()

        # Save the plot to static/images folder
        plot_path = os.path.join('static', 'images', 'input_plot.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return render_template('result.html', prediction=prediction, plot_url=plot_path)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
