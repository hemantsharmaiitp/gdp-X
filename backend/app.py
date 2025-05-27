from flask import Flask, render_template, request, url_for
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Flask app
app = Flask(__name__)

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model', 'gdp_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Function to generate plot and save it
def generate_gdp_plot(year, prediction):
    try:
        # Years for plotting (from 1960 to 2035)
        years = np.arange(1960, 2036).reshape(-1, 1)
        years_scaled = scaler.transform(years)
        predictions = model.predict(years_scaled)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(years, predictions, label='Predicted GDP (in US$)', color='green', linewidth=2)
        plt.axvline(x=year, color='blue', linestyle='--', label=f'Prediction for {year}')
        plt.scatter([year], [prediction], color='red', label='Predicted Point', zorder=5)
        plt.title("India GDP Prediction")
        plt.xlabel("Year")
        plt.ylabel("GDP (Current US$)")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_dir = os.path.join('static', 'images')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'plot.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = request.form.get('year')
        if not year or not year.isdigit():
            return render_template('result.html', prediction_text="Invalid input. Please enter a valid year.")

        year = int(year)

        # Apply the same scaling
        scaled_year = scaler.transform([[year]])
        prediction = model.predict(scaled_year)[0]

        # Generate GDP plot
        plot_path = generate_gdp_plot(year, prediction)

        return render_template(
            'result.html',
            prediction_text=f"Predicted GDP for {year}: ${prediction:,.2f}",
            plot_path=url_for('static', filename='images/plot.png')
        )
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
