import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# Use non-interactive backend to save figures without display
matplotlib.use('Agg')

def generate_gdp_plot(year, predicted_gdp):
    file_path = 'model/india_gdp/gdp.csv'
    
    # Check if the GDP data file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    # Load and process the GDP data
    data = pd.read_csv(file_path, skiprows=4)
    gdp_data = data[data['Indicator Name'] == 'GDP (current US$)']
    gdp_data = gdp_data.drop(columns=[
        'Country Name', 'Country Code', 'Indicator Name', 
        'Indicator Code', 'Unnamed: 68'
    ])
    gdp_data = gdp_data.melt(var_name='year', value_name='gdp').dropna()
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')

    # Plot actual GDP and the predicted point
    plt.figure(figsize=(10, 6))
    plt.scatter(gdp_data['year'], gdp_data['gdp'], color='green', label='Actual GDP')
    plt.scatter([year], [predicted_gdp], color='red', s=100, label=f'Predicted GDP for {year}')
    
    plt.xlabel('Year')
    plt.ylabel('GDP (Current US$)')
    plt.title(f'GDP Prediction for {year}')
    plt.legend()
    plt.grid(True)

    # Ensure the red dot is visible even if it's a future year
    plt.xlim(gdp_data['year'].min(), max(gdp_data['year'].max(), year + 1))

    # Save the plot to the static/images folder
    os.makedirs('static/images', exist_ok=True)
    plot_path = os.path.join('static', 'images', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path
