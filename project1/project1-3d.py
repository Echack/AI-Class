import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('weatherAUS.csv')
# Ensure the 'Time' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

start_date = '2008-01-01'
end_date = '2008-12-31'
location = 'Brisbane'
# Filter the data for the specified date range and location
filtered_data = data[
    (data['Date'] >= start_date) &
    (data['Date'] <= end_date) &
    (data['Location'] == location)
]

# Check if the location exists in the dataset
if filtered_data.empty:
    print(f"No data found for {location} in the given date range.")
else:
    # Calculate the daily average temperature and humidity
    filtered_data['AverageTemp'] = (filtered_data['MinTemp'] + filtered_data['MaxTemp']) / 2
    filtered_data['AverageHumidity'] = (filtered_data['Humidity9am'] + filtered_data['Humidity3pm']) / 2

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_data['AverageTemp'], filtered_data['AverageHumidity'], alpha=0.7, color='skyblue')
    plt.title(f'Temperature vs. Humidity in Brisbane (2008-01-01 to 2008-12-31)')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('Average Humidity (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('individual_location_plots_Brisbane.png', dpi=300)