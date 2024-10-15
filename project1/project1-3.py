import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('weatherAUS.csv')
# Ensure the 'Time' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

start_date = '2008-01-01'
end_date = '2009-12-31'
locations = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Canberra']
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date) & (data['Location'].isin(locations))]

filtered_data['AverageTemp'] = (filtered_data['MinTemp'] + filtered_data['MaxTemp']) / 2

# Calculate the average daily humidity using Humidity9am and Humidity3pm
filtered_data['AverageHumidity'] = (filtered_data['Humidity9am'] + filtered_data['Humidity3pm']) / 2

# Group the data by Location and calculate the mean of average temperature and humidity
grouped_data = filtered_data.groupby('Location')[['AverageTemp', 'AverageHumidity']].mean().reset_index()

# Calculate the correlation coefficient between temperature and humidity
correlation = grouped_data[['AverageTemp', 'AverageHumidity']].corr().iloc[0, 1]
print(f"Correlation between temperature and humidity: {correlation:.2f}")

# Create a scatter plot of Average Temperature vs. Average Humidity
plt.figure(figsize=(10, 6))
for location in locations:
    loc_data = filtered_data[filtered_data['Location'] == location]
    plt.scatter(loc_data['AverageTemp'], loc_data['AverageHumidity'], label=location, alpha=0.7)

plt.title('Temperature vs. Humidity Across Locations (2008-01-01 to 2008-12-31)')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Average Humidity (%)')
plt.legend(title='Location')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('average_humidity.png', dpi=300)
