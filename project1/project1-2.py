import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('weatherAUS.csv')
# Ensure the 'Time' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

start_date = '2008-01-01'
end_date = '2008-12-31'
locations = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Canberra']
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date) & (data['Location'].isin(locations))]

# Calculate the average temperature as the mean of MinTemp and MaxTemp
filtered_data['average_temp'] = (filtered_data['MinTemp'] + filtered_data['MaxTemp']) / 2
# Group by Location and calculate the mean of average_temp for each location 
average_temp = filtered_data.groupby('Location')['average_temp'].mean().reset_index()

# Plot temperature over time for the filtered data
plt.figure(figsize=(10, 6))
plt.bar(average_temp['Location'], average_temp['average_temp'], color='cornflowerblue')
plt.title('Average Temperature (2008-01-01 to 2008-12-31)') 
plt.xlabel('Location') 
plt.ylabel('Average Temperature (°C)') 
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.tight_layout()
plt.savefig('average_temperature_5_locations.png', dpi=300)



