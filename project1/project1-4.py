import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset 
data = pd.read_csv('weatherAUS.csv')

data['Date'] = pd.to_datetime(data['Date'])

# Define the date range and the locations of interest
start_date = '2008-01-01'
end_date = '2009-01-01'
locations = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Canberra']

# Filter the data for the specified date range and locations
filtered_data = data[
    (data['Date'] >= start_date) & 
    (data['Date'] <= end_date) & 
    (data['Location'].isin(locations))
].copy()

# Calculate the average daily wind speed from WindSpeed9am and WindSpeed3pm
filtered_data['AverageWindSpeed'] = (filtered_data['WindSpeed9am'] + filtered_data['WindSpeed3pm']) / 2

# Calculate the average temperature from MinTemp and MaxTemp
filtered_data['AverageTemp'] = (filtered_data['MinTemp'] + filtered_data['MaxTemp']) / 2

# Identify the hottest 5% of days
top_5_percent = filtered_data['AverageTemp'].quantile(0.95)

# Filter for the hottest days
hottest_days = filtered_data[filtered_data['AverageTemp'] >= top_5_percent]

# Group by location and prepare the wind speed data
hottest_wind_speeds = hottest_days.groupby('Location')['AverageWindSpeed'].apply(list).reindex(locations, fill_value=[]).tolist()

# Identify the hottest day for each location (date with the highest temperature)
hottest_day_info = (
    hottest_days.loc[hottest_days.groupby('Location')['AverageTemp'].idxmax()][['Location', 'Date', 'AverageTemp']]
)

# Plot the box plot for wind speed on the hottest days
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(hottest_wind_speeds, vert=True, patch_artist=True)

# Set x-axis labels to locations
ax.set_xticks(range(1, len(locations) + 1))
ax.set_xticklabels(locations, rotation=45)

# Add hottest day labels above each box plot
for i, row in enumerate(hottest_day_info.itertuples(), 1):
    ax.text(i, max(hottest_wind_speeds[i - 1]) + 0.5, f"{row.Date.date()}", 
            ha='center', va='bottom', fontsize=10, color='red')

# Set plot titles and labels
ax.set_title('Wind Speed on the Hottest Days')
ax.set_xlabel('Location')
ax.set_ylabel('Average Wind Speed (kmh)')


plt.tight_layout()


fig.savefig('wind_speed_hottest_days.png', dpi=300)

