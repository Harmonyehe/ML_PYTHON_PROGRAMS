import requests
import pandas as pd

url = "https://api.covid19api.com/dayone/country/usa"
response = requests.get(url)
data = response.json()

df = pd.DataFrame(data)
print(df.head())  # Preview the data

# Clean the data
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Confirmed', 'Deaths', 'Recovered']]

# Fill missing values (if any)
df = df.fillna(method='ffill')

print(df.head())  # Preview after cleaning

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare data for training
df['Days'] = (df['Date'] - df['Date'].min()).dt.days  # Convert dates to numerical values (days)
X = df[['Days']]  # Features (Days)
y = df['Confirmed']  # Target (Confirmed cases)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future cases
y_pred = model.predict(X_test)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, label='Predicted', linestyle='dashed')
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()

import folium

# Create a map centered at a specific location
map_center = [37.7749, -122.4194]  # Example: San Francisco
map = folium.Map(location=map_center, zoom_start=10)

# Add markers for confirmed cases (this is an example, you'd fetch real coordinates and data)
folium.CircleMarker([37.7749, -122.4194], radius=10, color='blue', fill=True).add_to(map)

# Save the map to an HTML file
map.save("disease_spread_map.html")

# Example prediction for the next 10 days
future_days = pd.DataFrame({'Days': [df['Days'].max() + i for i in range(1, 11)]})
future_predictions = model.predict(future_days)

# Display future predictions
print("Future Predictions (Confirmed cases for next 10 days):", future_predictions)

# Update map with predicted locations (you can add this to your map visualizations)
