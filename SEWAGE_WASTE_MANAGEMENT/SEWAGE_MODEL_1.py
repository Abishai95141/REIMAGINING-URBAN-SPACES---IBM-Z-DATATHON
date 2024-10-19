
# Sewage Synergy : Revolutionizing Sewage Systems with Data-Driven Pumping
# SAV148 : URBAN INNOVATORS

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from geopy.distance import geodesic

# Step 1: Load the data
data = pd.read_csv('/content/input_1.csv')

# Step 2: Feature selection
features = data[['Population_Density', 'Household_Size', 'Water_Consumption', 'Sewage_load_normal_day', 'Sewage_load_eventual_day']]
target = data['Sewage_load_normal_day']

# Step 3: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Prediction
data['Predicted_Future_Sewage_Load'] = model.predict(features) 

# Step 6: Add date column for prediction timestamp (ensure Timestamp column is available)
if 'Timestamp' in data.columns:
    data['Date'] = pd.to_datetime(data['Timestamp']) + timedelta(days=1)
else:
    print("Warning: Timestamp column not found, skipping Date calculation.")

# Step 7: Save updated data with the new columns
data.to_csv('updated_sewage_data_with_predictions.csv', index=False)

data = pd.read_csv('/content/updated_sewage_data_with_predictions.csv')

# Step 8: Calculate household sewage generation and capacity
data['Individual_Household_Sewage_Generation'] = ((data['Predicted_Future_Sewage_Load']) / (data['Household_Size'])).round(3)
data['Individual_Household_Sewage_Capacity'] = ((data['Sewage_load_normal_day']) / (data['Household_Size'])).round(3)
data['Days_left'] = ((data['Individual_Household_Sewage_Capacity']) / (data['Individual_Household_Sewage_Generation'])*24).round(3)

data.to_csv('updated_sewage_data_with_household_sewage.csv', index=False)

print("successfully added")

# Step 9: Plotting Actual vs Predicted Sewage Load
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Sewage_load_normal_day'], label='Actual Sewage Load', color='blue', marker='o', markersize=3, linestyle='-', alpha=0.7)
plt.plot(data['Date'], data['Predicted_Future_Sewage_Load'], label='Predicted Future Sewage Load', color='orange', marker='x', markersize=3, linestyle='--', alpha=0.7)
plt.title('Actual vs Predicted Sewage Load Over Time')
plt.xlabel('Date')
plt.ylabel('Sewage Load (liters/day)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Step 10: Zone-specific Data Generation
zones = data['Zone'].unique()
np.random.seed(42)
latitude = np.random.uniform(low=-90.0, high=90.0, size=len(zones))
longitude = np.random.uniform(low=-180.0, high=180.0, size=len(zones))
elevation = np.random.uniform(low=0, high=2000, size=len(zones))

zone_data = pd.DataFrame({
    'Zone': zones,
    'Latitude': latitude,
    'Longitude': longitude,
    'Elevation': elevation
})

# Step 11: Merge new data into the original dataset
updated_data = pd.merge(data, zone_data, on='Zone', how='left')

# Step 12: Function to calculate slope between two points
def calculate_slope(row1, row2):
    coords_1 = (row1['Latitude'], row1['Longitude'])
    coords_2 = (row2['Latitude'], row2['Longitude'])
    distance = geodesic(coords_1, coords_2).meters
    delta_elevation = abs(row1['Elevation'] - row2['Elevation'])
    slope = delta_elevation / distance if distance != 0 else 0
    return slope

# Step 13: Initialize slope column and calculate slopes
updated_data['Slope'] = np.nan
for i in range(1, len(updated_data)):
    updated_data.loc[i, 'Slope'] = calculate_slope(updated_data.iloc[i], updated_data.iloc[i-1])

# Step 14: Identify low-lying areas
elevation_threshold = 300
updated_data['Low-lying'] = np.where(updated_data['Elevation'] < elevation_threshold, 1, 0)

# Step 15: Save the updated dataset with slope and low-lying information
updated_data.to_csv('updated_sewage_data_with_slope_and_low_lying.csv', index=False)

data = pd.read_csv('updated_sewage_data_with_slope_and_low_lying.csv')

# Step 16: Calculate flow rate and pump capacity
data['Flow_Rate_L_s'] = (data['Predicted_Future_Sewage_Load'] * 1000) / 86400
safety_factor = 1.5
data['Pump_Capacity_L_s'] = data['Flow_Rate_L_s'] * safety_factor
data['Pump_Capacity_m3_h'] = data['Pump_Capacity_L_s'] * 3.6

# Step 17: Save the updated dataset with pump capacity
data.to_csv('updated_sewage_data_with_pump_capacity.csv', index=False)

# Step 18: Constants for power calculation
rho = 1000
g = 9.81
pump_efficiency = 0.8

pipe_length_m = 1000
pipe_diameter_m = 0.5
friction_loss_factor = 0.02

data['Static_Head_m'] = data['Elevation']

# Step 19: Hazen-Williams equation for calculating friction loss
def calculate_friction_loss(flow_rate_m3_s, pipe_length_m, pipe_diameter_m, c_factor=140):
    a = (10.67 * (flow_rate_m3_s*1.85) * pipe_length_m)
    b = (c_factor*1.85 * (pipe_diameter_m*4.87))
    return a / b

# Step 20: Calculate flow rate and friction head loss
data['Flow_Rate_m3_s'] = data['Pump_Capacity_L_s'] / 1000
data['Friction_Head_Loss_m'] = data.apply(lambda row: calculate_friction_loss(
    row['Flow_Rate_m3_s'], pipe_length_m, pipe_diameter_m), axis=1)

# Step 21: Calculate total head and power required for pumping
data['Total_Head_m'] = data['Static_Head_m'] + data['Friction_Head_Loss_m']
data['Power_Watts'] = (data['Flow_Rate_m3_s'] * data['Total_Head_m'] * rho * g) / pump_efficiency

# Step 22: Save the updated dataset with power calculations
data.to_csv('updated_sewage_data_with_power_calculations.csv', index=False)

print("Realistic total head and pump power calculations complete.")

# Step 23: Calculate average and threshold sewage loads
average_sewage_load = data['Sewage_load_normal_day'].mean()
peak_sewage_load = data['Sewage_load_normal_day'].max()
threshold_factor_sewage = 0.8
threshold_sewage_load = average_sewage_load * threshold_factor_sewage

# Step 24: Update the dataset with sewage load threshold
data['Threshold_Sewage_Load'] = threshold_sewage_load

# Step 25: Save the updated dataset with the threshold column
data.to_csv('updated_sewage_data_with_slope_and_low_lying.csv', index=False)

data = pd.read_csv('/content/FINALISED.csv')

# Step 26: Calculate priority index based on predicted load and low-lying areas
def calculate_priority(row):
    threshold = row['Threshold_Sewage_Load']
    predicted = row['Predicted_Future_Sewage_Load']
    low_lying = row['Low-lying']
    if predicted >= threshold * 0.95 and low_lying == 1:
        return 1
    elif predicted >= threshold * 0.95 and low_lying == 0:
        return 2
    elif predicted >= threshold * 0.60 and low_lying == 0:
        return 3
    return 0

data['Priority_Index'] = data.apply(calculate_priority, axis=1)

# Step 27: Save the updated dataset with the priority index
data.to_csv('Finalised_initial_input_for_first_model.csv', index=False)

# Step 28: Load the final dataset with all results
data = pd.read_csv('Finalised_initial_input_for_first_model.csv')

# Step 29: Plotting Actual vs Predicted Sewage Load
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Sewage_Load_Normal_Day'], label='Actual Sewage Load', color='blue', marker='o', markersize=3, linestyle='-', alpha=0.7)
plt.plot(data['Date'], data['Predicted_Future_Sewage_Load'], label='Predicted Future Sewage Load', color='orange', marker='x', markersize=3, linestyle='--', alpha=0.7)
plt.title('Actual vs Predicted Sewage Load Over Time')
plt.xlabel('Date')
plt.ylabel('Sewage Load (liters/day)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Step 30: Plotting Slope between Zones
plt.figure(figsize=(12, 6))
plt.bar(data['Zone'], data['Slope'], color='purple')
plt.title('Slope Between Zones')
plt.xlabel('Zone')
plt.ylabel('Slope (Elevation Change / Distance)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Step 31: Plotting Low-Lying Areas
plt.figure(figsize=(12, 6))
plt.bar(data['Zone'], data['Low-lying'], color='green')
plt.title('Low-Lying Areas (1 = Yes, 0 = No)')
plt.xlabel('Zone')
plt.ylabel('Low-Lying')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Step 32: Plotting Flow Rate
plt.figure(figsize=(12, 6))
plt.plot(data['Zone'], data['Flow_Rate_L_s'], label='Flow Rate (L/s)', color='teal', marker='o')
plt.title('Flow Rate Based on Predicted Sewage Load')
plt.xlabel('Zone')
plt.ylabel('Flow Rate (L/s)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Step 33: Plotting Pump Capacity
plt.figure(figsize=(12, 6))
plt.plot(data['Zone'], data['Pump_Capacity_L_s'], label='Pump Capacity (L/s)', color='orange', marker='x')
plt.title('Pump Capacity Required Based on Flow Rate')
plt.xlabel('Zone')
plt.ylabel('Pump Capacity (L/s)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Step 34: Plotting Power Calculations
plt.figure(figsize=(12, 6))
plt.plot(data['Zone'], data['Power_W'], label='Power Required (Watts)', color='red', marker='s')
plt.title('Power Required for Pumping Sewage')
plt.xlabel('Zone')
plt.ylabel('Power (Watts)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Step 35: Plotting Priority Index
plt.figure(figsize=(12, 6))
plt.bar(data['Zone'], data['Priority_Index'], color='gold')
plt.title('Priority Index for Sewage Management')
plt.xlabel('Zone')
plt.ylabel('Priority Index')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

print("All operations completed successfully.")
