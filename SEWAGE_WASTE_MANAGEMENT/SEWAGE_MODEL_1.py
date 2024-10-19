
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

# Step 1: Load the dataset
data = pd.read_csv('/content/input_1.csv')

# Step 2: Feature selection
features = data[['Population_Density', 'Household_Size', 'Water_Consumption', 'Sewage_load_normal_day', 'Sewage_load_eventual_day']]
target = data['Sewage_load_normal_day']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict sewage load
data['Predicted_Future_Sewage_Load'] = model.predict(features) * 0.57 * 2

# Step 6: Add date column if 'Timestamp' exists
if 'Timestamp' in data.columns:
    data['Date'] = pd.to_datetime(data['Timestamp']) + timedelta(days=1)
else:
    print("Warning: Timestamp column not found, skipping Date calculation.")

# Step 7: Save the data with predictions
data.to_csv('updated_sewage_data_with_predictions.csv', index=False)

# Step 8: Reload the updated data
data = pd.read_csv('/content/updated_sewage_data_with_predictions.csv')

# Step 9: Calculate household sewage generation and capacity
data['Individual_Household_Sewage_Generation'] = ((data['Predicted_Future_Sewage_Load']) / (data['Household_Size'])).round(3)
data['Individual_Household_Sewage_Capacity'] = ((data['Sewage_load_normal_day']) / (data['Household_Size'])).round(3)
data['Days_left'] = ((data['Individual_Household_Sewage_Capacity']) / (data['Individual_Household_Sewage_Generation'])*24).round(3)

# Step 10: Save the updated data with household sewage calculations
data.to_csv('updated_sewage_data_with_household_sewage.csv', index=False)

print("successfully added")

# Step 11: Plot actual vs predicted sewage load over time
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

# Step 12: Generate random zone data (latitude, longitude, elevation)
zones = data['Zone'].unique()
np.random.seed(42)
latitude = np.random.uniform(low=-90.0, high=90.0, size=len(zones))
longitude = np.random.uniform(low=-180.0, high=180.0, size=len(zones))
elevation = np.random.uniform(low=0, high=2000, size=len(zones))

# Step 13: Create a new DataFrame for zone-specific data
zone_data = pd.DataFrame({
    'Zone': zones,
    'Latitude': latitude,
    'Longitude': longitude,
    'Elevation': elevation
})

# Step 14: Merge zone-specific data into the original dataset
updated_data = pd.merge(data, zone_data, on='Zone', how='left')

# Step 15: Function to calculate slope between two points
def calculate_slope(row1, row2):
    coords_1 = (row1['Latitude'], row1['Longitude'])
    coords_2 = (row2['Latitude'], row2['Longitude'])
    distance = geodesic(coords_1, coords_2).meters
    delta_elevation = abs(row1['Elevation'] - row2['Elevation'])
    slope = delta_elevation / distance if distance != 0 else 0
    return slope

# Step 16: Initialize slope column and calculate slope for each zone
updated_data['Slope'] = np.nan
for i in range(1, len(updated_data)):
    updated_data.loc[i, 'Slope'] = calculate_slope(updated_data.iloc[i], updated_data.iloc[i-1])

# Step 17: Flag low-lying areas based on elevation threshold
elevation_threshold = 300
updated_data['Low-lying'] = np.where(updated_data['Elevation'] < elevation_threshold, 1, 0)

# Step 18: Save the updated dataset with slope and low-lying areas
updated_data.to_csv('updated_sewage_data_with_slope_and_low_lying.csv', index=False)

# Step 19: Reload updated data for further calculations
data = pd.read_csv('updated_sewage_data_with_slope_and_low_lying.csv')

# Step 20: Calculate flow rate and pump capacity based on sewage load
data['Flow_Rate_L_s'] = (data['Predicted_Future_Sewage_Load'] * 1000) / 86400
safety_factor = 1.5
data['Pump_Capacity_L_s'] = data['Flow_Rate_L_s'] * safety_factor
data['Pump_Capacity_m3_h'] = data['Pump_Capacity_L_s'] * 3.6

# Step 21: Save the updated dataset with pump capacity
data.to_csv('updated_sewage_data_with_pump_capacity.csv', index=False)

# Step 22: Set constants for pump power calculations
rho = 1000
g = 9.81
pump_efficiency = 0.8

# Step 23: Pipe details for friction loss calculations
pipe_length_m = 1000
pipe_diameter_m = 0.5
friction_loss_factor = 0.02

# Step 24: Calculate static head based on elevation
data['Static_Head_m'] = data['Elevation']

# Step 25: Function to calculate friction head loss
def calculate_friction_loss(flow_rate_m3_s, pipe_length_m, pipe_diameter_m, c_factor=140):
    a = (10.67 * (flow_rate_m3_s * 1.85) * pipe_length_m)
    b = (c_factor * 1.85 * (pipe_diameter_m * 4.87))
    return a / b

# Step 26: Calculate flow rate in cubic meters per second
data['Flow_Rate_m3_s'] = data['Pump_Capacity_L_s'] / 1000

# Step 27: Apply friction loss calculation to each row
data['Friction_Head_Loss_m'] = data.apply(lambda row: calculate_friction_loss(row['Flow_Rate_m3_s'], pipe_length_m, pipe_diameter_m), axis=1)

# Step 28: Calculate total head as the sum of static head and friction losses
data['Total_Head_m'] = data['Static_Head_m'] + data['Friction_Head_Loss_m']

# Step 29: Calculate power required for pumping
data['Power_Watts'] = (data['Flow_Rate_m3_s'] * data['Total_Head_m'] * rho * g) / pump_efficiency

# Step 30: Save the updated dataset with power calculations
data.to_csv('updated_sewage_data_with_power_calculations.csv', index=False)

# Step 31: Calculate average and threshold sewage loads
average_sewage_load = data['Sewage_load_normal_day'].mean()
peak_sewage_load = data['Sewage_load_normal_day'].max()
threshold_factor_sewage = 0.8
threshold_sewage_load = average_sewage_load * threshold_factor_sewage

# Step 32: Add threshold sewage load to the dataset
data['Threshold_Sewage_Load'] = threshold_sewage_load

# Step 33: Save the updated dataset with threshold sewage load
data.to_csv('updated_sewage_data_with_slope_and_low_lying.csv', index=False)

# Step 34: Reload the dataset for priority index calculations
data = pd.read_csv('updated_sewage_data_with_slope_and_low_lying.csv')

# Step 35: Function to calculate priority index based on conditions
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

# Step 36: Apply priority index calculation to the dataset
data['Priority_Index'] = data.apply(calculate_priority, axis=1)

# Step 37: Save the updated dataset with the priority index
data.to_csv('Finalised_initial_input_for_first_model.csv', index=False)

# Step 38: Reload the final dataset for visualization
data = pd.read_csv('Finalised_initial_input_for_first_model.csv')

# Step 39: Plot slope between zones
plt.figure(figsize=(12, 6))
plt.bar(data['Zone'], data['Slope'], color='purple')
plt.title('Slope Between Zones')
plt.xlabel('Zone')
plt.ylabel('Slope (Elevation Change / Distance)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Step 40: Plot low-lying areas
plt.figure(figsize=(12, 6))
plt.bar(data['Zone'], data['Low-lying'], color='green')
plt.title('Low-Lying Areas (1 = Yes, 0 = No)')
plt.xlabel('Zone')
plt.ylabel('Low-Lying')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Step 41: Plot flow rate
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

# Step 42: Plot pump capacity
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

# Step 43: Plot power calculations
plt.figure(figsize=(12, 6))
plt.plot(data['Zone'], data['Power_Watts'], label='Power Required (Watts)', color='red', marker='s')
plt.title('Power Required for Pumping Sewage')
plt.xlabel('Zone')
plt.ylabel('Power (Watts)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Step 44: Plot priority index
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

# Step 45: Reload final dataset for report generation
data = pd.read_csv('Finalised_initial_input_for_first_model.csv')

# Step 46: Function to describe priority levels
def priority_description(priority_index):
    if priority_index == 1:
        return "Most Important (Critical State)"
    elif priority_index == 2:
        return "Important (Needs Attention)"
    elif priority_index == 3:
        return "Moderate (Less Urgent)"
    else:
        return "Low Priority (No Immediate Action Required)"

# Step 47: Function to generate a sewage management report
def generate_sewage_report(timestamp, zone, actual_load, predicted_load, threshold_load, slope, low_lying, flow_rate, pump_capacity, power_required, priority_index):
    priority_text = priority_description(priority_index)
    report = f"""
    Sewage Management Report:
    - Date/Time: {timestamp}
    - Zone: {zone}
    
    Sewage Load Information:
    - Actual Sewage Load: {actual_load:.2f} liters/day
    - Predicted Sewage Load: {predicted_load:.2f} liters/day
    - Threshold Sewage Load: {threshold_load:.2f} liters/day

    Zone Characteristics:
    - Slope between zones: {slope:.2f} (Elevation change/distance)
    - Low-Lying Area: {'Yes' if low_lying == 1 else 'No'}

    Flow and Capacity Metrics:
    - Flow Rate: {flow_rate:.2f} liters/second
    - Pump Capacity: {pump_capacity:.2f} liters/second
    - Power Required for Pumping: {power_required:.2f} Watts

    Priority Assessment:
    - Priority Index: {priority_text}

    Summary:
    - Zone {zone} is currently in a {priority_text.lower()}. 
    {'Immediate attention is required due to low-lying area and high sewage levels.' if low_lying == 1 and predicted_load >= threshold_load else 'No immediate action required.'}
    """
    return report

# Step 48: Select top 10 entries for report generation
data_top_10 = data.head(10)

# Step 49: Display "Most Important" priority zones first
print("\n" + "="*40 + "\n" + "MOST IMPORTANT (CRITICAL STATE) ZONES\n" + "="*40)
for index, row in data_top_10[data_top_10['Priority_Index'] == 1].iterrows():
    report = generate_sewage_report(
        timestamp=row['Timestamp'],
        zone=row['Zone'],
        actual_load=row['Sewage_load_normal_day'],
        predicted_load=row['Predicted_Future_Sewage_Load'],
        threshold_load=row['Threshold_Sewage_Load'],
        slope=row['Slope'],
        low_lying=row['Low-lying'],
        flow_rate=row['Flow_Rate_L_s'],
        pump_capacity=row['Pump_Capacity_L_s'],
        power_required=row['Power_Watts'],
        priority_index=row['Priority_Index']
    )
    print(report)
    print("\n" + "="*80 + "\n")

# Step 50: Display other priority zones
print("\n" + "="*40 + "\n" + "OTHER PRIORITY ZONES\n" + "="*40)
for index, row in data_top_10[data_top_10['Priority_Index'] != 1].iterrows():
    report = generate_sewage_report(
        timestamp=row['Timestamp'],
        zone=row['Zone'],
        actual_load=row['Sewage_load_normal_day'],
        predicted_load=row['Predicted_Future_Sewage_Load'],
        threshold_load=row['Threshold_Sewage_Load'],
        slope=row['Slope'],
        low_lying=row['Low-lying'],
        flow_rate=row['Flow_Rate_L_s'],
        pump_capacity=row['Pump_Capacity_L_s'],
        power_required=row['Power_Watts'],
        priority_index=row['Priority_Index']
    )
    print(report)
    print("\n" + "="*80 + "\n")
