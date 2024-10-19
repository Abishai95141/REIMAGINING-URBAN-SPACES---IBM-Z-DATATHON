
# Smart Sewage Prediction and Capacity Analysis
# SAV148 : URBAN INNOVATORS

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# Step 1: Load the data
data = pd.read_csv('input.csv')

# Step 2: Feature selection
features = data[['Population_Density', 'Household_Size', 'Water_Consumption', 'Sewage_load_normal_day', 'Sewage_load_eventual_day']]
target = data['Sewage_load_normal_day']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Prediction
data['Predicted_Future_Sewage_Load'] = model.predict(features) * 0.57

# Step 6: Add date column for prediction timestamp
data['Date'] = pd.to_datetime(data['Timestamp']) + timedelta(days=1)

# Step 7: Save updated data with predictions
data.to_csv('updated_sewage_data_with_predictions.csv', index=False)

# Step 8: Load the updated data for individual sewage calculations
data = pd.read_csv('updated_sewage_data_with_predictions.csv')

# Step 9: Calculate individual household sewage generation
data['Individual_Household_Sewage_Generation'] = (data['Predicted_Future_Sewage_Load'] / (data['Household_Size'] * 1000)).round(3)

# Step 10: Set individual household sewage capacity
data['Individual_Household_Sewage_Capacity'] = 1600

# Step 11: Calculate days left until sewage capacity is reached
data['Days_left'] = (data['Individual_Household_Sewage_Capacity'] / data['Individual_Household_Sewage_Generation']).round(3)

# Step 12: Calculate hours left
data['Hours'] = data['Days_left'] * 24

# Step 13: Save the updated dataset with household sewage calculations
data.to_csv('updated_sewage_data_with_household_sewage.csv', index=False)

# Step 14: Print success message
print("Predicted future sewage load has been added and the updated data is saved as 'updated_sewage_data_with_predictions.csv'.")
