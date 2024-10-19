
<h3> TrafficTrack : Dynamic Traffic Forecasting and Intelligent Rerouting </h3>
<h4> SAV 148 : URBAN INNOVATORS </h4>

<hr>

<h3> PSEUDO CODE / ALGORITHM : </h3>

#### 1. Import necessary libraries
    - numpy, pandas, sklearn, tensorflow, matplotlib, seaborn, time

#### 2. Load and preprocess the dataset
    - Load CSV file containing traffic data with latitude and longitude columns.
    - Convert 'timestamp' to datetime format.
    - Set 'timestamp' as the DataFrame index.

#### 3. Handle missing data
    - Use forward-fill method to handle missing values.

#### 4. Calculate Altitude - Saved inside Dataset
    - Use the formula to calculate altitude:
      altitude = 300 + (50 * sin(latitude in radians)) + (10 * cos(longitude in radians))

#### 5. Calculate Road Width - Saved inside Dataset
    - Calculate road width using the formula:
      road_width = 1000 / (altitude + 1) + random noise between -5 and 5.
    - Clip the road width to a range between 5 and 50 meters.

#### 6. Define feature sets
    - Select features including geographic, traffic, and the newly calculated road width and altitude features.
    - Separate categorical and numerical features.

#### 7. Encode categorical features
    - Use OneHotEncoder to encode categorical features.

#### 8. Scale numerical features
    - Scale numerical features using MinMaxScaler.

#### 9. Combine scaled numerical and encoded categorical features
    - Concatenate both scaled numerical and encoded categorical features.

#### 10. Define function to calculate maximum vehicle capacity based on road width
    - Assume each meter of road width can accommodate 2 vehicles.

#### 11. Create sequences of data including road width for calculating capacity
    - Loop through data to create sequences, labels, and vehicle capacities based on road width.

#### 12. Extract road widths and create input sequences
    - Extract road width data from DataFrame.
    - Define sequence length.
    - Create sequences and vehicle capacities using previous function.

#### 13. Split data into training and test sets
    - Use train_test_split to divide data into training and testing sets.

#### 14. Define LSTM model for traffic prediction
    - Initialize a Sequential LSTM model.
    - Add LSTM layers, Dropout, and a Dense layer for prediction.

#### 15. Train the LSTM model
    - Train the LSTM model using the training data.

#### 16. Make predictions using the trained model
    - Use the trained model to predict traffic congestion on the test set.

#### 17. Rescale predictions to original scale
    - Rescale the predicted traffic values to match the original scale.

#### 18. Adjust vehicle counts based on road capacity
    - Compare predicted vehicle count with maximum capacity and adjust as needed.

#### 19. Define zones from the original dataset
    - Extract unique zones from the dataset.

#### 20. Perform rerouting if predicted congestion exceeds capacity
    - If congestion exceeds capacity, suggest rerouting to a different zone.

#### 21. Plot training and validation loss
    - Visualize the model's training and validation loss over epochs.

#### 22. Output predicted congestion and rerouting suggestions
    - Display congestion predictions and rerouting recommendations for the first 24 hours.

#### 23. Create a DataFrame with necessary features for the prediction report
    - Generate a DataFrame with predicted congestion levels, rerouted zones, and capacities.

#### 24. Save the report to a CSV file
    - Save the prediction report to a CSV file.

#### 25. Add rerouted zones to the DataFrame
    - Add rerouted zones back to the original DataFrame.

#### 26. Group by zones and calculate average vehicle count before and after rerouting
    - Calculate the mean vehicle count, latitude, longitude, and altitude for each zone.

#### 27. Perform K-Means clustering before rerouting
    - Use KMeans clustering to group zones by vehicle count and geographic features.

#### 28. Plot clustering results before rerouting
    - Visualize clustering results before rerouting.

#### 29. Handle mismatched lengths of rerouted zones and original DataFrame
    - Adjust the rerouted zones array to match the length of the original DataFrame.

#### 30. Group data by rerouted zones and perform clustering
    - Group the rerouted zones and perform KMeans clustering if enough samples exist.

#### 31. Plot clustering results after rerouting
    - Visualize clustering results after rerouting.

#### 32. Group data by zone before rerouting and rerouted zones after rerouting
    - Calculate average vehicle count before and after rerouting.

#### 33. Merge before and after rerouting data for comparison
    - Merge the grouped data for comparison.

#### 34. Plot congestion diversion across zones before and after rerouting
    - Visualize the change in vehicle count across zones before and after rerouting.

#### 35. Visualize traffic intensity across different zones and times of day
    - Plot a heatmap of traffic volume by zones and times of day.

#### 36. Plot a clustered bar chart of traffic volume before and after rerouting by zone
    - Create a bar chart comparing vehicle counts before and after rerouting.

#### 37. Generate a traffic report from the data
    - Create a function to generate traffic reports with congestion, rerouting, and capacity data.

#### 38. Loop through the CSV data and generate traffic reports
    - For each row in the prediction data, generate a report.
