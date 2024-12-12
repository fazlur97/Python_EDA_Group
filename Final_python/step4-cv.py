#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:14:28 2024

@author: nehathomas
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from joblib import Parallel, delayed

# Read data
final_df = pd.read_csv('../Processed_datasets/final_df.csv')


# Step 4: Perform Leave-One-Group-Out with the custom scorer

# Custom scoring function to calculate the difference between the minimum actual and minimum predicted values in each group
def custom_score(y_true, y_pred, suppliers):
    min_y_true = np.min(y_true)
    min_y_pred_index = np.argmin(y_pred)
    
    # Get the supplier name corresponding to the minimum predicted value
    corresponding_supplier = suppliers.iloc[min_y_pred_index] 
    # Take the true value of supplier cost selected by the ml model 
    actual_cost_by_prediction = y_true.iloc[min_y_pred_index]

    return min_y_true - actual_cost_by_prediction , corresponding_supplier  # Calculate the difference and return supplier name

# Initialization
X = final_df.iloc[:, 3:]  # Features 
y = final_df.iloc[:, 2]    # Costs
groups = final_df.iloc[:, 0]  # Group by Tasks ID
suppliers = final_df.iloc[:, 1]  # Supplier 

# Initialize Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Initialize the MLP Regressor
model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=100, random_state=41)

# List to hold custom scores, suppliers, and group names for each fold
custom_scores = []
supplier_names = []
group_names = []

# Running in parallel to make the process more efficient

# Function to evaluate model and calculate scores for each fold
def evaluate_model(train_index, test_index):
    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate the custom score for this fold
    score, corresponding_supplier = custom_score(y_test, y_pred, suppliers.iloc[test_index])
    return score, corresponding_supplier, groups.iloc[test_index[0]]  # Return score, supplier, and group name

# Use Parallel to compute custom scores in parallel
results = Parallel(n_jobs=-1)(
    delayed(evaluate_model)(train_index, test_index)
    for train_index, test_index in logo.split(X, y, groups)
)

# Unzip the results into separate lists
custom_scores, supplier_names, group_names = zip(*results)

# Create DataFrame for custom scores, suppliers, and group names
error_t_loocv = pd.DataFrame({
    'Group': group_names,
    'Supplier': supplier_names,
    'Error': custom_scores
})

# Calculate the RMSE
RMSE_loocv = np.sqrt(np.mean(np.array(error_t_loocv['Error'])**2))
print(f"RMSE: {RMSE_loocv}")