#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:37:43 2024

@author: nehathomas
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from joblib import Parallel, delayed

# Read data
final_df = pd.read_csv('../Processed_datasets/final_df.csv')

# Initialization
X = final_df.iloc[:, 3:]  # Features 
y = final_df.iloc[:, 2]    # Target 
groups = final_df.iloc[:, 0]  # Grouping 
suppliers = final_df.iloc[:, 1]  # Supplier 

# Get the test group
TestGroup = final_df['Task ID'].drop_duplicates().sample(n=20, random_state=41)

# Split into train and test based on TaskID
X_test = final_df[final_df['Task ID'].isin(TestGroup)][X.columns]
y_test = final_df[final_df['Task ID'].isin(TestGroup)]['Cost']

X_train = final_df[~final_df['Task ID'].isin(TestGroup)][X.columns]
y_train = final_df[~final_df['Task ID'].isin(TestGroup)]['Cost']

# Training a linear regression model
ln = LinearRegression()
ln.fit(X_train, y_train)

# Scoring on the test data
print("Score on the model: ", ln.score(X_test, y_test))

# Add the predicted cost for the tasks in Test Group
y_pred = ln.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns = ["Predicted Cost"])
y_pred.index = X_test.index
TestGroupDf = final_df[final_df['Task ID'].isin(TestGroup)][['Task ID','Supplier ID','Cost']]
TestGroupDf = pd.concat([TestGroupDf, y_pred], axis = 1)

# Get the actual cost for the supplier predicted by the model for each task
Predicted_Supplier = TestGroupDf.loc[TestGroupDf.groupby('Task ID')['Predicted Cost'].idxmin()][['Task ID','Supplier ID','Cost']].reset_index(drop = True)
Predicted_Supplier = Predicted_Supplier.set_index("Task ID", drop=True)
Predicted_Supplier = Predicted_Supplier.rename(columns={
    "Supplier ID": "Predicted Supplier ID",
    "Cost": "Predicted Supplier Cost"
})

# Get the cost for the supplier with the least actual cost for each task
Actual_Supplier = TestGroupDf.loc[TestGroupDf.groupby('Task ID')['Cost'].idxmin()][['Task ID','Supplier ID','Cost']].reset_index(drop = True)
Actual_Supplier = Actual_Supplier.set_index("Task ID", drop=True)
Actual_Supplier = Actual_Supplier.rename(columns={
    "Supplier ID": "Actual Supplier ID",
    "Cost": "Actual Cost"
})

Error_t = pd.concat([Actual_Supplier,Predicted_Supplier], axis = 1)
Error_t['Error'] = Error_t['Actual Cost'] - Error_t['Predicted Supplier Cost']

# Calculating the RMSE for the Test Group tasks
RMSE = np.sqrt(np.mean(Error_t['Error']**2))
print("RMSE score on the model: ", RMSE)


# Custom scoring function to calculate the difference between the minimum actual and minimum predicted values in each group
def custom_score(y_true, y_pred, suppliers):
    min_y_true = np.min(y_true)
    min_y_pred_index = np.argmin(y_pred)
    
    # Get the supplier name corresponding to the minimum predicted value
    corresponding_supplier = suppliers.iloc[min_y_pred_index] 
    # Take the true value of supplier cost selected by the ml model 
    actual_cost_by_prediction = y_true.iloc[min_y_pred_index]

    return min_y_true - actual_cost_by_prediction , corresponding_supplier  # Calculate the difference and return supplier name

# Initialize Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

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
    ln.fit(X_train, y_train)
    
    # Make predictions
    y_pred = ln.predict(X_test)
    
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
print(f"RMSE using logo CV: {RMSE_loocv}")