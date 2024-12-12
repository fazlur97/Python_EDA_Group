#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:27:24 2024

@author: nehathomas
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from joblib import Parallel, delayed
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor

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

# Training the model
dtr = DecisionTreeRegressor(max_depth=5, random_state=41)
dtr.fit(X_train, y_train)

# Scoring on the test data
print("Score on the model: ", dtr.score(X_test, y_test))

# Add the predicted cost for the tasks in Test Group
y_pred = dtr.predict(X_test)
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
    dtr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dtr.predict(X_test)
    
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


# Define the model for GridSearchCV
model = DecisionTreeRegressor(random_state=41)

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['squared_error']
}



# Custom scorer for GridSearchCV
def grid_search_custom_score(y_true, y_pred):
    score, _ = custom_score(y_true, y_pred, suppliers)
    return score

# Initialize GridSearchCV with Leave-One-Group-Out as the cross-validation strategy
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring=make_scorer(grid_search_custom_score),
                           cv=logo, n_jobs=-1)

# Fit the model to find the best parameters, passing groups
grid_search.fit(X, y, groups=groups)

# Get the best estimator and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

cv_results = grid_search.cv_results_
params = cv_results['params']  # List of all parameter combinations
mean_scores = cv_results['mean_test_score']  # Mean cross-validation score for each combination
std_scores = cv_results['std_test_score']  # Standard deviation of scores for each combination

# Print all parameter combinations with their corresponding scores
for param, mean, std in zip(params, mean_scores, std_scores):
    print(f"Params: {param}, Mean Score: {mean:.4f}, Std Dev: {std:.4f}")

# Print the best parameters found by GridSearchCV
print("Best Parameters:", best_params)


# Now you can use the best model for predictions
model = DecisionTreeRegressor(**best_params, random_state=41)

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
print(f"RMSE with Best Parameters: {RMSE_loocv}")