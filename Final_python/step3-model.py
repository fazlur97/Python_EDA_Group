#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:42:13 2024

@author: nehathomas
"""
# Import libraries
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

# Read data
df_task = pd.read_csv('../Processed_datasets/pca_tasks.csv', index_col="Task ID")
df_supply = pd.read_csv('../Processed_datasets/pca_suppliers.csv', index_col="Supplier ID")
df_cost = pd.read_csv('../Processed_datasets/best_suppliers_cost.csv')


# Step 3.1 - Merge data

# Merge cost_df with task_df on TaskID
merged_df = pd.merge(df_cost, df_task, on="Task ID")
# Merge the resulting DataFrame with supplier_df on SupplierID
final_df = pd.merge(merged_df, df_supply, on="Supplier ID")

# Get the model variables
X=final_df.iloc[:,3:]
y=final_df.iloc[:,2]


# Step 3.2

# Get the test group
TestGroup = final_df['Task ID'].drop_duplicates().sample(n=20, random_state=41)

# Split into train and test based on TaskID
X_test = final_df[final_df['Task ID'].isin(TestGroup)][X.columns]
y_test = final_df[final_df['Task ID'].isin(TestGroup)]['Cost']

X_train = final_df[~final_df['Task ID'].isin(TestGroup)][X.columns]
y_train = final_df[~final_df['Task ID'].isin(TestGroup)]['Cost']


# Step 3.3

# Training a neural network regression model
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=100, random_state=41)
mlp.fit(X_train, y_train)

# Scoring on the test data
print("Score on the model: ", mlp.score(X_test, y_test))


# Step 3.4

# Add the predicted cost for the tasks in Test Group
y_pred = mlp.predict(X_test)
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

# Save to csv
final_df.to_csv("../Processed_datasets/final_df.csv", index=False)