#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:17:39 2024

@author: nehathomas
"""

# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Read the task data
task_data = pd.read_csv('../Original_datasets/tasks.csv', index_col=0)

# Read the supplier data
supplier_data = pd.read_csv('../Original_datasets/suppliers.csv')

# Read the cost data
cost_data = pd.read_csv('../Original_datasets/cost.csv') 

# Step 1.1

# Check for missing data in all three datasets
task_missing = task_data.isnull().sum().sum()
supplier_missing = supplier_data.isnull().sum().sum()
cost_missing = cost_data.isnull().sum().sum()

# Display the results
print(f"Missing values in tasks.csv: {task_missing}")
print(f"Missing values in suppliers.csv: {supplier_missing}")
print(f"Missing values in cost.csv: {cost_missing}\n")

# Check for duplicates in each dataset
task_duplicate = task_data.duplicated().sum()
supplier_duplicate = supplier_data.duplicated().sum()
cost_duplicate = cost_data.duplicated().sum()

# Display the results
print(f"Duplicate rows in tasks.csv: {task_duplicate}")
print(f"Duplicate rows in suppliers.csv: {supplier_duplicate}")
print(f"Duplicate rows in cost.csv: {cost_duplicate}\n")

# Check the data format
print("Task data information:")
task_data.info()
print("\nSupplier data information:")
supplier_data.info()
print("\nCost data information:")
cost_data.info()
print("\n")


# Step 1.2 - Task data

print("Task data feature selecion:\n")

# Remove columns that contain the same value for all rows
# Find columns that contain only 1 value and  drop those columns
columns_to_drop = task_data.columns[task_data.nunique() == 1]  
task_data = task_data.drop(columns_to_drop,axis=1)
columns_to_drop = list(set(columns_to_drop))
# Print results
print(f"Columns dropped due to same values: {columns_to_drop}\n")

# Convert percentage(%) to float
task_data["TF5"] = task_data["TF5"].str.strip("%").astype(float)/100
task_data["TF7"] = task_data["TF7"].str.strip("%").astype(float)/100

# Check for correlation between the features and drop one of the columns if correlation is high
# Compute the correlation matrix
correlation_matrix = task_data.corr()
# Set the threshold for dropping features
threshold = 0.8
# Create a mask for the upper triangle
upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1)
# Identify columns to drop (keep only one column for each correlated pair)
columns_to_drop = []
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):  # Upper triangle check
        if correlation_matrix.iloc[i, j] > threshold:  # High correlation
            colname = correlation_matrix.columns[j]  # Keep one column
            columns_to_drop.append(colname)
# Drop duplicate columns from the dataset
columns_to_drop = list(set(columns_to_drop))  # Remove duplicates
new_tasks = task_data.drop(columns = columns_to_drop)
# Print results
print(f"Columns dropped due to high correlation: {columns_to_drop}")
print(f"\nRemaining columns: {new_tasks.columns.tolist()}\n")

# Box Plots to Identify Outliers for task_data
plt.figure(figsize=(20, 10))
sns.boxplot(data=task_data, palette="coolwarm")
plt.title("Distribution of Task Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Scaling tasks data
scaler = StandardScaler()
scaled_tasks = scaler.fit_transform(new_tasks)
scaled_tasks = pd.DataFrame(scaled_tasks, columns = new_tasks.columns, index = new_tasks.index)

# Feature selection with PCA
pca = PCA(n_components=10)
tasks_pca = pca.fit_transform(scaled_tasks) 
tasks_pca = pd.DataFrame(tasks_pca, index = scaled_tasks.index)
# Set the column names to TF-PC1, TF-PC2, ..., TF-PC10
tasks_pca.columns = [f"TF-PC{i+1}" for i in range(tasks_pca.shape[1])]


# Step 1.2 - Suppliers data

print("\nSuppliers data feature selecion:\n")

# Transposing the dataset to get features as the columns
supplier_data = supplier_data.transpose()
supplier_data.reset_index(inplace=True)
supplier_data.columns = supplier_data.iloc[0]  
supplier_data = supplier_data[1:]
supplier_data.rename(columns={supplier_data.columns[0]: "Supplier ID"}, inplace=True)
supplier_data.set_index("Supplier ID", inplace=True)

# Check for correlation between the features and drop one of the columns if correlation is high
# Compute the correlation matrix
correlation_matrix = supplier_data.corr()
# Set the threshold for dropping features
threshold = 0.8
# Create a mask for the upper triangle
upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1)
# Identify columns to drop (keep only one column for each correlated pair)
columns_to_drop = []
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):  # Upper triangle check
        if correlation_matrix.iloc[i, j] > threshold:  # High correlation
            colname = correlation_matrix.columns[j]  # Keep one column
            columns_to_drop.append(colname)
# Drop duplicate columns from the dataset
columns_to_drop = list(set(columns_to_drop))  # Remove duplicates
print(f"Columns dropped due to high correlation: {columns_to_drop}")
print("There is no multicollinearity in the suppliers dataset, hence no columns dropped\n")

# Box Plots to Identify Outliers for supplier_data
plt.figure(figsize=(20, 10))
sns.boxplot(data=supplier_data, palette="coolwarm")
plt.title("Distribution of Supplier Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Scaling suppliers data
scaled_suppliers = scaler.fit_transform(supplier_data)
scaled_suppliers = pd.DataFrame(scaled_suppliers, columns = supplier_data.columns, index = supplier_data.index)

# Feature selection with PCA
suppliers_pca = pca.fit_transform(scaled_suppliers) 
suppliers_pca = pd.DataFrame(suppliers_pca, index = scaled_suppliers.index)
# Set the column names to SF-PC1, SF-PC2, ..., SF-PC10
suppliers_pca.columns = [f"SF-PC{i+1}" for i in range(suppliers_pca.shape[1])]


# Step 1.3 - Top Supplier Identification

# Identifying the top-performing suppliers for each task
top_supplier = cost_data.loc[cost_data.groupby("Task ID")["Cost"].idxmin()]

# Identifying the top 10 suppliers for each task
new_cost = (
    cost_data.groupby('Task ID')
    .apply(lambda group: group.nsmallest(10, 'Cost'))
    .reset_index(drop=True)
)

# Save to csv
scaled_tasks.to_csv("../Processed_datasets/scaled_tasks.csv")
scaled_suppliers.to_csv("../Processed_datasets/scaled_suppliers.csv")
tasks_pca.to_csv("../Processed_datasets/pca_tasks.csv")
suppliers_pca.to_csv("../Processed_datasets/pca_suppliers.csv")
new_cost.to_csv("../Processed_datasets/best_suppliers_cost.csv", index=False)