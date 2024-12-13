# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
scaled_tasks = pd.read_csv('../Processed_datasets/scaled_tasks.csv', index_col="Task ID") 
tasks_pca = pd.read_csv('../Processed_datasets/pca_tasks.csv', index_col="Task ID")
scaled_suppliers = pd.read_csv('../Processed_datasets/scaled_suppliers.csv', index_col="Supplier ID") 
suppliers_pca = pd.read_csv('../Processed_datasets/pca_suppliers.csv', index_col="Supplier ID")
cost_data = pd.read_csv('../Original_datasets/cost.csv')
new_cost = pd.read_csv('../Processed_datasets/best_suppliers_cost.csv')

# Step 2.1 - EDA on task features

# Univariate analysis

# Histograms for Task Features before PCA
scaled_tasks.hist(bins=20, figsize=(20, 15), grid=False, color="skyblue", edgecolor="black")
plt.suptitle("Histograms of Task Features before PCA", fontsize=16)
plt.tight_layout()
plt.show()

# Histograms for Task Features after PCA
tasks_pca.hist(bins=20, figsize=(20, 15), grid=False, color="skyblue", edgecolor="black")
plt.suptitle("Histograms of Task Features after PCA", fontsize=16)
plt.tight_layout()
plt.show()

# Box Plots to Identify Outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=scaled_tasks, palette="coolwarm")
plt.title("Box Plot of Task Features before PCA")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Box Plots to Identify Outliers After PCA
plt.figure(figsize=(20, 10))
sns.boxplot(data=tasks_pca, palette="coolwarm")
plt.title("Box Plot of Task Features after PCA")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Multivariate analysis

# Correlation Matrix and Heatmap
correlation_matrix = scaled_tasks.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()


# Step 2.1 - EDA on suppliers features

# Univariate analysis

# Histograms for Suppliers Features before PCA
scaled_suppliers.hist(bins=20, figsize=(20, 15), grid=False, color="skyblue", edgecolor="black")
plt.suptitle("Histograms of Suppliers Features before PCA", fontsize=16)
plt.tight_layout()
plt.show()

# Histograms for Suppliers Features after PCA
suppliers_pca.hist(bins=20, figsize=(20, 15), grid=False, color="skyblue", edgecolor="black")
plt.suptitle("Histograms of Suppliers Features before PCA", fontsize=16)
plt.tight_layout()
plt.show()

# Box Plots to Identify Outliers Before PCA
plt.figure(figsize=(20, 10))
sns.boxplot(data=scaled_suppliers, palette="coolwarm")
plt.title("Box Plot of Supplier Features before PCA")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Box Plots to Identify Outliers After PCA
plt.figure(figsize=(20, 10))
sns.boxplot(data=suppliers_pca, palette="coolwarm")
plt.title("Box Plot of Supplier Features after PCA")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Multivariate analysis

# Suppliers Feature Correlation
corr_matrix = scaled_suppliers.corr()

plt.figure(figsize=(22, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size": 10},  # Adjust font size of annotations
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Step 2.2 - Analysing cost data

# Distribution of Costs
plt.figure(figsize=(10, 6))
sns.histplot(cost_data['Cost'], kde=True, bins=30, color='blue')
plt.title('Distribution of Costs', fontsize=16)
plt.xlabel('Cost', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Distribution of Costs of 45 best suppliers for each task
plt.figure(figsize=(10, 6))
sns.histplot(new_cost['Cost'], kde=True, bins=30, color='blue')
plt.title('Distribution of Costs of 45 best suppliers for each task', fontsize=16)
plt.xlabel('Cost', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Strip Plot of Costs Under 0.32 for Each Supplier Across Tasks
low_costs = new_cost[new_cost["Cost"]<0.32]
plt.figure(figsize=(15, 8))
sns.stripplot(data=low_costs, x='Supplier ID', y='Cost', hue='Task ID')
plt.title('Strip Plot of Costs Under 0.32 for Each Supplier Across Tasks', fontsize=16)
plt.xlabel('Supplier ID', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Task ID', loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# Step 2.3 - Exploring errors across suppliers for each task

# Copy the cost df of 45 best suppliers
cost_error = new_cost.copy()
# Compute the minimum cost for each task
cost_error['Min cost'] = cost_error.groupby('Task ID')['Cost'].transform('min')
# Calculate the error for all the suppliers for each tasks
cost_error['Error'] = cost_error['Min cost'] - cost_error['Cost']

# RMSE for each supplier across all the tasks
RMSE = cost_error.groupby('Supplier ID')[['Error']].apply(lambda x: np.sqrt(np.mean(x**2)))
RMSE = RMSE.reset_index()
RMSE.columns = ['Supplier ID', 'RMSE']
# RMSE across the suppliers
print("Max RMSE across the suppliers:", RMSE["RMSE"].max())
print("Min RMSE across the suppliers:", RMSE["RMSE"].min())
print("Mean RMSE across the suppliers:", RMSE["RMSE"].mean())

# Strip Plot of Errors for Each Task across 45 best Suppliers
plt.figure(figsize=(15, 8))
sns.stripplot(data=cost_error, x='Task ID', y='Error', jitter=True, alpha=0.7, hue = "Supplier ID", legend = False)
plt.title('Strip Plot of Errors for Each Task across 45 best Suppliers', fontsize=16)
plt.xlabel('Task ID', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Bar plot for the mean error per task
task_error_summary = cost_error.groupby('Task ID')['Error'].mean().reset_index()
# Sort tasks by mean error
task_error_summary = task_error_summary.sort_values('Error', ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(data=task_error_summary, x='Task ID', y='Error')
plt.title('Mean Error per Task across 45 best Suppliers', fontsize=16)
plt.xlabel('Task ID', fontsize=14)
plt.ylabel('Mean Error', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plotting RMSE across the suppliers
plt.figure(figsize=(12,6))
plt.bar(RMSE["Supplier ID"],RMSE["RMSE"])
plt.xticks(rotation=90)
plt.title('Root Mean Squared Error (RMSE) per Supplier', fontsize=16)
plt.xlabel('RMSE', fontsize=14)
plt.ylabel('Supplier ID', fontsize=14)
plt.tight_layout()
plt.show()