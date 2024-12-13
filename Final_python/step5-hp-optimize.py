# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed

# Read data
final_df = pd.read_csv('../Processed_datasets/final_df.csv')

# Initialization
X = final_df.iloc[:, 3:]  # Features 
y = final_df.iloc[:, 2]    # Target 
groups = final_df.iloc[:, 0]  # Grouping 
suppliers = final_df.iloc[:, 1]  # Supplier 

# Initialize Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Define the model
model = MLPRegressor(random_state=41)

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'max_iter': [100],
    'alpha': [0.0001, 0.001],
}


# Custom scoring function to calculate the difference between the minimum actual and minimum predicted values in each group
def custom_score(y_true, y_pred, suppliers):
    min_y_true = np.min(y_true)
    min_y_pred_index = np.argmin(y_pred)
    
    # Get the supplier name corresponding to the minimum predicted value
    corresponding_supplier = suppliers.iloc[min_y_pred_index] 
    
    # Take the true value of supplier cost selected by the ml model 
    actual_cost_by_prediction = y_true.iloc[min_y_pred_index]

    return min_y_true - actual_cost_by_prediction , corresponding_supplier  # Calculate the difference and return supplier name

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
model = MLPRegressor(**best_params, random_state=41)

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