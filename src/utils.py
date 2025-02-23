import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os

def load_data(path):
    base_path = Path().resolve()
    X = pd.read_csv(os.path.join(base_path, "data", "processed", path, "data.csv"), index_col=0)
    y = pd.read_csv(os.path.join(base_path, "data", "processed", path, "labels.csv"), index_col=0)
    return X, y

def define_model(model_type, **parameters):
    models = {
        "linear": LinearRegression,
        "lasso": Lasso,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "mlp": MLPRegressor
    }
    if model_type not in models:
        raise ValueError("Model type not supported. Choose from: linear, lasso, decision_tree, random_forest, mlp")
    
    model = models[model_type](**parameters)
    return model

def custom_score(y_true, y_pred, suppliers):
    # Custom scoring function to calculate the difference between the minimum actual and minimum predicted values in each group

    min_y_true = np.min(y_true)
    min_y_pred_index = np.argmin(y_pred)
    
    # Get the supplier name corresponding to the minimum predicted value
    corresponding_supplier = suppliers.iloc[min_y_pred_index] 
    # Take the true value of supplier cost selected by the ml model 
    actual_cost_by_prediction = y_true.iloc[min_y_pred_index]

    return min_y_true - actual_cost_by_prediction , corresponding_supplier  # Calculate the difference and return supplier name

def evaluate_model(model, X, y, train_index, test_index, groups, suppliers):
    # Function to evaluate model and calculate scores for each fold
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