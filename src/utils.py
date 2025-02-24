import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime
import json

base_path = Path().resolve()


def load_data(path):
    '''
    Loads the feature data and target labels from CSV files.

    Args:
        path (str): The relative path to the directory containing the data files (Either train or test).

    Returns:
        tuple: A tuple containing two elements:
            - X (DataFrame): The feature set (data).
            - y (DataFrame): The target variable (labels).
    '''
    X = pd.read_csv(
        os.path.join(base_path, "data", "processed", path, "data.csv"), index_col=0
    )
    y = pd.read_csv(
        os.path.join(base_path, "data", "processed", path, "labels.csv"), index_col=0
    )
    return X, y


def define_model(model_type, **parameters):
    '''
    Defines and initializes a machine learning model based on the specified type.

    Args:
        model_type (str): The type of model to initialize (e.g., "linear", "lasso").
        **parameters: Hyperparameters to be passed to the model constructor.

    Returns:
        model: The initialized machine learning model.

    Raises:
        ValueError: If the model type is unsupported.
    '''
    models = {
        "linear": LinearRegression,
        "lasso": Lasso,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "mlp": MLPRegressor,
    }
    if model_type not in models:
        raise ValueError(
            "Model type not supported. Choose from: linear, lasso, decision_tree, random_forest, mlp"
        )

    model = models[model_type](**parameters)
    return model


def custom_score(y_true, y_pred, suppliers):
    '''
    A custom scoring function that calculates the difference between the minimum true value 
    and the minimum predicted value, and returns the corresponding supplier and actual cost.

    Args:
        y_true (Series): The true values for the target variable.
        y_pred (Series): The predicted values from the model.
        suppliers (DataFrame): DataFrame containing supplier information.

    Returns:
        tuple: A tuple containing:
            - The difference between the minimum true value and the predicted value.
            - The supplier corresponding to the minimum predicted value.
    '''

    min_y_true = np.min(y_true)
    min_y_pred_index = np.argmin(y_pred)

    # Get the supplier name corresponding to the minimum predicted value
    corresponding_supplier = suppliers.iloc[min_y_pred_index]
    # Take the true value of supplier cost selected by the ml model
    actual_cost_by_prediction = y_true.iloc[min_y_pred_index]

    return (
        min_y_true - actual_cost_by_prediction,
        corresponding_supplier,
    )  # Calculate the difference and return supplier name


def evaluate_model(model, X, y, train_index, test_index, groups, suppliers):
    '''
    Evaluates the model using training and test splits, and calculates a custom score.

    Args:
        model: The machine learning model to evaluate.
        X (DataFrame): Features (input data).
        y (DataFrame): Target variable (labels).
        train_index (array): Indices for the training set.
        test_index (array): Indices for the test set.
        groups (DataFrame): Group information for cross-validation.
        suppliers (DataFrame): Supplier data used for scoring.

    Returns:
        tuple: A tuple containing:
            - The custom score for the test fold.
            - The corresponding supplier for the predicted minimum value.
            - The group name for the fold.
    '''
    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the custom score for this fold
    score, corresponding_supplier = custom_score(
        y_test, y_pred, suppliers.iloc[test_index]
    )
    return (
        score,
        corresponding_supplier,
        groups.iloc[test_index[0]],
    )  # Return score, supplier, and group name


def save_results(model_name, validation_type, params_type, params, rmse):
    '''
    Saves the model evaluation results (including hyperparameters and RMSE score) to a JSON file.

    Args:
        model_name (str): The name of the model
        validation_type (str): The type of validation used ("held_out_validation" or "cross_validation").
        params_type (str): Indicates if the parameters are "user_defined" or "best_params".
        params (dict): A dictionary of the model's hyperparameters.
        rmse (float): The RMSE (Root Mean Squared Error) score of the model.

    Returns:
        None: The results are saved to the `results.json` file.
    '''
    # Load existing results or create an empty dictionary
    results_dir = os.path.join(base_path, "results")  # Ensure 'results/' exists
    results_path = os.path.join(results_dir, "results.json")  # Path to results.json

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Create the results.json file if it doesn't exist
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            json.dump({}, f)  # Initialize with an empty JSON object

    with open(results_path, "r") as f:
        results = json.load(f)

    # Ensure model and validation structure exists
    if model_name not in results:
        results[model_name] = {}
    if validation_type not in results[model_name]:
        results[model_name][validation_type] = {}

    # Create result entry
    result_entry = {
        "params": params,
        "rmse": rmse,
        "timestamp": datetime.now().isoformat(),
    }

    # Append user-defined results
    if params_type == "user_defined":
        if "user_defined" not in results[model_name][validation_type]:
            results[model_name][validation_type]["user_defined"] = []
        entry_index = next(
            (
                i
                for i, entry in enumerate(
                    results[model_name][validation_type]["user_defined"]
                )
                if entry["params"] == params
            ),
            None,
        )

        if entry_index is not None:
            # Replace the existing entry
            results[model_name][validation_type]["user_defined"][
                entry_index
            ] = result_entry
        else:
            # Append new entry
            results[model_name][validation_type]["user_defined"].append(result_entry)

    # Overwrite best_params
    elif params_type == "best_params":
        results[model_name][validation_type]["best_params"] = result_entry

    # Write back to file
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved for {model_name} ({validation_type}, {params_type})")
