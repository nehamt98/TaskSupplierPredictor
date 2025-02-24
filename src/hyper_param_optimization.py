# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import argparse
import json
from src.utils import define_model, custom_score
import os
from pathlib import Path
import toml


def hp_optimize(model_type):
    config = toml.load("config.toml")

    base_path = Path().resolve()
    file_path = os.path.join(base_path, "data", "processed")
    final_df = pd.read_csv(os.path.join(file_path, "final_df.csv"))

    # Initialization
    X = final_df.iloc[:, 3:]  # Features
    y = final_df.iloc[:, 2]  # Target
    groups = final_df.iloc[:, 0]  # Grouping
    suppliers = final_df.iloc[:, 1]  # Supplier

    # Initialize Leave-One-Group-Out Cross-Validation
    logo = LeaveOneGroupOut()

    # Define the model
    model = define_model(model_type)

    # Define the parameter grid for GridSearchCV
    param_grid = config["models"][model_type]

    # Initialize GridSearchCV with Leave-One-Group-Out as the cross-validation strategy
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(grid_search_custom_score(suppliers)),
        cv=logo,
        n_jobs=-1,
    )

    # Fit the model to find the best parameters, passing groups
    grid_search.fit(X, y, groups=groups)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Save the best parameters to JSON
    best_params_path = os.path.join(base_path, "results", "best_params.json")

    # Load existing best_params.json if it exists
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params_dict = json.load(f)
    else:
        best_params_dict = {}

    # Update best parameters for the specific model
    best_params_dict[model_type] = best_params

    # Save updated best parameters back to best_params.json
    with open(best_params_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)

    print(f"Best parameters for {model_type} saved in best_params.json")


# Custom scorer for GridSearchCV
def grid_search_custom_score(suppliers):
    def custom_scorer(y_true, y_pred):
        score, _ = custom_score(y_true, y_pred, suppliers)
        return score

    return custom_scorer
