# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed
from pathlib import Path
import os
import toml
import json

from src.utils import define_model, evaluate_model, save_results


def cross_val(model_type):
    config = toml.load("config.toml")
    base_path = Path().resolve()

    # Check if best params should be used
    use_best = config.get("use_best_params", False)

    if use_best:
        best_params_file = os.path.join(base_path, "results", "best_params.json")
        params_type = "best_params"
        try:
            # Load best_params.json
            with open(best_params_file, "r") as f:
                best_params = json.load(f)
            if model_type in best_params:
                print("Using best parameters from best_params.json")
                params = best_params[model_type]
            else:
                print(
                    f"No best parameters found for {model_type}. Train the model with hp_optimize first."
                )
                return
        except FileNotFoundError:
            print(
                "best_params.json does not exist. Train the model with hp_optimize first."
            )
    else:
        params_type = "user_defined"
        # Load manually defined params from config.toml
        params = config["user_param"][model_type]
        print("Using manually defined parameters from config.toml")

    base_path = Path().resolve()
    file_path = os.path.join(base_path, "data", "processed")
    final_df = pd.read_csv(os.path.join(file_path, "final_df.csv"))

    X = final_df.iloc[:, 3:]  # Features
    y = final_df.iloc[:, 2]  # Costs
    groups = final_df.iloc[:, 0]  # Group by Tasks ID
    suppliers = final_df.iloc[:, 1]  # Supplier

    # Initialize Leave-One-Group-Out Cross-Validation
    logo = LeaveOneGroupOut()

    # Define the model
    model = define_model(model_type, **params)

    # List to hold custom scores, suppliers, and group names for each fold
    custom_scores = []
    supplier_names = []
    group_names = []

    # Use Parallel to compute custom scores in parallel
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(model, X, y, train_index, test_index, groups, suppliers)
        for train_index, test_index in logo.split(X, y, groups)
    )

    # Unzip the results into separate lists
    custom_scores, supplier_names, group_names = zip(*results)

    # Create DataFrame for custom scores, suppliers, and group names
    error_t_loocv = pd.DataFrame(
        {"Group": group_names, "Supplier": supplier_names, "Error": custom_scores}
    )

    # Calculate the RMSE
    RMSE_loocv = np.sqrt(np.mean(np.array(error_t_loocv["Error"]) ** 2))
    print(f"RMSE: {RMSE_loocv}")

    save_results(model_type, "cross_validation", params_type, params, RMSE_loocv)
