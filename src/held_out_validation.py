import pandas as pd
import numpy as np
from pathlib import Path
import toml
import json
from src.utils import load_data, define_model, save_results
import os


def held_out_val(model_type):
    config = toml.load("config.toml")
    base_path = Path().resolve()

    # Check if best params should be used
    use_best = config.get("use_best_params", False)

    if use_best:
        params_type = "best_params"
        best_params_file = os.path.join(base_path, "results", "best_params.json")
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
            return
    else:
        params_type = "user_defined"
        # Load manually defined params from config.toml
        params = config["user_param"][model_type]
        print("Using manually defined parameters from config.toml")

    X_train, y_train = load_data("train")

    model = define_model(model_type, **params)
    model.fit(X_train, y_train)

    base_path = Path().resolve()
    X_test, y_test = load_data("test")
    final_df = pd.read_csv(os.path.join(base_path, "data", "processed", "final_df.csv"))
    TestGroup = pd.read_csv(
        os.path.join(base_path, "data", "processed", "TestGroup.csv")
    )

    # Add the predicted cost for the tasks in Test Group
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["Predicted Cost"])
    y_pred.index = X_test.index
    TestGroupDf = final_df[final_df["Task ID"].isin(TestGroup["Task ID"])][
        ["Task ID", "Supplier ID", "Cost"]
    ]
    TestGroupDf = pd.concat([TestGroupDf, y_pred], axis=1)

    # Get the actual cost for the supplier predicted by the model for each task
    Predicted_Supplier = TestGroupDf.loc[
        TestGroupDf.groupby("Task ID")["Predicted Cost"].idxmin()
    ][["Task ID", "Supplier ID", "Cost"]].reset_index(drop=True)
    Predicted_Supplier = Predicted_Supplier.set_index("Task ID", drop=True)
    Predicted_Supplier = Predicted_Supplier.rename(
        columns={
            "Supplier ID": "Predicted Supplier ID",
            "Cost": "Predicted Supplier Cost",
        }
    )

    # Get the cost for the supplier with the least actual cost for each task
    Actual_Supplier = TestGroupDf.loc[TestGroupDf.groupby("Task ID")["Cost"].idxmin()][
        ["Task ID", "Supplier ID", "Cost"]
    ].reset_index(drop=True)
    Actual_Supplier = Actual_Supplier.set_index("Task ID", drop=True)
    Actual_Supplier = Actual_Supplier.rename(
        columns={"Supplier ID": "Actual Supplier ID", "Cost": "Actual Cost"}
    )

    Error_t = pd.concat([Actual_Supplier, Predicted_Supplier], axis=1)
    Error_t["Error"] = Error_t["Actual Cost"] - Error_t["Predicted Supplier Cost"]

    # Calculating the RMSE for the Test Group tasks
    RMSE = np.sqrt(np.mean(Error_t["Error"] ** 2))
    print("RMSE score on the model: ", RMSE)

    save_results(model_type, "held_out_validation", params_type, params, RMSE)
