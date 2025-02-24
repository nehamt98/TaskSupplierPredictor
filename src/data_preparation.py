# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from pathlib import Path
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)


def preprocess():
    base_path = Path().resolve()

    # Read the task data
    task_data = pd.read_csv(os.path.join(base_path, "data", "raw", "tasks.csv"))
    task_data.set_index("Task ID", inplace=True)

    # Read the supplier data
    supplier_data = pd.read_csv(os.path.join(base_path, "data", "raw", "suppliers.csv"))

    # Read the cost data
    cost_data = pd.read_csv(os.path.join(base_path, "data", "raw", "cost.csv"))

    # Task data feature selecion
    print("Task data feature selecion:\n")

    # Remove columns that contain the same value for all rows
    # Find columns that contain only 1 value and  drop those columns
    columns_to_drop = task_data.columns[task_data.nunique() == 1]
    task_data = task_data.drop(columns_to_drop, axis=1)
    columns_to_drop = list(set(columns_to_drop))
    # Print results
    print(f"Columns dropped due to same values: {columns_to_drop}\n")

    # Convert percentage(%) to float
    task_data["TF5"] = task_data["TF5"].str.strip("%").astype(float) / 100
    task_data["TF7"] = task_data["TF7"].str.strip("%").astype(float) / 100

    # Check for correlation between the features and drop one of the columns if correlation is high
    new_tasks = correlation(task_data, 0.8)

    # Scaling tasks data
    scaled_tasks = scale_data(new_tasks)

    # Feature selection with PCA
    tasks_pca = PCA_df(scaled_tasks, 10, "TF")

    # Suppliers data feature selecion
    print("\nSuppliers data feature selecion:\n")

    # Transposing the dataset to get features as the columns
    supplier_data = supplier_data.transpose()
    supplier_data.reset_index(inplace=True)
    supplier_data.columns = supplier_data.iloc[0]
    supplier_data = supplier_data[1:]
    supplier_data.rename(
        columns={supplier_data.columns[0]: "Supplier ID"}, inplace=True
    )
    supplier_data.set_index("Supplier ID", inplace=True)

    # Check for correlation between the features and drop one of the columns if correlation is high
    new_suppliers = correlation(supplier_data, 0.8)

    # Scaling suppliers data
    scaled_suppliers = scale_data(new_suppliers)

    # Feature selection with PCA
    suppliers_pca = PCA_df(scaled_suppliers, 10, "SF")

    # Identifying the top 45 suppliers for each task
    new_cost = (
        cost_data.groupby("Task ID", sort=False)
        .apply(lambda group: group.nsmallest(45, "Cost"))
        .sort_index()
        .reset_index(drop=True)
    )

    # Merge cost_df with task_df on TaskID
    merged_df = pd.merge(new_cost, tasks_pca, on="Task ID")
    # Merge the resulting DataFrame with supplier_df on SupplierID
    final_df = pd.merge(merged_df, suppliers_pca, on="Supplier ID")

    # Get the model variables
    X = final_df.iloc[:, 3:]
    y = final_df.iloc[:, 2]

    # Get the test group as a random set of 20 tasks
    TestGroup = final_df["Task ID"].drop_duplicates().sample(n=20, random_state=41)

    # Save to csv
    path = os.path.join(base_path, "data", "processed")
    os.makedirs(path, exist_ok=True)
    final_df.to_csv(os.path.join(path, "final_df.csv"), index=False)
    TestGroup.to_csv(os.path.join(path, "TestGroup.csv"), index=False)

    # Split into train and test based on TaskID
    X_test = final_df[final_df["Task ID"].isin(TestGroup)][X.columns]
    y_test = final_df[final_df["Task ID"].isin(TestGroup)]["Cost"]

    X_train = final_df[~final_df["Task ID"].isin(TestGroup)][X.columns]
    y_train = final_df[~final_df["Task ID"].isin(TestGroup)]["Cost"]

    # Save to csv
    path = os.path.join(base_path, "data", "processed", "test")
    os.makedirs(path, exist_ok=True)
    X_test.to_csv(os.path.join(path, "data.csv"), index=True)
    y_test.to_csv(os.path.join(path, "labels.csv"), index=True)

    path = os.path.join(base_path, "data", "processed", "train")
    os.makedirs(path, exist_ok=True)
    X_train.to_csv(os.path.join(path, "data.csv"), index=True)
    y_train.to_csv(os.path.join(path, "labels.csv"), index=True)


def correlation(df, threshold):
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    # Create a mask for the upper triangle (excluding the diagonal)
    upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

    # Get column names where correlation is above the threshold
    columns_to_drop = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if upper_triangle_mask[i, j] and correlation_matrix.iloc[i, j] > threshold:
                colname = correlation_matrix.columns[j]
                columns_to_drop.add(colname)

    # Convert to list for output
    columns_to_drop = list(columns_to_drop)
    new_df = df.drop(columns=columns_to_drop)
    print(f"Columns dropped due to high correlation: {columns_to_drop}")
    return new_df


def scale_data(df):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns, index=df.index)
    return scaled_df


def PCA_df(df, columns, col_name):
    pca = PCA(n_components=columns)
    df_pca = pca.fit_transform(df)
    df_pca = pd.DataFrame(df_pca, index=df.index)
    # Set the column names
    df_pca.columns = [f"{col_name}-PC{i+1}" for i in range(df_pca.shape[1])]
    return df_pca
