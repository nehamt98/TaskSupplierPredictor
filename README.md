# TaskSupplierPredictor
## Project Overview
This project explores the use of Machine Learning models to select the optimal supplier for a given task. The focus is on comparing and tuning different models and hyperparameters. The aim is to evaluate model performance on the given dataset and optimize parameters for better decision-making.

The dataset consists of task and supplier data, along with recorded costs. The goal is to develop an ML approach to predict the best supplier for each task based on cost efficiency.

> **Note:** This project is designed to analyze and compare models rather than serve as a deployable ML system for real-time predictions.

## Installation
### Create & Activate Conda Environment
```
conda create --name supplier_predictor_venv python=3.12
conda activate supplier_predictor_venv
```
### Install dependencies
conda env create -f environment.yml

## Running the models
Execute the following command with desired options:
```
python main.py --model <model_name> [options]
```
#### **Available Arguments:**
| Argument        | Description  |
|----------------|-------------|
| `--model`      | Model to run (`linear`, `lasso`, `decision_tree`, `random_forest`, `mlp`). **(Mandatory)** |
| `--preprocess` | Runs preprocessing before training. |
| `--hp_optimize` | Performs hyperparameter tuning and stores best parameters in `results/best_params.json`. |
| `--cross_val` | Runs cross-validation instead of held-out validation and stores RMSE in `results/results.json`. |

By default, if --cross_val is not passed, held-out validation is performed.

## Configuration: config.toml
The config.toml file contains the parameters to tune the model.
### Selecting Parameter Source
If use_best_params = true, held-out validation and cross-validation will use parameters from results/best_params.json. (Make sure to do hyper-parameter optimization before this)

If use_best_params = false, they will use parameters from [user_param].
### User-Defined Parameters
User parameters for each model can be modified in this file under [user_param.<model_name>].
### Hyperparameter Optimization Grid
The parameter grid for hyper-parameter optimization can be modified under [models.<model_name>]

## Model Evaluation: RMSE Calculation
The Root Mean Squared Error (**RMSE**) is used as the evaluation metric, calculated as follows:

1. **Predict costs** for test tasks.
2. **Identify the supplier** chosen by the model for each task (with the lowest predicted cost).
3. **Compare the actual cost** for the selected supplier vs. the best possible supplier.
4. Compute RMSE using:

```math
RMSE = \sqrt{ \frac{1}{N} \sum (\text{Actual Cost} - \text{Predicted Supplier Cost})^2 }
```

## Results Storage
Best Hyperparameters are stored in results/best_params.json.

RMSE Scores for held-out and cross-validation runs are stored in results/results.json.

The project does not store trained models since the focus is on comparing different approaches.
