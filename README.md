# Field Evolution and Breast Cancer Prediction

## Overview

This project includes two main parts:

1. **Field Evolution Simulation**: A numerical simulation of field values evolving over space and time.
2. **Breast Cancer Prediction**: A machine learning model predicting breast cancer diagnosis using the Breast Cancer dataset.

## Files

- `model_code.py`: Contains the code for simulating the field evolution and training a linear regression model.
- `field_evolution_code.py`: Similar to `model_code.py`, simulates field evolution.
- `breast_cancer_code.py`: Contains the code for training and evaluating a linear regression model on the Breast Cancer dataset.

## Field Evolution Simulation

### What It Does

- Simulates the evolution of five fields (phi, H, W, Z, A) over space and time.
- Uses finite difference methods to approximate derivatives.
- Updates the fields using a discretized version of the Euler-Lagrange equation.
- Introduces dropout to improve stability.
- Trains a linear regression model to predict the field values.
- Evaluates the model and plots the results.

### Key Parameters

- `dx`: Spatial step size.
- `dt`: Time step size.
- `L`: Length of the spatial domain.
- `T`: Number of time steps.
- `dropout_rate`: Probability of dropping an update to improve stability.

### Outputs

- `field_evolution.png`: Plot showing the evolution of the fields over space.
- `field_predictions.png`: Plot comparing actual vs predicted field values.

## Breast Cancer Prediction

### What It Does

- Loads the Breast Cancer dataset from `sklearn.datasets`.
- Splits the data into training and testing sets.
- Trains a linear regression model on the training data.
- Evaluates the model using Mean Squared Error (MSE).
- Plots the actual vs predicted values for the test data.

### Outputs

- `breast_cancer_predictions.png`: Plot comparing actual vs predicted breast cancer diagnosis values.

## Running the Code

1. Ensure you have Python and the required libraries installed (`numpy`, `matplotlib`, `sklearn`).
2. Run the scripts (`model_code.py`, `field_evolution_code.py`, `breast_cancer_code.py`) in a Python environment.
3. View the generated plots and evaluation metrics.

### Run Command for ech

```sh
python model_code.py
python field_evolution_code.py
python breast_cancer_code.py
