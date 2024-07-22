
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X_bc = breast_cancer.data
y_bc = breast_cancer.target

# Split the data into training and testing sets
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

# Train a simple linear regression model
model_bc = LinearRegression()
model_bc.fit(X_train_bc, y_train_bc)

# Make predictions
y_pred_bc = model_bc.predict(X_test_bc)

# Evaluate the model
mse_bc = mean_squared_error(y_test_bc, y_pred_bc)
print(f"Mean Squared Error (Breast Cancer): {mse_bc}")

# Plot predictions vs actual
plt.figure(figsize=(12, 8))
plt.scatter(y_test_bc, y_pred_bc, label='Predicted vs Actual')
plt.plot([min(y_test_bc), max(y_test_bc)], [min(y_test_bc), max(y_test_bc)], color='red', linestyle='dashed', label='Ideal Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual Values (Breast Cancer Dataset)')
plt.legend()
plt.savefig('/mnt/data/breast_cancer_predictions.png')
plt.close()
