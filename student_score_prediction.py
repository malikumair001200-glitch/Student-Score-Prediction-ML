# Advance Machine Learning Project: Linear Regression Model
# Predict Final Exam Scores based on Weekly Study Hours
# Author: Waqas Manzoor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
# Load data from CSV file
data = pd.read_csv("student_scores.csv")  # Pass your CSV file path here

# Input feature (X) and Target output (Y)
X = data[['study-hours-per-week']]  # 2D Array / DataFrame
Y = data['Final-Exam-Score']        # 1D Series

# ---------------------------------------------------
# 2. Train Model
# ---------------------------------------------------
model = LinearRegression()
model.fit(X, Y)

# Make Predictions
Predicted_scores = model.predict(X)

# ---------------------------------------------------
# 3. Model Evaluation Metrics
# ---------------------------------------------------
mae = mean_absolute_error(Y, Predicted_scores)
mse = mean_squared_error(Y, Predicted_scores)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Predicted_scores)

# Display Results
print("--- Model Performance Metrics ---")
print("Mean Absolute Error (MAE)   :", round(mae, 2))
print("Mean Squared Error (MSE)    :", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R^2 Score (Model Accuracy)  :", round(r2, 4))

# ---------------------------------------------------
# 4. Data Visualization: Histogram
# ---------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(data['Final-Exam-Score'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Final Exam Score")
plt.xlabel("Final Exam Score")
plt.ylabel("Number of Students")
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 5. Data Visualization: Regression Line vs Actual Data
# ---------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Actual Scores')
plt.plot(X, Predicted_scores, color='red', linewidth=2, label='Predicted Scores (Regression Line)')
plt.title("Model Prediction vs Actual Score")
plt.xlabel("Study Hours Per Week")
plt.ylabel("Final Exam Score")
plt.legend()
plt.grid(True)
plt.show()
