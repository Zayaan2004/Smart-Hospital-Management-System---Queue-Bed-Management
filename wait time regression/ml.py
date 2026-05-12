import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# =========================
# 1. LOAD DATA & TRAIN MODEL
# =========================

# Load dataset
df = pd.read_csv("data.csv")

# Feature columns and target
X = df[["queue_length", "avg_service_time", "priority_level"]]
y = df["wait_time"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=70
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MODEL TRAINED SUCCESSFULLY =====\n")
print("Coefficients:")
print(f"queue_length:     {model.coef_[0]:.4f}")
print(f"avg_service_time: {model.coef_[1]:.4f}")
print(f"priority_level:   {model.coef_[2]:.4f}")
print(f"Intercept:        {model.intercept_:.4f}")

print("\n===== MODEL ACCURACY =====")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
print("\n======================================")

# Optional: keep these if you want to copy into React later
A = model.coef_[0]
B = model.coef_[1]
C = model.coef_[2]
D = model.intercept_

# =========================
# 2. PREDICT & APPEND NEW ROW
# =========================

def predict_wait_time_ml(queue_length, avg_service_time, priority_level):
    """
    Uses the trained sklearn model to predict wait time.
    """
    features = [[queue_length, avg_service_time, priority_level]]
    wait = model.predict(features)[0]
    return max(0, round(wait))

print("\nEnter new patient details:")
queue_length = float(input("Current queue length: "))
avg_service_time = float(input("Avg service time (mins): "))
priority_level = float(input("Priority level (0 = low, 3 = high): "))

predicted_wait = predict_wait_time_ml(queue_length, avg_service_time, priority_level)
print(f"\nPredicted Wait Time: {predicted_wait} minutes")

new_row = {
    "queue_length": queue_length,
    "avg_service_time": avg_service_time,
    "priority_level": priority_level,
    "wait_time": predicted_wait  # storing predicted wait time
}

# Append new data correctly using concat
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

try:
    df.to_csv("data.csv", index=False)
    print("\nNew data point saved to data.csv!")
except PermissionError:
    print("\nERROR: Could not write to data.csv. Please close the file if it's open and try again.")

print("\n============== DONE ==================\n")
print("Note: Coefficients for React integration:")
print(f"A (queue_length):     {A:.4f}")
print(f"B (avg_service_time): {B:.4f}")
print(f"C (priority_level):   {C:.4f}")
print(f"D (intercept):        {D:.4f}")
