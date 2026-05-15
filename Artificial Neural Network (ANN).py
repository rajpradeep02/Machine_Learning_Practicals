import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
experience = np.array([
    1.1, 1.3, 1.5, 2.0, 2.2,
    2.9, 3.0, 3.2, 3.7, 4.0,
    4.5, 5.1, 5.9, 6.8, 7.9,
    8.2, 9.0, 9.6, 10.3, 10.5
]).reshape(-1,1)

salary = np.array([
    3.9, 4.6, 4.8, 5.6, 6.0,
    6.5, 6.8, 7.4, 7.9, 8.2,
    9.3, 10.4, 11.5, 12.6, 14.2,
    15.0, 16.5, 17.3, 18.4, 18.9
])

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    experience,
    salary,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Slope and Intercept
print("Slope :", model.coef_[0])
print("Intercept :", model.intercept_)

# Prediction for 6.5 years experience
pred_salary = model.predict([[6.5]])

print("Predicted Salary for 6.5 years experience :",
      pred_salary[0], "Lakhs")

# Predictions on test set
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# Plot
plt.scatter(experience, salary, color='blue')

plt.plot(
    experience,
    model.predict(experience),
    color='red'
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary (Lakhs)")
plt.title("Simple Linear Regression")

plt.show()
