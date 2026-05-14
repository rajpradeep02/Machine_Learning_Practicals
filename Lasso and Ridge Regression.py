import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
 
data = fetch_california_housing()
X, y = data.data, data.target
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
 
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)':  Ridge(alpha=1.0),
    'Lasso (alpha=0.1)':  Lasso(alpha=0.1),
}
 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"{name:25s} => MSE: {mse:.4f}, R2: {r2:.4f}")
 
print("\nLasso zero coefficients:", sum(Lasso(alpha=0.1).fit(X_train,y_train).coef_ == 0))
