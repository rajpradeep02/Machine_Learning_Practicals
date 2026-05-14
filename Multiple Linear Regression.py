import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 
data = fetch_california_housing()
X, y = data.data, data.target
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
 
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
 
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
cv  = cross_val_score(model, X, y, cv=5, scoring='r2')
 
print("Coefficients:", np.round(model.coef_, 4))
print("Intercept:", round(model.intercept_, 4))
print("MSE:", round(mse, 4))
print("R2 Score:", round(r2, 4))
print("RMSE:", round(np.sqrt(mse), 4))
print("5-Fold CV R2: %.4f +/- %.4f" % (cv.mean(), cv.std()))
for name, coef in zip(data.feature_names, model.coef_):
    print(f"  {name:12s}: {coef:.4f}")
