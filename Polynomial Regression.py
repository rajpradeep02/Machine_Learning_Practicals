import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
 
# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = 0.5*X.ravel()**3 - X.ravel()**2 + 2*X.ravel() + np.random.randn(200)*2
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
 
for degree in [1, 2, 3, 4]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"Degree {degree} => MSE: {mse:.4f}, R2: {r2:.4f}")
