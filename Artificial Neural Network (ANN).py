from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
 
data = load_breast_cancer()
X, y = data.data, data.target
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
 
# ANN with 2 hidden layers
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
 
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
auc = roc_auc_score(y_test, y_prob)
cv  = cross_val_score(model, X_scaled, y, cv=5)
 
print(f"Accuracy:    {acc:.4f}")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"AUC:         {auc:.4f}")
print(f"5-Fold CV:   {cv.mean():.4f} +/- {cv.std():.4f}")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=['Malignant','Benign']))
