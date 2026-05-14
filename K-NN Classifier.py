import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
 
data = load_iris()
X, y = data.data, data.target
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)
 
# Find best K
best_k, best_acc = 1, 0
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_scaled, y, cv=5).mean()
    if score > best_acc:
        best_acc, best_k = score, k
 
print(f"Best K = {best_k}, CV Accuracy = {best_acc:.4f}")
 
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
 
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
y_bin = label_binarize(y_test, classes=[0,1,2])
auc = roc_auc_score(y_bin, y_prob, multi_class='ovr')
 
print(f"Test Accuracy: {acc:.4f}")
print(f"AUC (OvR):     {auc:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))
