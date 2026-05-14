import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
 
data = load_iris()
X, y = data.data, data.target
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
 
model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
 
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
y_bin = label_binarize(y_test, classes=[0,1,2])
auc = roc_auc_score(y_bin, y_prob, multi_class='ovr')
cv  = cross_val_score(model, X, y, cv=5)
 
print(f"Accuracy:     {acc:.4f}")
print(f"AUC:          {auc:.4f}")
print(f"5-Fold CV:    {cv.mean():.4f} +/- {cv.std():.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))
print("\nTree Structure (first 3 levels):")
print(export_text(model, feature_names=data.feature_names, max_depth=3))
