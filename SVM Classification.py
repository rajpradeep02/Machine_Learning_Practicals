import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
 
data = load_breast_cancer()
X, y = data.data, data.target
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
 
for kernel in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(svm, X_scaled, y, cv=5).mean()
    print(f"Kernel={kernel:6s} | Acc={acc:.4f} | AUC={auc:.4f} | CV={cv:.4f}")
 
# Best model detailed metrics
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nRBF SVM => TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Recall={tp/(tp+fn):.4f}, Specificity={tn/(tn+fp):.4f}")
print("\n", classification_report(y_test, y_pred,
      target_names=['Malignant','Benign']))
