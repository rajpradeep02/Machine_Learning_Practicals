import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
 

data = load_iris()
X, y = data.data, data.target
 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
 

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
 

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)
 

y_bin = label_binarize(y_test, classes=[0,1,2])
y_prob = model.predict_proba(X_test)
auc = roc_auc_score(y_bin, y_prob, multi_class='ovr')
 
print("Accuracy:", round(acc, 4))
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred,
      target_names=data.target_names))
print("5-Fold CV Accuracy: %.4f +/- %.4f" % (cv_scores.mean(), cv_scores.std()))
print("AUC Score:", round(auc, 4))
