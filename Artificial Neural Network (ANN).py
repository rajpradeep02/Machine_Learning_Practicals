import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Wine Dataset
wine = datasets.load_wine()

X = wine.data
y = wine.target
feature_names = wine.feature_names

# ===========================
# (xix) Apply Standard Scaling
# ===========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# (xx) Feature Reduction using RFE
# ===========================

svc_linear = SVC(kernel='linear')

rfe = RFE(estimator=svc_linear, n_features_to_select=6)
X_rfe = rfe.fit_transform(X_scaled, y)

selected_features = np.array(feature_names)[rfe.support_]

print("Top 6 Features:")
for f in selected_features:
    print(f)

# ===========================
# (xxi) Train-Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X_rfe,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ===========================
# (xxii) Train SVM Models
# ===========================

models = {
    "Linear": SVC(kernel='linear', probability=True),
    "Polynomial": SVC(kernel='poly', degree=3, probability=True),
    "RBF": SVC(kernel='rbf', probability=True)
}

# ===========================
# (xxiii) Tune RBF Parameters
# ===========================

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

print("\nBest Parameters for RBF:")
print(grid.best_params_)

# Replace RBF model with best estimator
models["RBF"] = grid.best_estimator_

# ===========================
# (xxiv) Evaluation
# ===========================

best_kernel = ""
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred)

    # One-vs-Rest AUC
    y_prob = model.predict_proba(X_test)

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class='ovr'
    )

    print("\n======================")
    print("Kernel :", name)
    print("======================")

    print("Accuracy :", acc)
    print("Macro F1 Score :", f1)
    print("Confusion Matrix :")
    print(cm)
    print("OVR AUC :", auc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_kernel = name

print("\nBest Performing Kernel :", best_kernel)

# ===========================
# (xxv) Agglomerative Clustering
# ===========================

# Dendrogram
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Agglomerative Clustering
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

clusters = agg.fit_predict(X_scaled)

print("\nAgglomerative Clustering Labels:")
print(clusters)
