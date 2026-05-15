import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Scaling
X = StandardScaler().fit_transform(X)

# Feature Selection
rfe = RFE(SVC(kernel='linear'), n_features_to_select=6)
X = rfe.fit_transform(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# SVM Models
models = {
    "Linear": SVC(kernel='linear'),
    "Poly": SVC(kernel='poly'),
    "RBF": SVC(kernel='rbf')
}

# Accuracy check
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "Accuracy =", accuracy_score(y_test, pred))

# Dendrogram
linked = linkage(X, 'ward')
dendrogram(linked)
plt.show()

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X)

print("Cluster Labels:")
print(labels)
