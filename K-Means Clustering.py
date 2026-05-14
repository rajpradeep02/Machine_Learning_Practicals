import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             davies_bouldin_score)
 
data = load_iris()
X, y_true = data.data, data.target
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Elbow Method
print("=== Elbow Method (Inertia) ===")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    print(f"  k={k} | Inertia={km.inertia_:.2f} | "
          f"Silhouette={silhouette_score(X_scaled, km.labels_):.4f}")
 
# Final model with k=3
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
 
sil = silhouette_score(X_scaled, labels)
ari = adjusted_rand_score(y_true, labels)
db  = davies_bouldin_score(X_scaled, labels)
 
print("\n=== K=3 Final Results ===")
print(f"Silhouette Score:      {sil:.4f}  (1=best)")
print(f"Adjusted Rand Index:   {ari:.4f}  (1=perfect match)")
print(f"Davies-Bouldin Index:  {db:.4f}   (lower=better)")
print(f"Cluster Centers Shape: {km.cluster_centers_.shape}")
print("Cluster Sizes:", np.bincount(labels))
