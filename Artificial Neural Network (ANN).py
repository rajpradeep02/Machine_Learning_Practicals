# ===========================
# Q1. K-Means Clustering
# ===========================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate synthetic dataset
X, y = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.80,
    random_state=42
)

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Choose optimal K
optimal_k = 4

# Train KMeans with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='viridis')

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:,0],
    centers[:,1],
    c='red',
    s=300,
    marker='X',
    label='Centroids'
)

plt.title("K-Means Clustering")
plt.legend()
plt.show()

# Final Inertia
print("Final Inertia :", kmeans.inertia_)

# Silhouette Score
sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score :", sil_score)
