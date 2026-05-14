import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
 
data = load_iris()
X, y_true = data.data, data.target
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Compare linkage methods
print("=== Linkage Method Comparison (k=3) ===")
for linkage_type in ['ward', 'complete', 'average', 'single']:
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage_type)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(y_true, labels)
    print(f"  {linkage_type:8s} | Silhouette={sil:.4f} | ARI={ari:.4f}")
 
# Best model
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X_scaled)
 
print("\n=== Ward Linkage Final Results ===")
print(f"Silhouette Score:    {silhouette_score(X_scaled, labels):.4f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, labels):.4f}")
print("Cluster distribution:", np.bincount(labels))
 
# Cophenetic correlation for dendrogram quality
from scipy.cluster.hierarchy import cophenet
Z = linkage(X_scaled[:50], method='ward')  # subset for speed
c, _ = cophenet(Z, pdist(X_scaled[:50]))
print(f"Cophenetic Correlation: {c:.4f}  (closer to 1 = better dendrogram)")
