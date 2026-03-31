from data_cleaning import data_cleaning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import os

path = os.path.join(os.path.dirname(__file__), "store_customers.csv")

#data preprocessing
df = data_cleaning(path)

#dropping gender before clustering - binary feature dominates distance space
df_cluster = df.drop("Gender", axis=1)

#finding optimal eps using k-distance plot (k = min_samples)
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(df_cluster)
distances, _ = neighbors.kneighbors(df_cluster)
distances = np.sort(distances[:, min_samples - 1])

#k-distance plot to find optimal eps (look for the "elbow")
plt.figure(figsize=(8, 5))
plt.plot(distances, color="steelblue")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{min_samples}-NN distance")
plt.title("K-Distance Plot — find elbow to choose eps")
plt.tight_layout()
plt.show()

#creating DBSCAN model with chosen eps and min_samples
model = DBSCAN(eps=0.32, min_samples=min_samples)
df["Cluster"] = model.fit_predict(df_cluster)

#-1 means noise points
n_clusters = len(set(df["Cluster"])) - (1 if -1 in df["Cluster"].values else 0)
n_noise = (df["Cluster"] == -1).sum()

print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {n_noise}")

if n_clusters > 1:
    mask = df["Cluster"] != -1
    sil = silhouette_score(df_cluster[mask], df[mask]["Cluster"])
    print(f"Silhouette Score (without noise): {sil:.3f}")
else:
    sil = None
    print("Silhouette Score: not available (only 1 cluster)")

print(f"\nCluster sizes:\n{df['Cluster'].value_counts().sort_index()}")
cluster_means = df.groupby("Cluster").mean()
print(f"\nCluster means:\n{cluster_means}")

with open(os.path.join(os.path.dirname(__file__), "results.txt"), "a") as f:
    f.write(f"DBSCAN.py: eps=0.32, min_samples={min_samples}, n_clusters={n_clusters}, noise_points={n_noise}")
    f.write(f", Silhouette Score={sil:.3f}\n" if sil else ", Silhouette Score=N/A\n")
    f.write(f"\nCluster sizes:\n{df['Cluster'].value_counts().sort_index().to_string()}\n")
    f.write(f"\nCluster means:\n{cluster_means.to_string()}\n")