from data_cleaning import data_cleaning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

path = os.path.join(os.path.dirname(__file__), "store_customers.csv")

#data preprocessing
df = data_cleaning(path)

#finding optimal number of clusters using elbow method
inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(df)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(df, model.labels_))

#elbow plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker="o", color="steelblue")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, marker="o", color="tomato")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")

plt.tight_layout()
plt.show()

#creating final kmeans model with chosen k
optimal_k = silhouettes.index(max(silhouettes)) + 2
print(f"Optimal k (best silhouette): {optimal_k}")

model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(df)

sil = silhouette_score(df.drop("Cluster", axis=1), df["Cluster"])
cluster_sizes = df["Cluster"].value_counts().sort_index()
cluster_means = df.groupby("Cluster").mean()

print(f"Silhouette Score: {sil:.3f}")
print(f"\nCluster sizes:\n{cluster_sizes}")
print(cluster_means)

with open(os.path.join(os.path.dirname(__file__), "results.txt"), "a") as f:
    f.write(f"KMeans.py: optimal_k={optimal_k}, Silhouette Score={sil:.3f}\n")
    f.write(f"\nCluster sizes:\n{cluster_sizes.to_string()}\n")
    f.write(f"\nCluster means:\n{cluster_means.to_string()}\n")
