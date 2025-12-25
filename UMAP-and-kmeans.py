import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

df = pd.read_csv("/path/predict-prob-forstrain.csv")

strains = pd.concat([df['virus_1'], df['virus_2']]).unique()
strain_to_index = {strain: index for index, strain in enumerate(strains)}
n_strains = len(strains)
similarity_matrix = np.zeros((n_strains, n_strains))

for index, row in df.iterrows():
    i = strain_to_index[row['virus_1']]
    j = strain_to_index[row['virus_2']]
    similarity = row['prob']
    if i < n_strains and j < n_strains:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

reducer = umap.UMAP(n_components=3, random_state=42)
umap_embedding = reducer.fit_transform(similarity_matrix)

wcss = []
silhouette_scores = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(umap_embedding)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(umap_embedding, kmeans.labels_))

plt.figure(figsize=(10, 6))
plt.plot(range(2, 20), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig("elbow_method.png")
plt.show()

optimal_k = int(input("Enter the number of clusters (K): "))

max_iter = 1000

kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=max_iter)
kmeans.fit(umap_embedding)
clusters = kmeans.labels_

cluster_colors = {
    0: '#FFFF00',
    1: '#FF9900',
    2: '#FF00FF',
    3: '#33CC00',
    4: '#FF0033',
    5: '#66FF00',
    6: '#009999',
    7: '#00FFFF',
    8: '#0033FF',
    9: '#9900FF',
    10: '#FF6666'
}

colors = [cluster_colors[i] if i in cluster_colors else '#000000' for i in clusters]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], umap_embedding[:, 2], c=colors, s=30)
ax.set_title(f'Clustering of Samples Based on Pairwise Relations (K={optimal_k})')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.set_zlabel('UMAP Component 3')

handles, labels = scatter.legend_elements()
ax.legend(handles, labels, title="Clusters")

plt.savefig("plot.png")
plt.show()

with open("/path/cluster_info-all-sample-3D.txt", 'w') as file:
    for strain, cluster_id in zip(strains, clusters):
        file.write(f"{strain}\t{cluster_id}\n")

print(f"Cluster information has been saved to 'cluster_info-all-sample-3D.txt'")
