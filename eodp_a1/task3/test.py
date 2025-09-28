import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def compute_sse(data_points, labels, centroids):
    """
    Compute Sum of Squared Errors (SSE) - sum of squared distances of objects from their cluster centroids.
    
    Args:
        data_points: array of data points
        labels: cluster assignments for each data point
        centroids: cluster centroids
    
    Returns:
        SSE value
    """
    sse = 0
    for i, point in enumerate(data_points):
        cluster_id = labels[i]
        centroid = centroids[cluster_id]
        sse += np.sum((point - centroid) ** 2)
    return sse

# usually we don't specify the random points as sklearn has a built-in method of setting fixed random states
initial_clusters = np.array([[1,1], [2,1]])
data_points = np.array([[1,1], [2,1], [4,3], [5,4]])

kmean = KMeans(n_clusters=2, init=initial_clusters)
kmean.fit(data_points)

# Create 2D plot
plt.figure(figsize=(8, 6))

# Plot data points colored by cluster
colors = ['red', 'blue']
for i in range(2):
    cluster_points = data_points[kmean.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], label=f'Cluster {i}', s=100, alpha=0.7)

# Plot centroids
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], 
            c='black', marker='x', s=200, linewidths=3, label='Centroids')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('2D K-means Clustering')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print('kmean assignments [A, B, C, D]:', kmean.labels_)
print('Final Centroid for cluster 0:')
print(kmean.cluster_centers_[0])
print('Final Centroid for cluster 1:')
print(kmean.cluster_centers_[1])

# Compute and display SSE
sse = compute_sse(data_points, kmean.labels_, kmean.cluster_centers_)
print(f'\nSum of Squared Errors (SSE): {sse:.4f}')

# Verify using sklearn's inertia_ attribute
print(f'sklearn inertia_ (should match SSE): {kmean.inertia_:.4f}')