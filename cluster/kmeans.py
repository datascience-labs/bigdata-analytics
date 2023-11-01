import numpy as np

def kmeans(data, k, max_iters=100, tol=1e-4):
    # 1. Randomly initialize the centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for i in range(max_iters):
        # 2. Assign each data point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Update centroids to be the mean of the assigned data points
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels

# Test
data = np.array([[1, 2], [5, 6], [1, 1], [5, 5], [1, 3], [5, 7]])
centroids, labels = kmeans(data, k=2)
centroids, labels