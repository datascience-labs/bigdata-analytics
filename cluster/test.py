from time import time
import numpy as np
from kmeans import kmeans
from streaming_kmeans import StreamingKMeans

# Sum of Squared Error(SSE)
def sse(centroids, labels, is_streaming=False):
    if is_streaming:
        return np.sum([np.sum(np.linalg.norm(data[np.array(labels) == i] - centroid)**2) for i, centroid in enumerate(centroids.centroids)])
    else:
        return np.sum([np.sum(np.linalg.norm(data[labels == i] - centroid)**2) for i, centroid in enumerate(centroids)])


# Generate synthetic data
np.random.seed(42)
cluster_1 = np.random.randn(500, 2)
cluster_2 = np.random.randn(500, 2) + [5, 5]
data = np.vstack([cluster_1, cluster_2])

# Traditional k-Means
start_time = time()
centroids_kmeans, labels_kmeans = kmeans(data, k=2)
kmeans_time = time() - start_time

kmeans_sse = sse(centroids_kmeans, labels_kmeans)

# Streaming k-Means
streaming_kmeans = StreamingKMeans(k=2)
start_time = time()
for data_point in data:
    streaming_kmeans.update(data_point)
streaming_kmeans_time = time() - start_time
labels_streaming_kmeans = [streaming_kmeans.predict(data_point) for data_point in data]

streaming_kmeans_sse = sse(streaming_kmeans, labels_streaming_kmeans, True)

# kmeans_time, kmeans_sse, streaming_kmeans_time, streaming_kmeans_sse
print(kmeans_time, kmeans_sse, streaming_kmeans_time, streaming_kmeans_sse)