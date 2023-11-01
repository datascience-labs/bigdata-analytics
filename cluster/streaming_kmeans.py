import numpy as np

class StreamingKMeans:
    def __init__(self, k, alpha=0.1):
        self.k = k
        self.centroids = None
        self.alpha = alpha  # Learning rate

    def _initialize(self, data_point):
        """Initialize centroids if not already initialized."""
        if self.centroids is None:
            self.centroids = np.random.randn(self.k, data_point.shape[0])

    def update(self, data_point):
        """Update centroids with a new data point."""
        self._initialize(data_point)

        # Find the nearest centroid
        distances = np.linalg.norm(self.centroids - data_point, axis=1)
        closest_idx = np.argmin(distances)

        # Update the closest centroid
        self.centroids[closest_idx] = (1 - self.alpha) * self.centroids[closest_idx] + self.alpha * data_point

    def predict(self, data_point):
        """Predict the closest cluster index for a given data point."""
        self._initialize(data_point)
        distances = np.linalg.norm(self.centroids - data_point, axis=1)
        return np.argmin(distances)

# Test
streaming_kmeans = StreamingKMeans(k=2)
test_data = [[1, 2], [5, 6], [1, 1], [5, 5], [1, 3], [5, 7]]

# Simulate streaming data and update centroids
for data_point in test_data:
    streaming_kmeans.update(np.array(data_point))

# Predict clusters for the test data
labels = [streaming_kmeans.predict(np.array(data_point)) for data_point in test_data]

streaming_kmeans.centroids, labels