import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import time

# Load Fashion-MNIST data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Preprocess data
train_images = train_images.reshape(-1, 28*28)
test_images = test_images.reshape(-1, 28*28)
all_images = np.vstack((train_images, test_images))

print(all_images.shape)

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_images)

print(scaled_features.shape)

# K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)

start_time = time.time()
kmeans.fit(scaled_features)
end_time = time.time()
execution_time_non_parallel = end_time - start_time

print("K-Means Clustering (Non-Parallel):")
print(f"Execution Time: {execution_time_non_parallel} seconds")

# Save execution time to file
with open('non_parallel_time.txt', 'w') as f:
    f.write(str(execution_time_non_parallel))

cluster_labels = kmeans.labels_
print(cluster_labels[:10])
