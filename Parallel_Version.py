import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from mpi4py import MPI
import numpy as np
from sklearn.cluster import KMeans
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load Fashion-MNIST data only on rank 0
if rank == 0:
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    all_images = np.vstack((train_images.reshape(-1, 28 * 28), test_images.reshape(-1, 28 * 28)))

    # Split data for parallel processing
    chunks = np.array_split(all_images, size, axis=0)
else:
    chunks = None

# Scatter data to all processes
data_chunk = comm.scatter(chunks, root=0)

# Standardize features within each chunk
scaler = StandardScaler()
scaled_chunk = scaler.fit_transform(data_chunk)

# K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
start_time = time.time()
kmeans.fit(scaled_chunk)
end_time = time.time()
execution_time_parallel = end_time - start_time

# Gather execution times to rank 0
exec_times = comm.gather(execution_time_parallel, root=0)

if rank == 0:
    total_execution_time_parallel = max(exec_times)
    print("Execution times from all ranks:", exec_times)
    print("Parallel Version (MPI):")
    print(f"Execution Time: {total_execution_time_parallel} seconds")

    # Calculate speedup
    with open('non_parallel_time.txt', 'r') as f:
        execution_time_non_parallel = float(f.read())
    speedup = execution_time_non_parallel / total_execution_time_parallel
    print(f"Speedup: {speedup}")

# Save execution time to file
if rank == 0:
    with open('parallel_time.txt', 'w') as f:
        f.write(str(total_execution_time_parallel))
