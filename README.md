# PAForASD-Project
Course Project for Parallel Algorithms for the Analysis and Synthesis of Data

# K-Means Clustering with MPI Parallelization

# 1. Algorithm and Parallelization Method
This project implements K-Means clustering on the Fashion-MNIST dataset using Python. The parallelization method used is MPI (Message Passing Interface) to distribute the data processing across multiple processes.

# 2. Instructions to Reproduce Results and Start the Algorithm

# Requirements:

Python 3.9
Required Python packages:
tensorflow
numpy
scikit-learn
mpi4py

# Running the algorithm:

Run the non-parallel version to obtain the baseline execution time:

python Non-Parallel_Version.py

Run the parallel version using MPI to observe the speedup:

mpiexec -np 5 python Parallel_Version.py (-np "n" n=number of processes)

# Data Placement:
The script automatically downloads and preprocesses the Fashion-MNIST dataset. No need to manually place any data files.

# 3. Short Explanation of Parallelized Algorithm Part
The part of the algorithm that was parallelized involves the data preprocessing and the K-Means clustering steps. The dataset is divided into chunks and distributed across multiple processes. Each process performs standardization and clustering on its respective chunk of data. Finally, execution times are gathered and combined to calculate the overall performance and speedup.

# 4. Speedup Calculation
The speedup is calculated as the ratio of the non-parallel execution time to the parallel execution time. Here are the results(In my case):

Non-Parallel Time: 4.015321969985962 seconds

Parallel Version (MPI): 2.699493885040283 seconds

Speedup: 1.4874351048682013
