import matplotlib.pyplot as plt

# Read execution times from txt files
with open('non_parallel_time.txt', 'r') as f:
    non_parallel_time = float(f.read())

with open('parallel_time.txt', 'r') as f:
    parallel_time = float(f.read())

# Calculate speedup
speedup = non_parallel_time / parallel_time

# Data for plotting
labels = ['Non-Parallel', 'Parallel']
times = [non_parallel_time, parallel_time]

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(labels, times, color=['blue', 'orange'])
plt.xlabel('Execution Type')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Execution Times')
plt.text(1, parallel_time, f'Speedup: {speedup:.2f}', ha='center', va='bottom', color='black')
plt.ylim(0, max(times) * 1.1)

# Show the plot
plt.show()
