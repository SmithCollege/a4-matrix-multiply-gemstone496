# Import seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Apply default theme
sns.set_theme()

# Load data
PATH = "data/"
cpuTime = pd.read_csv(PATH + "cpu.csv", header=0, index_col=0)
gpuTime = pd.read_csv(PATH + "gpu.csv", header=0, index_col=0)
tiledTime = pd.read_csv(PATH + "tiled.csv", header=0, index_col=0)
cublasTime = pd.read_csv(PATH + "cublas.csv", header=0, index_col=0)

# Transpose
cpuTime = cpuTime.transpose().rename(columns={4096: 'cpu4096', 16384: 'cpu16384', 65536: 'cpu65536', 262144: 'cpu262144', 1048576: 'cpu1048576'})
gpuTime = gpuTime.transpose().rename(columns={4096: 'gpu4096', 16384: 'gpu16384', 65536: 'gpu65536', 262144: 'gpu262144', 1048576: 'gpu1048576'})
tiledTime = tiledTime.transpose().rename(columns={4096: 'tiled4096', 16384: 'tiled16384', 65536: 'tiled65536', 262144: 'tiled262144', 1048576: 'tiled1048576'})
cublasTime = cublasTime.transpose().rename(columns={4096: 'cublas4096', 16384: 'cublas16384', 65536: 'cublas65536', 262144: 'cublas262144', 1048576: 'cublas1048576'})

# Join
times = cpuTime.join(gpuTime, lsuffix='cpu', rsuffix='gpu').join(tiledTime, rsuffix='tiled').join(cublasTime, rsuffix='cublas')
times['run'] = times.index # turn this back into a proper row

print(times)

# Pull out size through a wide-long pivot
times = pd.wide_to_long(times, ['cpu', 'gpu', 'tiled', 'cublas'], i='run', j='size')
times = times.reset_index() # turn back to a proper row

print(times)

# Pull out method through a melting pivot
times  = pd.melt(times, id_vars=['run', 'size'], var_name='method', value_name='time').reset_index()

print(times)

# Create graph
plot = sns.scatterplot(data=times, x='size', y='time', hue='method')
plt.xscale('log')
plt.yscale('log')

# Export graph
plt.savefig("timers.png")
