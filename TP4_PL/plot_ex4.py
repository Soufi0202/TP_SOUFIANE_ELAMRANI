import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('results.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Speedup vs Number of Processes
ax1.plot(data['Processes'], data['Speedup'], 'o-', color='blue', linewidth=2, markersize=8)
ax1.set_title('Speedup vs Number of Processes', fontsize=14)
ax1.set_xlabel('Number of Processes', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.grid(True)

# Add line for ideal speedup
max_procs = data['Processes'].max()
ideal = np.arange(1, max_procs+1)
ax1.plot(ideal, ideal, '--', color='red', linewidth=1.5, label='Ideal Speedup')
ax1.legend()

# Plot Efficiency vs Number of Processes
ax2.plot(data['Processes'], data['Efficiency'], 'o-', color='green', linewidth=2, markersize=8)
ax2.set_title('Efficiency vs Number of Processes', fontsize=14)
ax2.set_xlabel('Number of Processes', fontsize=12)
ax2.set_ylabel('Efficiency', fontsize=12)
ax2.grid(True)

# Add line for ideal efficiency (always 1.0)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Ideal Efficiency')
ax2.legend()

# Set y-axis limits for efficiency plot (0 to 1.2)
ax2.set_ylim(0, 1.2)

# Add a main title for the whole figure
fig.suptitle(f'Matrix-Vector Multiplication Performance (Matrix Size: {data["Size"][0]})', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
plt.savefig('mpi_performance_plots.png', dpi=300)
plt.show()

# Print a summary table
print("\nSUMMARY OF RESULTS:")
print("=" * 80)
print(f"{'Processes':<10}{'Serial Time (s)':<20}{'Parallel Time (s)':<20}{'Speedup':<15}{'Efficiency':<15}")
print("-" * 80)
for _, row in data.iterrows():
    print(f"{int(row['Processes']):<10}{row['SerialTime']:<20.6f}{row['ParallelTime']:<20.6f}{row['Speedup']:<15.6f}{row['Efficiency']:<15.6f}")
print("=" * 80)