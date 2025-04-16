import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# CSV file path
csv_file = 'pi_results.csv' 

def try_read_csv(filename):
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(filename, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding} encoding: {e}")
    
    print(f"Failed to read {filename} with any encoding. Check if file exists and is valid CSV.")
    return None

data = try_read_csv(csv_file)

if data is None:
    print("Could not read data file. Please check file path and format.")
    sys.exit(1)

print("Successfully loaded data:")
print(data.head())


if 'Processes' not in data.columns:
    print(f"Warning: 'Processes' column not found. Available columns: {data.columns.tolist()}")
    for col in data.columns:
        if 'proc' in col.lower():
            print(f"Using '{col}' as processes column")
            data['Processes'] = data[col]
            break
    if 'Processes' not in data.columns and len(data.columns) >= 2:
        print(f"Guessing that column 1 contains processes")
        data['Processes'] = data.iloc[:, 1]

# Extract performance metrics, handling potential naming variations
def get_column_with_fallbacks(data, primary_name, fallbacks):
    if primary_name in data.columns:
        return data[primary_name]
    
    for name in fallbacks:
        if name in data.columns:
            print(f"Using '{name}' instead of '{primary_name}'")
            return data[name]
    
    print(f"Warning: Could not find column {primary_name} or alternatives")
    return None

processes = get_column_with_fallbacks(data, 'Processes', ['processes', 'procs', 'nprocs'])
serial_times = get_column_with_fallbacks(data, 'SerialTime', ['Serial', 'serial_time', 'serialtime'])
parallel_times = get_column_with_fallbacks(data, 'ParallelTime', ['Parallel', 'parallel_time', 'paralleltime'])
speedups = get_column_with_fallbacks(data, 'Speedup', ['speedup', 'speed_up'])
efficiencies = get_column_with_fallbacks(data, 'Efficiency', ['efficiency', 'eff'])

# If we couldn't find some columns,we try to calculate them
if speedups is None and serial_times is not None and parallel_times is not None:
    print("Calculating speedup from times")
    speedups = serial_times / parallel_times

if efficiencies is None and speedups is not None and processes is not None:
    print("Calculating efficiency from speedup and processes")
    efficiencies = speedups / processes

# We convert to numeric to be safe
processes = pd.to_numeric(processes, errors='coerce')
speedups = pd.to_numeric(speedups, errors='coerce')
efficiencies = pd.to_numeric(efficiencies, errors='coerce')
serial_times = pd.to_numeric(serial_times, errors='coerce')
parallel_times = pd.to_numeric(parallel_times, errors='coerce')

# We sort by number of processes
if processes is not None:
    sorted_indices = processes.argsort()
    processes = processes.iloc[sorted_indices]
    if speedups is not None: speedups = speedups.iloc[sorted_indices]
    if efficiencies is not None: efficiencies = efficiencies.iloc[sorted_indices]
    if serial_times is not None: serial_times = serial_times.iloc[sorted_indices]
    if parallel_times is not None: parallel_times = parallel_times.iloc[sorted_indices]

# We create plots
if processes is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Speedup vs Number of Processes
    if speedups is not None:
        ax1.plot(processes, speedups, 'o-', color='blue', linewidth=2, markersize=8)
        ax1.set_title('Speedup vs Number of Processes', fontsize=14)
        ax1.set_xlabel('Number of Processes', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.grid(True)

        # Add line for ideal speedup
        max_procs = processes.max()
        ideal = np.arange(1, max_procs+1)
        ax1.plot(ideal, ideal, '--', color='red', linewidth=1.5, label='Ideal Speedup')
        ax1.legend()

        # Set x-axis ticks to match the actual process counts
        ax1.set_xticks(processes)

    # Plot Efficiency vs Number of Processes
    if efficiencies is not None:
        ax2.plot(processes, efficiencies, 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_title('Efficiency vs Number of Processes', fontsize=14)
        ax2.set_xlabel('Number of Processes', fontsize=12)
        ax2.set_ylabel('Efficiency', fontsize=12)
        ax2.grid(True)

        # Add line for ideal efficiency (always 1.0)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Ideal Efficiency')
        ax2.legend()

        # Set x-axis ticks to match the actual process counts
        ax2.set_xticks(processes)

        # Set y-axis limits for efficiency plot (0 to 1.2)
        max_eff = max(1.2, efficiencies.max() * 1.1)
        ax2.set_ylim(0, max_eff)

    # Add a main title for the whole figure
    fig.suptitle('Pi Calculation Performance', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig('pi_performance_plots.png', dpi=300)
    plt.show()

    # Print a summary table
    print("\nSUMMARY OF RESULTS:")
    print("=" * 80)
    print(f"{'Processes':<10}{'Serial Time (s)':<20}{'Parallel Time (s)':<20}{'Speedup':<15}{'Efficiency':<15}")
    print("-" * 80)
    for i in range(len(processes)):
        p = processes.iloc[i]
        st = serial_times.iloc[i] if serial_times is not None else float('nan')
        pt = parallel_times.iloc[i] if parallel_times is not None else float('nan')
        sp = speedups.iloc[i] if speedups is not None else float('nan')
        ef = efficiencies.iloc[i] if efficiencies is not None else float('nan')
        print(f"{int(p):<10}{st:<20.6f}{pt:<20.6f}{sp:<15.6f}{ef:<15.6f}")
    print("=" * 80)

    # Add a plot for execution times
    if serial_times is not None and parallel_times is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(processes, serial_times, 'o-', color='red', linewidth=2, markersize=8, label='Serial Time')
        plt.plot(processes, parallel_times, 'o-', color='blue', linewidth=2, markersize=8, label='Parallel Time')
        plt.title('Execution Times vs Number of Processes', fontsize=14)
        plt.xlabel('Number of Processes', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.xticks(processes)
        plt.savefig('pi_execution_times.png', dpi=300)
        plt.show()
else:
    print("Could not plot - process data not available")