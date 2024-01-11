import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def read_usage_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip headers
    timestamps = []
    memory_usage = []
    cpu_usage = []
    gpu_usage = []
    phases = []  # To store phase information
    for line in lines:
        data = line.strip().split(", ")
        timestamps.append(data[0])
        memory_usage.append(float(data[1]))
        cpu_usage.append(float(data[2]))
        gpu_usage.append(float(data[3]) if len(data) > 3 else None)
        phases.append(data[4] if len(data) > 4 else 'unknown')  # Add phase or 'unknown'
    return timestamps, memory_usage, cpu_usage, gpu_usage, phases

def plot_usage_graphs(timestamps, memory_usage, cpu_usage, gpu_usage, phases):
    dates = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]

    # Create three separate figures
    fig_memory, ax_memory = plt.subplots(figsize=(10, 5))
    fig_cpu, ax_cpu = plt.subplots(figsize=(10, 5))
    fig_gpu, ax_gpu = plt.subplots(figsize=(10, 5))

    # Define colors for each phase
    phase_colors = {'prey': 'blue', 'predator': 'red', 'initializing': 'green'}

    # Variables to store the start of the current segment
    segment_start_idx = 0
    current_phase = phases[0]

    for i in range(1, len(phases)):
        if phases[i] != current_phase or i == len(phases) - 1:
            # Plot the segment on the corresponding subplot
            ax_memory.plot(dates[segment_start_idx:i+1], memory_usage[segment_start_idx:i+1], marker='o', linestyle='-', color=phase_colors.get(current_phase, 'black'))
            ax_cpu.plot(dates[segment_start_idx:i+1], cpu_usage[segment_start_idx:i+1], marker='o', linestyle='-', color=phase_colors.get(current_phase, 'black'))
            ax_gpu.plot(dates[segment_start_idx:i+1], gpu_usage[segment_start_idx:i+1], marker='o', linestyle='-', color=phase_colors.get(current_phase, 'black'))

            # Start a new segment
            segment_start_idx = i
            current_phase = phases[i]

    # Set labels and legends for each subplot
    ax_memory.set_ylabel('Memory Usage (MB)')
    ax_cpu.set_ylabel('CPU Usage (%)')
    ax_gpu.set_ylabel('GPU Usage (%)')

    # Creating a custom legend
    custom_lines = [plt.Line2D([0], [0], color=color, lw=2) for color in phase_colors.values()]
    fig_memory.legend(custom_lines, phase_colors.keys(), loc='upper right')

    # Set x-axis date format for each subplot
    for ax in [ax_memory, ax_cpu, ax_gpu]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    plt.xticks(rotation=45)
    
    # Save each figure separately
    fig_memory.savefig('memory_usage.png')
    fig_cpu.savefig('cpu_usage.png')
    fig_gpu.savefig('gpu_usage.png')
