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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_usage_graphs(timestamps, memory_usage, cpu_usage, gpu_usage, phases):
    dates = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Define colors for each phase
    phase_colors = {'prey': 'blue', 'predator': 'red', 'initializing': 'green'}

    # Variables to store the start of the current segment
    segment_start_idx = 0
    current_phase = phases[0]

    for i in range(1, len(phases)):
        if phases[i] != current_phase or i == len(phases) - 1:
            # Plot the segment
            end_idx = i if phases[i] != current_phase else i + 1
            axs[0].plot(dates[segment_start_idx:end_idx], memory_usage[segment_start_idx:end_idx], marker='o', linestyle='-', color=phase_colors.get(current_phase, 'black'))
            axs[1].plot(dates[segment_start_idx:end_idx], cpu_usage[segment_start_idx:end_idx], marker='o', linestyle='-', color=phase_colors.get(current_phase, 'black'))
            axs[2].plot(dates[segment_start_idx:end_idx], gpu_usage[segment_start_idx:end_idx], marker='o', linestyle='-', color=phase_colors.get(current_phase, 'black'))

            # Start a new segment
            segment_start_idx = i
            current_phase = phases[i]

    # Set labels and legends
    axs[0].set_ylabel('Memory Usage (MB)')
    axs[1].set_ylabel('CPU Usage (%)')
    axs[2].set_ylabel('GPU Usage (%)')

    # Creating a custom legend
    custom_lines = [plt.Line2D([0], [0], color=color, lw=2) for color in phase_colors.values()]
    fig.legend(custom_lines, phase_colors.keys(), loc='upper right')

    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Call the functions
timestamps, memory_usage, cpu_usage, gpu_usage, phases = read_usage_log('memory_usage_log.txt')
plot_usage_graphs(timestamps, memory_usage, cpu_usage, gpu_usage, phases)
