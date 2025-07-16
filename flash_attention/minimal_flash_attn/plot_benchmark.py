import matplotlib.pyplot as plt
import numpy as np

# Your data from the table
seq_lens = [64, 128, 256, 512, 1280, 2560]
manual_attention_times = [0.083, 0.292, 1.005, 3.681, 21.856, 89.0]
flash_attention_times = [1.194, 8.895, 32.739, 127.939, 810.178, 3246.74]
flash_attention_v2_times = [1.703, 6.562, 25.782, 102.313, 636.632, 2543.349]

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figure size for better readability

bar_width = 0.2 # Width of each bar
# The x-axis positions for the groups of bars
index = np.arange(len(seq_lens))

# Create bars for each method
# Manual Attention bars will be slightly to the left of the tick mark
ax.bar(index - bar_width, manual_attention_times, bar_width, label='Manual Attention', color='skyblue')
# Flash Attention bars will be centered on the tick mark
ax.bar(index, flash_attention_times, bar_width, label='Flash Attention', color='lightcoral')
# Flash Attention V2 bars will be slightly to the right of the tick mark
ax.bar(index + bar_width, flash_attention_v2_times, bar_width, label='Flash Attention V2', color='lightgreen')

# Add labels and title
ax.set_xlabel('Sequence Length (seq_len)', fontsize=12)
ax.set_ylabel('Execution Time (ms)', fontsize=12)
ax.set_title('Attention Method Performance vs. Sequence Length', fontsize=14)

# Set x-axis ticks and labels
ax.set_xticks(index)
ax.set_xticklabels(seq_lens)

# Add a legend for clarity
ax.legend(fontsize=10)

# Add a grid for better readability, only on the y-axis
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure layout is tight to prevent labels from overlapping
plt.tight_layout()

# Save the plot to a JPG file
plot_filename = 'time_compare.jpg'
plt.savefig(plot_filename, dpi=300) # dpi for higher resolution
print(f"Plot saved to {plot_filename}")

# Optionally display the plot
plt.show()