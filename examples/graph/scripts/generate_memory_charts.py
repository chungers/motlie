#!/usr/bin/env python3
"""
Generate memory usage trend charts for graph algorithm examples.
Compares motlie_db vs in-memory implementations across different scales.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

# Memory data collected from test runs (in KB)
# Format: {algorithm: {scale: (reference_kb, motlie_kb)}}

memory_data = {
    'DFS': {
        1: (144, 144),
        10: (16, 288),
        100: (16, 240),
        1000: (304, 592),
        # 10000: DFS failed correctness check at this scale
    },
    'BFS': {
        1: (144, 144),
        10: (112, 240),
        100: (32, 240),
        1000: (224, 736),
        10000: (3686, 3866),  # 3.61 MB, 3.78 MB
        100000: (33741, 69999),  # 32.95 MB, 68.36 MB
    },
    'Topological Sort': {
        1: (144, 144),
        10: (16, 288),
        100: (32, 288),
        1000: (368, 2324),
        10000: (1301, 4485),
        # 100000: Still running at time of chart generation
    },
    'Dijkstra': {
        1: (144, 144),
        10: (32, 288),
        100: (64, 288),
        1000: (1280, 1055),  # 1.25 MB, 1.03 MB - motlie_db WINS!
        10000: (10895, 6881),  # 10.64 MB, 6.72 MB - motlie_db WINS!
        # 100000: (0, 118497),  # pathfinding: 0 bytes (cached), motlie_db: 115.72 MB
    },
    'PageRank': {
        1: (144, 144),
        10: (240, 240),
        100: (368, 272),
        1000: (3891, 3983),  # 3.80 MB, 3.89 MB
        10000: (32051, 13946),  # 31.30 MB, 13.62 MB - motlie_db WINS!
        100000: (347504, 232070),  # 339.36 MB, 226.62 MB - motlie_db WINS BIGGER!
    },
}

scales = [1, 10, 100, 1000, 10000, 100000]

# Output directory (relative to this script's location)
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '..')

# Create individual charts for each algorithm
for algo_name, data in memory_data.items():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    ref_mem = []
    motlie_mem = []
    valid_scales = []

    for scale in scales:
        if scale in data and data[scale][0] is not None:
            ref_mem.append(data[scale][0])
            motlie_mem.append(data[scale][1])
            valid_scales.append(scale)

    # Plot
    ax.plot(valid_scales, ref_mem, marker='o', linewidth=2.5, markersize=10,
            label='In-Memory (petgraph/pathfinding)', color='#2E86AB', alpha=0.8)
    ax.plot(valid_scales, motlie_mem, marker='s', linewidth=2.5, markersize=10,
            label='motlie_db (persistent)', color='#A23B72', alpha=0.8)

    ax.set_xlabel('Scale Factor', fontsize=13, fontweight='bold')
    ax.set_ylabel('Memory Usage (KB)', fontsize=13, fontweight='bold')
    ax.set_title(f'{algo_name}: Memory Usage vs Scale', fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')

    # Add value labels
    for i, scale in enumerate(valid_scales):
        # Format memory values
        if ref_mem[i] >= 1024:
            ref_label = f'{ref_mem[i]/1024:.2f} MB'
        else:
            ref_label = f'{ref_mem[i]} KB'

        if motlie_mem[i] >= 1024:
            motlie_label = f'{motlie_mem[i]/1024:.2f} MB'
        else:
            motlie_label = f'{motlie_mem[i]} KB'

        ax.annotate(ref_label,
                   xy=(scale, ref_mem[i]),
                   xytext=(8, 8),
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#2E86AB'))
        ax.annotate(motlie_label,
                   xy=(scale, motlie_mem[i]),
                   xytext=(8, -18),
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#A23B72'))

    plt.tight_layout()
    filename = f'{output_dir}/memory_trend_{algo_name.lower().replace(" ", "_").replace("'", "")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Generated: {filename}')
    plt.close()

# Create a combined chart showing all algorithms at scale=100
fig, ax = plt.subplots(figsize=(12, 8))

algorithms = ['DFS', 'BFS', 'Topological Sort', 'Dijkstra', 'PageRank']
ref_100 = []
motlie_100 = []
valid_algos = []

for algo in algorithms:
    if 100 in memory_data[algo] and memory_data[algo][100][0] is not None:
        ref_100.append(memory_data[algo][100][0])
        motlie_100.append(memory_data[algo][100][1])
        valid_algos.append(algo)

x = np.arange(len(valid_algos))
width = 0.35

bars1 = ax.bar(x - width/2, ref_100, width, label='In-Memory',
               alpha=0.8, color='#2E86AB', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, motlie_100, width, label='motlie_db',
               alpha=0.8, color='#A23B72', edgecolor='black', linewidth=1.2)

ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
ax.set_ylabel('Memory Usage (KB)', fontsize=13, fontweight='bold')
ax.set_title('Memory Usage Comparison at Scale=100', fontsize=15, fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(valid_algos, rotation=15, ha='right', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)} KB',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
filename = f'{output_dir}/memory_comparison_scale100.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f'Generated: {filename}')
plt.close()

# Create convergence/ratio trend chart
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#E63946', '#F77F00', '#06A77D', '#118AB2', '#073B4C']
markers = ['o', 's', '^', 'D', 'v']

for idx, (algo_name, data) in enumerate(memory_data.items()):
    ratios = []
    valid_scales_ratio = []

    for scale in scales:
        if scale in data and data[scale][0] is not None and data[scale][0] > 0:
            ratio = data[scale][1] / data[scale][0]
            ratios.append(ratio)
            valid_scales_ratio.append(scale)

    if len(ratios) > 0:
        ax.plot(valid_scales_ratio, ratios, marker=markers[idx], linewidth=2.5,
                markersize=10, label=algo_name, color=colors[idx], alpha=0.8)

        # Add annotations for key points
        for i, (scale, ratio) in enumerate(zip(valid_scales_ratio, ratios)):
            if scale in [100, 1000]:  # Annotate scale 100 and 1000
                ax.annotate(f'{ratio:.2f}x',
                           xy=(scale, ratio),
                           xytext=(10, 5),
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.7, edgecolor=colors[idx]))

# Add reference line at ratio=1.0
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal memory (ratio=1.0)', alpha=0.7)

ax.set_xlabel('Scale Factor', fontsize=13, fontweight='bold')
ax.set_ylabel('Memory Ratio (motlie_db / in-memory)', fontsize=13, fontweight='bold')
ax.set_title('Memory Convergence Trend: motlie_db vs In-Memory', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best')

# Set y-axis limits to better show the convergence
ax.set_ylim([0.5, 20])

plt.tight_layout()
filename = f'{output_dir}/memory_ratio_trend.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f'Generated: {filename}')
plt.close()

# Create a detailed comparison for DFS, BFS, PageRank at scale 1000
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

comparison_algos = ['DFS', 'BFS', 'PageRank']
for idx, algo in enumerate(comparison_algos):
    ax = axes[idx]

    if 1000 in memory_data[algo]:
        ref_val = memory_data[algo][1000][0]
        motlie_val = memory_data[algo][1000][1]

        # Convert to MB if needed
        if ref_val >= 1024 or motlie_val >= 1024:
            ref_val /= 1024
            motlie_val /= 1024
            unit = 'MB'
        else:
            unit = 'KB'

        bars = ax.bar(['In-Memory', 'motlie_db'], [ref_val, motlie_val],
                     color=['#2E86AB', '#A23B72'], alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        ax.set_ylabel(f'Memory Usage ({unit})', fontsize=11, fontweight='bold')
        ax.set_title(f'{algo} at Scale=1000', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f} {unit}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')

        # Add ratio annotation
        ratio = memory_data[algo][1000][1] / memory_data[algo][1000][0]
        ax.text(0.5, 0.95, f'Ratio: {ratio:.2f}x',
               transform=ax.transAxes,
               ha='center', va='top',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
filename = f'{output_dir}/memory_comparison_scale1000.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f'Generated: {filename}')
plt.close()

print('\n✅ All charts generated successfully!')
print(f'\nGenerated files in {output_dir}:')
print('  - memory_trend_dfs.png')
print('  - memory_trend_bfs.png')
print('  - memory_trend_topological_sort.png')
print('  - memory_trend_dijkstra.png')
print('  - memory_trend_pagerank.png')
print('  - memory_comparison_scale100.png')
print('  - memory_comparison_scale1000.png')
print('  - memory_ratio_trend.png')
