"""
Figure 2: Scaling Analysis
Line chart showing TSR (%) vs context length (log scale) for all 6 methods.
Shaded error bands (+/-1 std). ContextOS line is thicker with a distinct marker.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
methods = ['Full Context', 'Truncation', 'RAG-Only', 'MemGPT', 'RAPTOR', 'ContextOS']
context_lengths = np.array([512, 2048, 8192, 32768])
context_labels = ['512', '2K', '8K', '32K']

tsr = np.array([
    [82.3, 71.4, 54.2, 31.5],   # Full Context
    [81.1, 65.8, 42.1, 20.3],   # Truncation
    [79.5, 74.2, 68.3, 52.4],   # RAG-Only
    [80.1, 75.8, 70.2, 58.9],   # MemGPT
    [80.8, 73.9, 69.1, 57.2],   # RAPTOR
    [84.2, 80.5, 77.8, 71.3],   # ContextOS
])

std = np.array([
    [2.1, 2.8, 3.2, 4.1],
    [2.0, 3.1, 3.8, 4.5],
    [2.3, 2.5, 2.9, 3.6],
    [2.2, 2.4, 2.7, 3.2],
    [2.1, 2.6, 2.8, 3.4],
    [1.8, 2.0, 2.2, 2.5],
])

colors = ['#4472C4', '#ED7D31', '#A9D18E', '#FFC000', '#70AD47', '#FF0000']
markers = ['o', 's', '^', 'D', 'v', '*']
line_widths = [1.5, 1.5, 1.5, 1.5, 1.5, 3.0]   # ContextOS thicker
marker_sizes = [6, 6, 6, 6, 6, 10]               # ContextOS larger marker

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

x_log = np.log2(context_lengths)   # evenly spaced on log2 axis

for i, method in enumerate(methods):
    mean_vals = tsr[i]
    err_vals = std[i]
    col = colors[i]
    lw = line_widths[i]
    ms = marker_sizes[i]
    mk = markers[i]
    zord = 5 if method == 'ContextOS' else 3

    # Line + markers
    ax.plot(x_log, mean_vals,
            color=col,
            linewidth=lw,
            marker=mk,
            markersize=ms,
            markerfacecolor=col,
            markeredgecolor='white' if method == 'ContextOS' else col,
            markeredgewidth=0.8,
            label=method,
            zorder=zord)

    # Shaded error band
    alpha = 0.20 if method == 'ContextOS' else 0.12
    ax.fill_between(x_log,
                    mean_vals - err_vals,
                    mean_vals + err_vals,
                    color=col,
                    alpha=alpha,
                    zorder=zord - 1)

# ---------------------------------------------------------------------------
# Axes formatting
# ---------------------------------------------------------------------------
ax.set_title('Figure 2: Scaling Analysis — TSR vs Context Length',
             fontsize=13, fontweight='bold', fontfamily='serif', pad=10)
ax.set_xlabel('Context Length (tokens)', fontsize=11, fontfamily='serif', labelpad=8)
ax.set_ylabel('Task Success Rate (%)', fontsize=11, fontfamily='serif', labelpad=8)

# Set x ticks at the log2 positions with readable labels
ax.set_xticks(x_log)
ax.set_xticklabels(context_labels, fontsize=10, fontfamily='serif')
ax.set_xlim(x_log[0] - 0.3, x_log[-1] + 0.3)

ax.set_ylim(10, 95)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

ax.grid(axis='both', which='major', linestyle='--', linewidth=0.6,
        alpha=0.65, zorder=0)
ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.35,
        alpha=0.4, zorder=0)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Annotate ContextOS endpoints
ctx_idx = methods.index('ContextOS')
for xi, yi in zip(x_log, tsr[ctx_idx]):
    ax.annotate(f'{yi:.1f}',
                xy=(xi, yi),
                xytext=(0, 9),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                color='#CC0000', fontfamily='serif')

# Add a subtle note about log scale
ax.text(x_log[-1] + 0.05, 12,
        'x-axis: log₂ scale', fontsize=8,
        color='#888888', ha='right', fontfamily='serif', style='italic')

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend = ax.legend(loc='upper right',
                   fontsize=9,
                   frameon=True,
                   framealpha=0.92,
                   edgecolor='#AAAAAA',
                   title='Method',
                   title_fontsize=9,
                   prop={'family': 'serif', 'size': 9})
legend.get_title().set_fontfamily('serif')
legend.get_title().set_fontweight('bold')

fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = (r'C:/Users/sudheer.pv/Documents/Research/'
            r'3.ContextOS-main/figures/fig2_scaling_analysis.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved: {out_path}')
plt.close(fig)
