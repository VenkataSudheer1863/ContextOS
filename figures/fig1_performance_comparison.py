"""
Figure 1: Main Performance Comparison
Grouped bar chart showing TSR (%) across context lengths for all methods.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
methods = ['Full Context', 'Truncation', 'RAG-Only', 'MemGPT', 'RAPTOR', 'ContextOS']
context_lengths = [512, 2048, 8192, 32768]
context_labels = ['512', '2K', '8K', '32K']

tsr = {
    512:   [82.3, 81.1, 79.5, 80.1, 80.8, 84.2],
    2048:  [71.4, 65.8, 74.2, 75.8, 73.9, 80.5],
    8192:  [54.2, 42.1, 68.3, 70.2, 69.1, 77.8],
    32768: [31.5, 20.3, 52.4, 58.9, 57.2, 71.3],
}

std = {
    512:   [2.1, 2.0, 2.3, 2.2, 2.1, 1.8],
    2048:  [2.8, 3.1, 2.5, 2.4, 2.6, 2.0],
    8192:  [3.2, 3.8, 2.9, 2.7, 2.8, 2.2],
    32768: [4.1, 4.5, 3.6, 3.2, 3.4, 2.5],
}

colors = ['#4472C4', '#ED7D31', '#A9D18E', '#FFC000', '#70AD47', '#FF0000']
edge_colors = ['#2E4F8C', '#B85D1F', '#5A9E5A', '#C49000', '#3D7A28', '#CC0000']

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Figure 1: Task Success Rate by Context Length',
             fontsize=15, fontweight='bold', fontfamily='serif', y=0.98)

axes_flat = axes.flatten()
n_methods = len(methods)
bar_width = 0.13
x_base = np.arange(n_methods)
# Center the group of bars around each x tick
offsets = np.linspace(-(n_methods - 1) / 2 * bar_width,
                       (n_methods - 1) / 2 * bar_width,
                       n_methods)

for ax_idx, (ctx, ctx_label) in enumerate(zip(context_lengths, context_labels)):
    ax = axes_flat[ax_idx]
    values = tsr[ctx]
    errors = std[ctx]

    for i, (method, val, err, col, ecol, offset) in enumerate(
            zip(methods, values, errors, colors, edge_colors, offsets)):
        lw = 2.0 if method == 'ContextOS' else 0.8
        bar = ax.bar(
            i + offset,          # single bar per method, grouped visually
            val,
            width=bar_width,
            color=col,
            edgecolor=ecol,
            linewidth=lw,
            zorder=3,
            yerr=err,
            capsize=3,
            error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1.2),
        )
        # Emphasise ContextOS with a bold top outline
        if method == 'ContextOS':
            ax.bar(i + offset, val, width=bar_width,
                   color='none', edgecolor='#CC0000',
                   linewidth=2.5, zorder=4)

    # Axes formatting
    ax.set_title(f'Context Length: {ctx_label} tokens',
                 fontsize=11, fontweight='bold', fontfamily='serif', pad=6)
    ax.set_ylabel('Task Success Rate (%)', fontsize=9, fontfamily='serif')
    ax.set_xlim(-0.6, n_methods - 0.4)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, fontsize=8, fontfamily='serif',
                        rotation=25, ha='right')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.6,
            alpha=0.7, zorder=0)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.4,
            alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Annotate ContextOS bar value
    ctx_idx = methods.index('ContextOS')
    ctx_val = values[ctx_idx]
    ctx_err = errors[ctx_idx]
    ax.text(ctx_idx + offsets[ctx_idx], ctx_val + ctx_err + 1.5,
            f'{ctx_val:.1f}', ha='center', va='bottom',
            fontsize=7.5, fontweight='bold', color='#CC0000',
            fontfamily='serif')

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

# ---------------------------------------------------------------------------
# Shared legend
# ---------------------------------------------------------------------------
legend_patches = []
for method, col, ecol in zip(methods, colors, edge_colors):
    lw = 2.0 if method == 'ContextOS' else 0.8
    patch = mpatches.Patch(facecolor=col, edgecolor=ecol,
                            linewidth=lw, label=method)
    legend_patches.append(patch)

fig.legend(handles=legend_patches,
           loc='lower center',
           ncol=6,
           fontsize=9,
           frameon=True,
           framealpha=0.9,
           edgecolor='#AAAAAA',
           bbox_to_anchor=(0.5, 0.01),
           prop={'family': 'serif', 'size': 9})

fig.tight_layout(rect=[0, 0.06, 1, 0.97])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = (r'C:/Users/sudheer.pv/Documents/Research/'
            r'3.ContextOS-main/figures/fig1_performance_comparison.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved: {out_path}')
plt.close(fig)
