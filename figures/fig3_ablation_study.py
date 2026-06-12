import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
configurations = [
    'Full ContextOS',
    'w/o LT Memory',
    'w/o Governance',
    'w/o Prioritization',
    'w/o Scheduling',
    'w/o Compression',
    'Scheduling Only',
    'Compression Only'
]
tsr_values = [77.8, 61.4, 62.8, 65.2, 68.4, 71.2, 70.1, 63.4]
std_values = [2.2, 3.1, 2.9, 2.7, 2.6, 2.4, 2.5, 2.8]
deltas = [0.0, -16.4, -15.0, -12.6, -9.4, -6.6, -7.7, -14.4]

# Sort by delta descending (Full ContextOS first, then descending magnitude of drop)
sorted_indices = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)
sorted_configs = [configurations[i] for i in sorted_indices]
sorted_tsr = [tsr_values[i] for i in sorted_indices]
sorted_std = [std_values[i] for i in sorted_indices]
sorted_deltas = [deltas[i] for i in sorted_indices]

# Colors: dark blue for Full ContextOS, steel blue for others
bar_colors = []
for cfg in sorted_configs:
    if cfg == 'Full ContextOS':
        bar_colors.append('#1a3a6b')
    else:
        bar_colors.append('#5b8db8')

fig, ax = plt.subplots(figsize=(10, 6))

y_positions = np.arange(len(sorted_configs))
bars = ax.barh(
    y_positions,
    sorted_tsr,
    xerr=sorted_std,
    color=bar_colors,
    edgecolor='white',
    linewidth=0.6,
    height=0.6,
    capsize=4,
    error_kw=dict(elinewidth=1.2, ecolor='#444444', capthick=1.2)
)

# Dashed vertical line at Full ContextOS performance
full_tsr = tsr_values[0]  # 77.8
ax.axvline(x=full_tsr, color='#1a3a6b', linestyle='--', linewidth=1.5,
           label=f'Full ContextOS ({full_tsr:.1f}%)', alpha=0.85)

# Delta labels on bars
for i, (tsr, delta, std) in enumerate(zip(sorted_tsr, sorted_deltas, sorted_std)):
    x_end = tsr + std + 0.5  # just past the error bar
    if delta == 0.0:
        label = 'Baseline'
        color = '#1a3a6b'
        fontweight = 'bold'
    else:
        label = f'{delta:+.1f}%'
        color = '#b22222'
        fontweight = 'normal'
    ax.text(x_end, i, label, va='center', ha='left',
            fontsize=9, color=color, fontweight=fontweight)

ax.set_yticks(y_positions)
ax.set_yticklabels(sorted_configs, fontsize=10)
ax.set_xlabel('Task Success Rate (%)', fontsize=11)
ax.set_title('Ablation Study (8K Token Context)', fontsize=13, fontweight='bold', pad=12)

ax.set_xlim(55, 86)
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5, color='#cccccc')
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='lower right', fontsize=9, framealpha=0.85)

plt.tight_layout()
output_path = 'C:/Users/sudheer.pv/Documents/Research/3.ContextOS-main/figures/fig3_ablation_study.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()
