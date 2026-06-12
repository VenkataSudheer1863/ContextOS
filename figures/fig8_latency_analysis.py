import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

components = ['Retrieval', 'Prioritization', 'Scheduling', 'Compression', 'Governance', 'Total']
latency_ms = [42.1, 8.3, 12.7, 31.4, 18.9, 156.2]
latency_std = [8.4, 1.2, 2.1, 6.8, 3.2, 18.4]

colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#4C72B0', '#4C72B0', '#C44E52']

fig, ax = plt.subplots(figsize=(9, 5.5))

y_pos = np.arange(len(components))
bars = ax.barh(y_pos, latency_ms, xerr=latency_std, color=colors,
               edgecolor='white', linewidth=0.8, height=0.55,
               error_kw=dict(ecolor='#555555', capsize=4, elinewidth=1.2, capthick=1.2),
               zorder=3)

for bar, val, std in zip(bars, latency_ms, latency_std):
    ax.text(val + std + 2, bar.get_y() + bar.get_height() / 2,
            f'{val} ms', va='center', ha='left', fontsize=9, color='#333333')

ax.set_yticks(y_pos)
ax.set_yticklabels(components, fontsize=11)
ax.set_xlabel('Latency (ms)', fontsize=12)
ax.set_title('ContextOS Latency Breakdown per Component', fontsize=13, fontweight='bold')
ax.xaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(0, 210)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.axhline(y=4.5, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)

ax.annotate(
    'Total latency is 3.5x faster\nthan end-to-end inference latency',
    xy=(156.2, 5), xytext=(130, 4.5),
    fontsize=8.5, color='#C44E52', fontstyle='italic',
    ha='center', va='top',
    arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', fc='#fff5f5', ec='#C44E52', alpha=0.85)
)

fig.tight_layout()
out_path = 'C:/Users/sudheer.pv/Documents/Research/3.ContextOS-main/figures/fig8_latency_analysis.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved: {out_path}')
