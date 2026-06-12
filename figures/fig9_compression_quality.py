import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rouge_l = [0.38, 0.52, 0.63, 0.72, 0.79, 0.84, 0.88, 0.92, 0.96, 1.0]
semantic_sim = [0.71, 0.79, 0.84, 0.88, 0.91, 0.93, 0.95, 0.97, 0.98, 1.0]

strategies = ['ExtractiveCompressor', 'AbstractiveCompressor', 'HierarchicalCompressor']
strategy_points = {
    'ExtractiveCompressor':   [(0.4, 0.71), (0.6, 0.83)],
    'AbstractiveCompressor':  [(0.3, 0.79), (0.5, 0.88)],
    'HierarchicalCompressor': [(0.2, 0.74), (0.4, 0.88)],
}
strategy_colors = ['#DD8452', '#55A868', '#8172B2']
strategy_markers = ['o', 's', '^']

ratios_arr = np.array(ratios)
rouge_arr = np.array(rouge_l)
sem_arr = np.array(semantic_sim)

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()

ax1.axvspan(0.35, 0.55, alpha=0.12, color='#4C72B0', label='Recommended region (0.35-0.55)')

line1, = ax1.plot(ratios_arr, rouge_arr, color='#4C72B0', linewidth=2.2,
                  marker='D', markersize=5, label='ROUGE-L', zorder=4)
line2, = ax2.plot(ratios_arr, sem_arr, color='#C44E52', linewidth=2.2,
                  linestyle='--', marker='D', markersize=5, label='Semantic Similarity', zorder=4)

scatter_handles = []
for (strategy, points), color, marker in zip(strategy_points.items(), strategy_colors, strategy_markers):
    xs = [p[0] for p in points]
    ys_rouge = [p[1] for p in points]
    sc = ax1.scatter(xs, ys_rouge, color=color, marker=marker, s=80, zorder=5,
                     edgecolors='white', linewidths=0.8, label=strategy)
    scatter_handles.append(sc)

ax1.set_xlabel('Compression Ratio', fontsize=12)
ax1.set_ylabel('ROUGE-L Score', fontsize=12, color='#4C72B0')
ax2.set_ylabel('Semantic Similarity', fontsize=12, color='#C44E52')
ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#C44E52')
ax1.set_xlim(0.05, 1.05)
ax1.set_ylim(0.25, 1.08)
ax2.set_ylim(0.60, 1.08)
ax1.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

ax1.set_title('Compression Quality vs. Ratio Tradeoff', fontsize=13, fontweight='bold')

shade_patch = plt.Rectangle((0, 0), 1, 1, fc='#4C72B0', alpha=0.2, label='Recommended region (0.35-0.55)')
all_handles = [line1, line2, shade_patch] + scatter_handles
all_labels = [h.get_label() for h in all_handles]
ax1.legend(all_handles, all_labels, fontsize=8.5, loc='upper left',
           framealpha=0.9, edgecolor='#cccccc')

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.tight_layout()
out_path = 'C:/Users/sudheer.pv/Documents/Research/3.ContextOS-main/figures/fig9_compression_quality.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved: {out_path}')
