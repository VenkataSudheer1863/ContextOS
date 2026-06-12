import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

K_values = [1, 3, 5, 10, 20]

methods = [
    'BM25',
    'Dense (BGE-M3)',
    'Dense (E5)',
    'Hybrid',
    'ContextOS (Hybrid+Rerank)',
]

precision_at_k = {
    'BM25':                      [0.58, 0.52, 0.48, 0.41, 0.35],
    'Dense (BGE-M3)':            [0.74, 0.68, 0.63, 0.55, 0.48],
    'Dense (E5)':                [0.71, 0.65, 0.60, 0.52, 0.45],
    'Hybrid':                    [0.79, 0.73, 0.68, 0.61, 0.53],
    'ContextOS (Hybrid+Rerank)': [0.89, 0.83, 0.78, 0.71, 0.63],
}

ndcg_at_k = {
    'BM25':                      [0.58, 0.54, 0.51, 0.47, 0.43],
    'Dense (BGE-M3)':            [0.74, 0.70, 0.67, 0.63, 0.59],
    'Dense (E5)':                [0.71, 0.67, 0.64, 0.60, 0.56],
    'Hybrid':                    [0.79, 0.75, 0.72, 0.68, 0.64],
    'ContextOS (Hybrid+Rerank)': [0.89, 0.85, 0.82, 0.78, 0.74],
}

colors  = ['#7F8C8D', '#2980B9', '#27AE60', '#E67E22', '#C0392B']
markers = ['s',       'o',       '^',       'D',       '*']
lws     = [1.5,        1.8,       1.8,       2.0,       2.5]
ms      = [7,          8,         8,         8,         12]

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
fig.suptitle('Retrieval Quality Comparison: P@K and NDCG@K',
             fontsize=14, fontweight='bold', y=1.01)

panel_data = [
    (axes[0], precision_at_k, 'Precision@K (P@K)', 'P@K'),
    (axes[1], ndcg_at_k,      'NDCG@K',            'NDCG@K'),
]

for ax, data_dict, title, ylabel in panel_data:
    for method, color, marker, lw, ms_ in zip(methods, colors, markers, lws, ms):
        vals = data_dict[method]
        zorder = 5 if 'ContextOS' in method else 3
        ax.plot(K_values, vals,
                color=color, marker=marker, linewidth=lw,
                markersize=ms_, label=method, zorder=zorder,
                markerfacecolor=color if 'ContextOS' not in method else '#FDEDEC',
                markeredgecolor=color, markeredgewidth=1.5)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(K_values)
    ax.set_xlim(0.5, 21)
    ax.set_ylim(0.28, 0.96)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate ContextOS final point
    ctx_vals = data_dict['ContextOS (Hybrid+Rerank)']
    for k, v in zip(K_values, ctx_vals):
        ax.annotate(f'{v:.2f}', xy=(k, v),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7.5, color='#C0392B', fontweight='bold')

# Shared legend below both panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', ncol=3,
           fontsize=9.5, framealpha=0.95,
           edgecolor='#BDC3C7',
           bbox_to_anchor=(0.5, -0.10))

plt.tight_layout()
plt.savefig('C:/Users/sudheer.pv/Documents/Research/3.ContextOS-main/figures/fig6_retrieval_quality.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved fig6_retrieval_quality.png")
