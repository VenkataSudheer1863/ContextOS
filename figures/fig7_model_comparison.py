import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['GPT-OSS-20B', 'Qwen3', 'GLM-4.5']
methods = ['Full Context', 'RAG-Only', 'MemGPT', 'ContextOS']

tsr_by_model = {
    'GPT-OSS-20B': [58.4, 72.1, 74.8, 78.2],
    'Qwen3':       [56.2, 70.4, 73.1, 76.9],
    'GLM-4.5':     [53.8, 68.7, 71.4, 75.1],
}

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

x = np.arange(len(models))
n_methods = len(methods)
bar_width = 0.18
offsets = np.linspace(-(n_methods - 1) / 2 * bar_width, (n_methods - 1) / 2 * bar_width, n_methods)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (method, color) in enumerate(zip(methods, colors)):
    values = [tsr_by_model[m][i] for m in models]
    bars = ax.bar(x + offsets[i], values, width=bar_width, label=method, color=color,
                  edgecolor='white', linewidth=0.8, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f'{val}', ha='center', va='bottom', fontsize=7.5, color='#333333')

ax.set_xlabel('Backbone LLM', fontsize=12)
ax.set_ylabel('Task Success Rate (%)', fontsize=12)
ax.set_title('ContextOS Performance Across Backbone LLMs', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(45, 85)
ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(title='Method', fontsize=10, title_fontsize=10, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
out_path = 'C:/Users/sudheer.pv/Documents/Research/3.ContextOS-main/figures/fig7_model_comparison.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved: {out_path}')
