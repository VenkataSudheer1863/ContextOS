import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data
methods = ['Full Context', 'MemGPT', 'RAPTOR', 'Truncation', 'RAG-Only', 'ContextOS']
mean_tokens = [8420, 5240, 4980, 4096, 3820, 3120]
std_tokens = [1240, 760, 820, 0, 890, 420]

# TSR values at 8K context (from paper data — aligned to methods order)
# Full Context ~77.8 but uses all tokens; ContextOS is 77.8 with 3120 tokens
# Approximate TSR per method at 8K window
tsr_8k = [77.8, 68.9, 66.1, 58.4, 64.3, 77.8]

# Token savings vs Full Context
full_context_tokens = mean_tokens[0]
savings_pct = [(full_context_tokens - t) / full_context_tokens * 100 for t in mean_tokens]

# Color scheme: consistent steel-blue palette, ContextOS in dark blue
bar_colors = ['#9db8d2', '#7aa3c4', '#6a95b8', '#5a87a8', '#4a7898', '#1a3a6b']
line_color = '#b85c00'

x = np.arange(len(methods))
bar_width = 0.55

fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Bar chart: token usage ---
bars = ax1.bar(
    x,
    mean_tokens,
    width=bar_width,
    color=bar_colors,
    edgecolor='white',
    linewidth=0.7,
    yerr=std_tokens,
    capsize=5,
    error_kw=dict(elinewidth=1.2, ecolor='#444444', capthick=1.2),
    zorder=3,
    label='Mean Tokens Used'
)

# Token savings annotation above each bar (skip Full Context = 0%)
for i, (bar, pct, std) in enumerate(zip(bars, savings_pct, std_tokens)):
    top = bar.get_height() + std + 80
    if pct == 0.0:
        label = 'Baseline'
        color = '#444444'
    else:
        label = f'-{pct:.0f}%'
        color = '#1a6b3a'
    ax1.text(bar.get_x() + bar.get_width() / 2, top, label,
             ha='center', va='bottom', fontsize=8.5, color=color, fontweight='bold')

ax1.set_ylabel('Mean Tokens Used', fontsize=11, color='#333333')
ax1.set_ylim(0, 11500)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_xlabel('Method', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#333333')
ax1.set_axisbelow(True)
ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, color='#cccccc', zorder=0)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Secondary axis: TSR line ---
ax2 = ax1.twinx()
ax2.plot(
    x, tsr_8k,
    color=line_color,
    marker='o',
    markersize=7,
    linewidth=2.2,
    linestyle='-',
    zorder=4,
    label='TSR @ 8K (%)'
)

# Annotate TSR points
for xi, tsr in zip(x, tsr_8k):
    ax2.text(xi, tsr + 0.9, f'{tsr:.1f}', ha='center', va='bottom',
             fontsize=8, color=line_color, fontweight='bold')

ax2.set_ylabel('Task Success Rate @ 8K (%)', fontsize=11, color=line_color)
ax2.tick_params(axis='y', labelcolor=line_color)
ax2.set_ylim(45, 92)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Title
ax1.set_title('Token Efficiency vs. Task Success Rate by Method', fontsize=13,
              fontweight='bold', pad=12)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2,
           loc='upper right', fontsize=9, framealpha=0.85)

plt.tight_layout()
output_path = 'C:/Users/sudheer.pv/Documents/Research/3.ContextOS-main/figures/fig4_token_efficiency.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()
