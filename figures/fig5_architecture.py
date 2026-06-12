import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Color palette
COLOR_USER       = '#4A90D9'
COLOR_ORCH       = '#2C5F8A'
COLOR_RETRIEVAL  = '#27AE60'
COLOR_COMPRESS   = '#E67E22'
COLOR_GOVERN     = '#8E44AD'
COLOR_SCHED      = '#2980B9'
COLOR_WORKING    = '#16A085'
COLOR_LONGTERM   = '#1A5276'
COLOR_AGENT      = '#C0392B'
COLOR_SUB        = '#BDC3C7'
TEXT_LIGHT       = 'white'
TEXT_DARK        = '#2C3E50'


def draw_box(ax, cx, cy, w, h, label, color, fontsize=11, text_color='white',
             sublabel=None, radius=0.25):
    x = cx - w / 2
    y = cy - h / 2
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.05,rounding_size={radius}",
                         linewidth=1.5, edgecolor='#2C3E50',
                         facecolor=color, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.18, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
        ax.text(cx, cy - 0.22, sublabel, ha='center', va='center',
                fontsize=fontsize - 2.5, color=text_color, zorder=4, style='italic')
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)


def arrow(ax, x1, y1, x2, y2, color='#2C3E50', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=2)


# ── 1. User Query ──────────────────────────────────────────────────
draw_box(ax, 5, 13.0, 3.0, 0.7, 'User Query', COLOR_USER, fontsize=12)

# ── 2. Context Orchestrator ────────────────────────────────────────
draw_box(ax, 5, 11.6, 6.5, 0.85, 'Context Orchestrator', COLOR_ORCH, fontsize=13)

arrow(ax, 5, 12.65, 5, 12.02)   # User → Orchestrator

# ── 3. Three engine branches ───────────────────────────────────────
# Engine boxes (row 1)
draw_box(ax, 1.6, 10.15, 2.6, 0.70, 'Retrieval Engine', COLOR_RETRIEVAL, fontsize=10)
draw_box(ax, 5.0, 10.15, 2.6, 0.70, 'Compression Engine', COLOR_COMPRESS, fontsize=10)
draw_box(ax, 8.4, 10.15, 2.6, 0.70, 'Governance Engine', COLOR_GOVERN, fontsize=10)

# Arrows: Orchestrator bottom → each engine
arrow(ax, 5, 11.18, 1.6, 10.50)
arrow(ax, 5, 11.18, 5.0, 10.50)
arrow(ax, 5, 11.18, 8.4, 10.50)

# Sub-component boxes (row 2)
draw_box(ax, 1.6, 9.0, 2.6, 0.65, 'BGE-M3 / E5-Large-V2', COLOR_SUB,
         fontsize=9, text_color=TEXT_DARK)
draw_box(ax, 5.0, 9.0, 2.6, 0.65, 'Extractive / Abstractive', COLOR_SUB,
         fontsize=9, text_color=TEXT_DARK)
draw_box(ax, 8.4, 9.0, 2.6, 0.65, 'Retention / Forgetting', COLOR_SUB,
         fontsize=9, text_color=TEXT_DARK)

arrow(ax, 1.6, 9.80, 1.6, 9.33)
arrow(ax, 5.0, 9.80, 5.0, 9.33)
arrow(ax, 8.4, 9.80, 8.4, 9.33)

# ── 4. Context Scheduler ──────────────────────────────────────────
draw_box(ax, 5, 7.65, 3.8, 0.70, 'Context Scheduler', COLOR_SCHED, fontsize=11)
# Collect arrows from sub-boxes into scheduler
arrow(ax, 1.6, 8.68, 1.6, 7.65)
ax.annotate('', xy=(3.1, 7.65), xytext=(1.6, 7.65),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5), zorder=2)
arrow(ax, 5.0, 8.68, 5.0, 8.00)
arrow(ax, 8.4, 8.68, 8.4, 7.65)
ax.annotate('', xy=(6.9, 7.65), xytext=(8.4, 7.65),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5), zorder=2)

# ── 5. Working Memory ─────────────────────────────────────────────
draw_box(ax, 5, 6.45, 3.8, 0.70,
         'Working Memory', COLOR_WORKING, fontsize=11,
         sublabel='(Short-Term)')

arrow(ax, 5, 7.30, 5, 6.80)

# ── 6. Long-Term Memory ───────────────────────────────────────────
draw_box(ax, 5, 5.15, 5.2, 0.80,
         'Long-Term Memory', COLOR_LONGTERM, fontsize=11,
         sublabel='Episodic  |  Semantic  |  Procedural')

arrow(ax, 5, 6.10, 5, 5.55)

# ── 7. Agent Runtime ─────────────────────────────────────────────
draw_box(ax, 5, 3.80, 3.8, 0.70, 'Agent Runtime', COLOR_AGENT, fontsize=12)

arrow(ax, 5, 4.75, 5, 4.15)

# ── Title ─────────────────────────────────────────────────────────
ax.text(5, 13.72, 'ContextOS System Architecture',
        ha='center', va='center', fontsize=15, fontweight='bold',
        color='#1A252F')

# ── Legend strip ──────────────────────────────────────────────────
legend_items = [
    (COLOR_USER,      'User Interface'),
    (COLOR_ORCH,      'Orchestration'),
    (COLOR_RETRIEVAL, 'Retrieval'),
    (COLOR_COMPRESS,  'Compression'),
    (COLOR_GOVERN,    'Governance'),
    (COLOR_SCHED,     'Scheduling'),
    (COLOR_WORKING,   'Working Memory'),
    (COLOR_LONGTERM,  'Long-Term Memory'),
    (COLOR_AGENT,     'Agent Runtime'),
]
patches = [mpatches.Patch(facecolor=c, edgecolor='#2C3E50', label=l)
           for c, l in legend_items]
ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.01),
          ncol=3, fontsize=8, framealpha=0.9, edgecolor='#2C3E50')

plt.tight_layout()
plt.savefig('C:/Users/sudheer.pv/Documents/Research/3.ContextOS-main/figures/fig5_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved fig5_architecture.png")
