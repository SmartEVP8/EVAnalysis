import matplotlib.pyplot as plt
import numpy as np

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT2  = "#81c784"
ACCENT3  = "#e57373"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"

def render(df, simtime_ms):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    if df.is_empty() or "queue_size" not in df.columns:
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center", color=SUBTEXT)
        return fig

    values = df["queue_size"].to_numpy()

    # Optional: clamp to avoid crazy outliers ruining chart
    values = np.clip(values, 0, 20)

    bins = np.arange(0, 22)  # 0–21 edges → bars at integers

    ax.hist(values, bins=bins, color=ACCENT2, edgecolor="#000000", alpha=0.85)

    ax.set_title("Queue Size Distribution", color=TEXT, fontsize=12)
    ax.set_xlabel("Queue Size", color=SUBTEXT)
    ax.set_ylabel("# Stations", color=SUBTEXT)

    ax.set_xticks(np.arange(0, 21, 2))
    ax.tick_params(colors=SUBTEXT)

    return fig