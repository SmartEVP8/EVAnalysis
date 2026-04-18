import matplotlib.pyplot as plt
import numpy as np

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#4fc3f7"
ACCENT3  = "#e57373"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"


def render(df, simtime_ms):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    if df.is_empty() or "utilization" not in df.columns:
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center", color=SUBTEXT)
        return fig

    values = df["utilization"].to_numpy()

    values = np.clip(values, 0, 1)

    bins = np.linspace(0, 1, 21)

    ax.hist(values, bins=bins, color=ACCENT, edgecolor="#000000", alpha=0.85)

    ax.set_title("Utilization Distribution", color=TEXT, fontsize=12)
    ax.set_xlabel("Utilization", color=SUBTEXT)
    ax.set_ylabel("# Stations", color=SUBTEXT)

    ax.tick_params(colors=SUBTEXT)

    return fig