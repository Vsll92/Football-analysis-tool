"""
Pitch and chart visualizations using matplotlib.
Opta coordinates: x ∈ [0, 100], y ∈ [0, 100].
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


# ─── Color setup ──────────────────────────────────────────────────────────────
PITCH_COLOR = "#1a472a"
LINE_COLOR = "#FFFFFF"
CMAP_HEAT = LinearSegmentedColormap.from_list("heat", ["#1a472a", "#FFFF00", "#FF4500", "#8B0000"])
CMAP_COOL = LinearSegmentedColormap.from_list("cool", ["#313695", "#4575B4", "#ABD9E9", "#FFFFBF", "#FEE090", "#F46D43", "#A50026"])


def draw_pitch(ax, orientation="horizontal", half=False, color=PITCH_COLOR, lw=1.5):
    """Draw a football pitch on the given axes. Opta coords: x=0–100, y=0–100."""
    ax.set_facecolor(color)

    if orientation == "horizontal":
        # Pitch outline
        ax.plot([0, 100], [0, 0], color=LINE_COLOR, lw=lw)
        ax.plot([0, 100], [100, 100], color=LINE_COLOR, lw=lw)
        ax.plot([0, 0], [0, 100], color=LINE_COLOR, lw=lw)
        ax.plot([100, 100], [0, 100], color=LINE_COLOR, lw=lw)
        # Halfway
        ax.plot([50, 50], [0, 100], color=LINE_COLOR, lw=lw)
        ax.add_patch(plt.Circle((50, 50), 9.15, color=LINE_COLOR, fill=False, lw=lw))
        ax.plot(50, 50, "o", color=LINE_COLOR, ms=3)
        # Left penalty area
        ax.add_patch(patches.Rectangle((0, 21.1), 17, 57.8, fill=False, ec=LINE_COLOR, lw=lw))
        ax.add_patch(patches.Rectangle((0, 36.8), 5.5, 26.4, fill=False, ec=LINE_COLOR, lw=lw))
        ax.plot(11.5, 50, "o", color=LINE_COLOR, ms=3)
        # Right penalty area
        ax.add_patch(patches.Rectangle((83, 21.1), 17, 57.8, fill=False, ec=LINE_COLOR, lw=lw))
        ax.add_patch(patches.Rectangle((94.5, 36.8), 5.5, 26.4, fill=False, ec=LINE_COLOR, lw=lw))
        ax.plot(88.5, 50, "o", color=LINE_COLOR, ms=3)

        if half:
            ax.set_xlim(50, 101)
        else:
            ax.set_xlim(-1, 101)
        ax.set_ylim(-1, 101)

    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_shot_map(shots_df, title="Shot Map", team_color="#E8443A", figsize=(10, 7)):
    """Plot shots on a half-pitch with xG-sized markers."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, half=True)

    if len(shots_df) == 0:
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
        return fig

    for _, s in shots_df.iterrows():
        x, y = s.get("x", 50), s.get("y", 50)
        xg = s.get("xg", 0.05)
        is_goal = s.get("event") == "Goal"
        size = max(xg * 800, 30)
        color = "#2ECC71" if is_goal else team_color
        edge = "white" if is_goal else "#333"
        marker = "*" if is_goal else "o"
        ax.scatter(x, y, s=size, c=color, edgecolors=edge, linewidth=1.2,
                   marker=marker, zorder=5, alpha=0.85)

    # Legend
    ax.scatter([], [], s=200, c="#2ECC71", marker="*", label="Goal")
    ax.scatter([], [], s=80, c=team_color, edgecolors="#333", label="Shot (size=xG)")
    ax.legend(loc="lower right", fontsize=9, facecolor="#2C3E50",
              edgecolor="white", labelcolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#2C3E50")
    plt.tight_layout()
    return fig


def plot_pass_map(passes_df, title="Pass Map", color="#2E86AB", figsize=(10, 7),
                  show_failed=True, progressive_only=False):
    """Plot passes on full pitch."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax)

    if progressive_only:
        mask = (passes_df["Pass End X"] - passes_df["x"]) >= 10
        passes_df = passes_df[mask]

    success = passes_df[passes_df["outcome"] == 1]
    failed = passes_df[passes_df["outcome"] == 0]

    for _, p in success.iterrows():
        ax.annotate("", xy=(p["Pass End X"], p["Pass End Y"]),
                     xytext=(p["x"], p["y"]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.5))

    if show_failed and len(failed) > 0:
        for _, p in failed.iterrows():
            ax.annotate("", xy=(p["Pass End X"], p["Pass End Y"]),
                         xytext=(p["x"], p["y"]),
                         arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=0.6, alpha=0.3))

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#2C3E50")
    plt.tight_layout()
    return fig


def plot_heatmap(events_df, title="Action Heatmap", event_types=None, figsize=(10, 7)):
    """Plot heatmap of event locations."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, color="#1a2e1a")

    if event_types:
        events_df = events_df[events_df["event"].isin(event_types)]

    data = events_df[events_df["x"].notna() & events_df["y"].notna()]
    if len(data) == 0:
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
        fig.patch.set_facecolor("#2C3E50")
        return fig

    # KDE-style using 2D histogram
    x_bins = np.linspace(0, 100, 26)
    y_bins = np.linspace(0, 100, 26)
    heatmap, xedges, yedges = np.histogram2d(data["x"], data["y"], bins=[x_bins, y_bins])

    from scipy.ndimage import gaussian_filter
    heatmap = gaussian_filter(heatmap.T, sigma=1.5)

    ax.imshow(heatmap, extent=[0, 100, 0, 100], origin="lower",
              cmap=CMAP_HEAT, alpha=0.7, aspect="auto", zorder=2)

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#2C3E50")
    plt.tight_layout()
    return fig


def plot_defensive_actions(team_events, title="Defensive Actions", figsize=(10, 7)):
    """Plot tackles, interceptions, recoveries by location."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax)

    event_styles = {
        "Tackle": {"color": "#E74C3C", "marker": "^", "label": "Tackle"},
        "Interception": {"color": "#3498DB", "marker": "s", "label": "Interception"},
        "Ball recovery": {"color": "#2ECC71", "marker": "o", "label": "Recovery"},
        "Clearance": {"color": "#F39C12", "marker": "D", "label": "Clearance"},
    }

    for etype, style in event_styles.items():
        ev = team_events[(team_events["event"] == etype) & team_events["x"].notna()]
        ax.scatter(ev["x"], ev["y"], c=style["color"], marker=style["marker"],
                   s=40, alpha=0.6, label=style["label"], zorder=5, edgecolors="white", lw=0.5)

    ax.legend(loc="lower right", fontsize=8, facecolor="#2C3E50",
              edgecolor="white", labelcolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#2C3E50")
    plt.tight_layout()
    return fig


def plot_pass_network(avg_positions, title="Average Positions", figsize=(10, 7)):
    """Plot player average positions on pitch."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax)

    if len(avg_positions) == 0:
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
        fig.patch.set_facecolor("#2C3E50")
        return fig

    # Size by events
    max_events = avg_positions["events"].max()
    sizes = (avg_positions["events"] / max(max_events, 1)) * 600 + 100

    ax.scatter(avg_positions["avg_x"], avg_positions["avg_y"],
               s=sizes, c="#2E86AB", edgecolors="white", linewidth=2,
               zorder=5, alpha=0.85)

    for _, row in avg_positions.iterrows():
        name_short = row["player_name"].split()[-1] if isinstance(row["player_name"], str) else ""
        jersey = int(row["jersey"]) if pd.notna(row.get("jersey")) else ""
        label = f"{jersey} {name_short}" if jersey else name_short
        ax.annotate(label, (row["avg_x"], row["avg_y"]),
                    fontsize=7, fontweight="bold", color="white",
                    ha="center", va="bottom",
                    xytext=(0, 12), textcoords="offset points",
                    zorder=6)

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#2C3E50")
    plt.tight_layout()
    return fig


def plot_zone_entries(passes_df, title="Final Third Entries", figsize=(10, 7)):
    """Plot final-third entry passes."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax)

    entries = passes_df[
        (passes_df["x"] <= 66.6) &
        (passes_df["Pass End X"] > 66.6) &
        (passes_df["outcome"] == 1) &
        passes_df["x"].notna() &
        passes_df["Pass End X"].notna()
    ]

    for _, p in entries.iterrows():
        ax.annotate("", xy=(p["Pass End X"], p["Pass End Y"]),
                     xytext=(p["x"], p["y"]),
                     arrowprops=dict(arrowstyle="->", color="#E8443A", lw=0.8, alpha=0.5))

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#2C3E50")
    plt.tight_layout()
    return fig


def plot_corner_delivery(deliveries, title="Corner Delivery Zones", figsize=(8, 7)):
    """Plot where corner kicks are delivered to (half pitch)."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, half=True)

    if len(deliveries) == 0:
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
        fig.patch.set_facecolor("#2C3E50")
        return fig

    arr = np.array(deliveries)
    ax.scatter(arr[:, 0], arr[:, 1], c="#F39C12", s=80, alpha=0.7,
               edgecolors="white", zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    fig.patch.set_facecolor("#2C3E50")
    plt.tight_layout()
    return fig


def plot_comparison_radar(metrics_dict, labels, title="Window Comparison"):
    """Simple radar/polar chart for comparing metrics across windows."""
    categories = list(labels)
    n = len(categories)
    if n == 0:
        fig, ax = plt.subplots()
        return fig

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#1B2A4A")
    fig.patch.set_facecolor("#2C3E50")

    colors = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12"]
    for i, (window, values) in enumerate(metrics_dict.items()):
        vals = [values.get(cat, 0) for cat in categories]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=window, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, color="white")
    ax.tick_params(axis="y", colors="white", labelsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=9, facecolor="#2C3E50", edgecolor="white", labelcolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=20)
    plt.tight_layout()
    return fig


def plot_bar_comparison(data_dict, title="", ylabel="", figsize=(10, 5)):
    """Bar chart comparing metrics across windows."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#1B2A4A")
    fig.patch.set_facecolor("#2C3E50")

    windows = list(data_dict.keys())
    values = list(data_dict.values())
    colors = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12"][:len(windows)]

    bars = ax.bar(windows, values, color=colors, edgecolor="white", linewidth=0.5, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f"{val}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel(ylabel, color="white", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_lane_distribution(left, center, right, title="Lane Distribution", figsize=(6, 4)):
    """Horizontal stacked bar for left/center/right distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#1B2A4A")
    fig.patch.set_facecolor("#2C3E50")

    ax.barh(0, left, color="#3498DB", edgecolor="white", label=f"Left {left}%")
    ax.barh(0, center, left=left, color="#2ECC71", edgecolor="white", label=f"Center {center}%")
    ax.barh(0, right, left=left + center, color="#E74C3C", edgecolor="white", label=f"Right {right}%")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold", color="white", pad=10)
    ax.tick_params(colors="white")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3,
              fontsize=8, facecolor="#2C3E50", edgecolor="white", labelcolor="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    plt.tight_layout()
    return fig
