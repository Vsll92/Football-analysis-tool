"""
Interactive Plotly visualizations — v3 complete rewrite.
Fixes: surrealistic heatmaps, spaghetti pass maps, blob recovery maps.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd

PITCH_COLOR = "#1a472a"
PITCH_LINE = "#FFFFFF"
BG_COLOR = "#0e1117"


def _sp(value, fallback="?"):
    """Safe player short name: returns surname or fallback for NaN/None/empty."""
    if value is None:
        return fallback
    if isinstance(value, float):
        return fallback
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return fallback
    parts = s.split()
    return parts[-1] if parts else fallback


def _sf(value, fallback=""):
    """Safe full name."""
    if value is None:
        return fallback
    if isinstance(value, float):
        return fallback
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return fallback
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# PITCH SHAPES
# ═══════════════════════════════════════════════════════════════════════════════

def _pitch_shapes(half=False):
    shapes = []
    def rect(x0, y0, x1, y1):
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color=PITCH_LINE, width=1.5),
                           fillcolor="rgba(0,0,0,0)", layer="below"))
    def line(x0, y0, x1, y1):
        shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color=PITCH_LINE, width=1.5), layer="below"))
    def circ(xc, yc, r):
        shapes.append(dict(type="circle", x0=xc-r, y0=yc-r, x1=xc+r, y1=yc+r,
                           line=dict(color=PITCH_LINE, width=1.5),
                           fillcolor="rgba(0,0,0,0)", layer="below"))
    def dot(xc, yc, r=0.3):
        shapes.append(dict(type="circle", x0=xc-r, y0=yc-r, x1=xc+r, y1=yc+r,
                           fillcolor=PITCH_LINE, line=dict(color=PITCH_LINE, width=0)))

    rect(0, 0, 100, 100)
    line(50, 0, 50, 100)
    circ(50, 50, 9.15)
    rect(0, 21.1, 17, 78.9)
    rect(0, 36.8, 5.5, 63.2)
    rect(83, 21.1, 100, 78.9)
    rect(94.5, 36.8, 100, 63.2)
    dot(11.5, 50); dot(88.5, 50); dot(50, 50)
    return shapes


def _pitch_layout(title="", half=False, height=550, width=800):
    x_range = [48, 102] if half else [-2, 102]
    return go.Layout(
        title=dict(text=title, font=dict(color="white", size=16), x=0.5),
        xaxis=dict(range=x_range, showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=False, constrain="domain"),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", fixedrange=False),
        plot_bgcolor=PITCH_COLOR, paper_bgcolor=BG_COLOR,
        shapes=_pitch_shapes(half), height=height, width=width,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(font=dict(color="white", size=11), bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="gray", borderwidth=1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# REALISTIC HEATMAP — bin-based, not KDE contour
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_heatmap(events_df, title="Action Heatmap", event_types=None,
                        nx=20, ny=16):
    """
    Professional bin-based heatmap. Uses rectangular bins instead of
    Histogram2dContour to avoid surrealistic blob shapes.
    nx/ny control grid resolution (20×16 = ~5m×4.25m cells).
    """
    fig = go.Figure(layout=_pitch_layout(title))

    if event_types:
        events_df = events_df[events_df["event"].isin(event_types)]

    data = events_df[events_df["x"].notna() & events_df["y"].notna()]
    if data.empty:
        return fig

    x = data["x"].values
    y = data["y"].values

    # Create bins
    x_edges = np.linspace(0, 100, nx + 1)
    y_edges = np.linspace(0, 100, ny + 1)
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    H = H.T  # shape (ny, nx)

    # Normalize to per-match density
    n_matches = data["match_id"].nunique() if "match_id" in data.columns else 1
    n_matches = max(n_matches, 1)
    H = H / n_matches

    # Cell centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Professional red-to-transparent colorscale
    colorscale = [
        [0.0, "rgba(26,71,42,0)"],
        [0.15, "rgba(255,255,200,0.2)"],
        [0.3, "rgba(255,230,100,0.45)"],
        [0.5, "rgba(255,180,50,0.6)"],
        [0.7, "rgba(255,100,30,0.75)"],
        [0.85, "rgba(220,40,20,0.85)"],
        [1.0, "rgba(160,10,10,0.95)"],
    ]

    fig.add_trace(go.Heatmap(
        z=H, x=x_centers, y=y_centers,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="x: %{x:.0f}<br>y: %{y:.0f}<br>actions/match: %{z:.1f}<extra></extra>",
        zsmooth="best",  # light smoothing on bin values, not raw points
        dx=(x_edges[1] - x_edges[0]),
        dy=(y_edges[1] - y_edges[0]),
    ))

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# RECOVERY DOT MAP — dots + zone grid with counts
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_recovery_map(events_df, title="Recovery Zones", show_zones=True):
    """
    Recovery map showing individual dots + optional zone grid with counts.
    Much more tactically useful than a contour blob.
    """
    fig = go.Figure(layout=_pitch_layout(title))

    recoveries = events_df[
        (events_df["event"] == "Ball recovery") &
        events_df["x"].notna() & events_df["y"].notna()
    ]

    if recoveries.empty:
        return fig

    n_matches = recoveries["match_id"].nunique() if "match_id" in recoveries.columns else 1
    n_matches = max(n_matches, 1)

    # Zone grid: 6 columns × 3 rows
    x_edges = [0, 33.3, 50, 66.6, 83, 100]  # def third, mid-low, mid-high, final approach, box
    y_edges = [0, 33.3, 66.6, 100]  # right, center, left
    zone_labels_x = ["Def Third", "Mid-Low", "Mid-High", "Final Approach", "Box Area"]
    zone_labels_y = ["Right", "Center", "Left"]

    if show_zones:
        # Draw zone rectangles with count labels
        total = len(recoveries)
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                x0, x1 = x_edges[i], x_edges[i + 1]
                y0, y1 = y_edges[j], y_edges[j + 1]

                mask = (
                    (recoveries["x"] >= x0) & (recoveries["x"] < x1) &
                    (recoveries["y"] >= y0) & (recoveries["y"] < y1)
                )
                count = mask.sum()
                pct = round(count / max(total, 1) * 100, 1)
                per_match = round(count / n_matches, 1)

                # Zone rectangle
                opacity = min(0.5, pct / 25)  # scale opacity by density
                fig.add_shape(
                    type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                    fillcolor=f"rgba(46,134,171,{opacity})",
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5),
                    layer="above",
                )

                # Count label
                if pct >= 1:
                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2
                    fig.add_annotation(
                        x=cx, y=cy, text=f"<b>{pct:.0f}%</b><br>{per_match}/m",
                        showarrow=False,
                        font=dict(color="white", size=10),
                        opacity=0.85,
                    )

    # Individual dots
    hovers = recoveries.apply(
        lambda r: f"<b>{r.get('player_name', '?')}</b><br>Ball Recovery<br>Min: {int(r.get('time_min', 0))}",
        axis=1
    )
    fig.add_trace(go.Scatter(
        x=recoveries["x"], y=recoveries["y"],
        mode="markers",
        marker=dict(size=5, color="#2ECC71", opacity=0.5,
                    line=dict(color="white", width=0.3)),
        hovertext=hovers, hoverinfo="text",
        name="Recovery", showlegend=True,
    ))

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD-UP PITCH VIZ — arrows, zones, percentages on pitch
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_buildup_pitch(team_events, num_matches, title="Build-Up Structure"):
    """
    Professional build-up visualization showing:
    - GK distribution arrows (short vs long, directional)
    - First-receiver zones with labels
    - Build-up exit lanes (left/center/right) with thickness
    - Progression arrows into middle third
    All with percentages directly on the pitch.
    """
    fig = go.Figure(layout=_pitch_layout(title, height=620, width=900))

    passes = team_events[team_events["event"] == "Pass"]
    gk_passes = passes[(passes["Player Position"] == 1.0) | (passes["position"] == "GK")]
    def_passes = passes[passes["x"] <= 33.3]
    succ_def = def_passes[def_passes["outcome"] == 1]

    nm = max(num_matches, 1)
    total_gk = max(len(gk_passes), 1)

    # ─── GK distribution arrows ──────────────────────────────────────
    gk_short = gk_passes[~gk_passes["is_Long ball"] & ~gk_passes["is_Goal Kick"] & ~gk_passes["is_GK hoof"]]
    gk_long = gk_passes[gk_passes["is_Long ball"] | gk_passes["is_Goal Kick"] | gk_passes["is_GK hoof"]]

    short_pct = round(len(gk_short) / total_gk * 100, 1)
    long_pct = round(len(gk_long) / total_gk * 100, 1)

    # GK zone marker
    fig.add_annotation(
        x=7, y=50, text=f"<b>GK</b><br>Short: {short_pct}%<br>Long: {long_pct}%",
        showarrow=False, font=dict(color="white", size=11),
        bgcolor="rgba(0,0,0,0.6)", bordercolor="white", borderwidth=1, borderpad=6,
    )

    # Short pass target zones (aggregate into 3 lanes)
    def _lane_pct(df, label_prefix):
        """Get left/center/right percentages for pass destinations."""
        if df.empty:
            return {"left": 0, "center": 0, "right": 0}
        total = len(df)
        left = len(df[df["Pass End Y"] >= 66.6])
        center = len(df[(df["Pass End Y"] >= 33.3) & (df["Pass End Y"] < 66.6)])
        right = len(df[df["Pass End Y"] < 33.3])
        return {
            "left": round(left / total * 100, 1),
            "center": round(center / total * 100, 1),
            "right": round(right / total * 100, 1),
        }

    short_lanes = _lane_pct(gk_short[gk_short["outcome"] == 1], "GK Short")
    long_lanes = _lane_pct(gk_long[gk_long["outcome"] == 1], "GK Long")

    # Draw GK short arrows (3 directions)
    arrow_configs = [
        ("left", 25, 78, short_lanes["left"], "#3498DB"),
        ("center", 28, 50, short_lanes["center"], "#2ECC71"),
        ("right", 25, 22, short_lanes["right"], "#E74C3C"),
    ]
    for lane, ax, ay, pct, color in arrow_configs:
        if pct > 0:
            width = max(1, pct / 10)
            fig.add_annotation(
                x=ax, y=ay, ax=8, ay=50,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=width, arrowcolor=color, opacity=0.8,
            )

    # ─── First receivers in defensive third ──────────────────────────
    # Who receives in the first third
    if len(succ_def) > 0:
        first_recv_zone = succ_def[
            (succ_def["Pass End X"] <= 40) & (succ_def["Pass End X"] > 10)
        ]
        if len(first_recv_zone) > 0:
            # Group by receiving player
            recv_counts = first_recv_zone.groupby("player_name").size().sort_values(ascending=False).head(4)
            # Get average positions of top receivers
            for rank, (pname, count) in enumerate(recv_counts.items()):
                player_passes = first_recv_zone[first_recv_zone["player_name"] == pname]
                avg_end_x = player_passes["Pass End X"].mean()
                avg_end_y = player_passes["Pass End Y"].mean()
                pct = round(count / max(len(first_recv_zone), 1) * 100, 1)
                short_name = _sp(pname, "?")
                fig.add_trace(go.Scatter(
                    x=[avg_end_x], y=[avg_end_y],
                    mode="markers+text",
                    marker=dict(size=18 - rank * 2, color="#2E86AB", opacity=0.8,
                                line=dict(color="white", width=1.5)),
                    text=f"{short_name}<br>{pct}%",
                    textfont=dict(color="white", size=9),
                    textposition="top center",
                    hovertext=f"<b>{pname}</b><br>Receives: {count}<br>{pct}% of first-third receptions",
                    hoverinfo="text",
                    showlegend=False,
                ))

    # ─── Build-up exit lanes (from def third into mid third) ─────────
    exits = succ_def[succ_def["Pass End X"] > 33.3]
    exit_lanes = _lane_pct(exits, "Exit")
    total_exits = len(exits)

    # Lane zone annotations at the 33.3 line
    lane_y_centers = [83, 50, 17]
    lane_labels = ["Left", "Center", "Right"]
    lane_pcts = [exit_lanes["left"], exit_lanes["center"], exit_lanes["right"]]
    lane_colors = ["#3498DB", "#2ECC71", "#E74C3C"]

    for ly, lbl, pct, clr in zip(lane_y_centers, lane_labels, lane_pcts, lane_colors):
        if pct > 0:
            # Exit arrow
            width = max(1.5, pct / 8)
            fig.add_annotation(
                x=40, y=ly, ax=28, ay=ly,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=width, arrowcolor=clr, opacity=0.7,
            )
            # Label
            fig.add_annotation(
                x=42, y=ly,
                text=f"<b>{lbl}</b><br>{pct}%<br>{round(total_exits * pct / 100 / nm, 1)}/m",
                showarrow=False, font=dict(color="white", size=10),
                bgcolor=f"rgba(0,0,0,0.5)", borderpad=4,
            )

    # ─── Progression into final third ────────────────────────────────
    mid_passes = passes[(passes["x"] > 33.3) & (passes["x"] <= 66.6) & (passes["outcome"] == 1)]
    ft_entries = mid_passes[mid_passes["Pass End X"] > 66.6]
    ft_lanes = _lane_pct(ft_entries, "FT Entry")

    # Arrow at the 66.6 line
    for ly, pct, clr in zip([83, 50, 17], [ft_lanes["left"], ft_lanes["center"], ft_lanes["right"]], lane_colors):
        if pct > 3:
            width = max(1, pct / 10)
            fig.add_annotation(
                x=72, y=ly, ax=62, ay=ly,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1,
                arrowwidth=width, arrowcolor=clr, opacity=0.5,
            )

    # ─── Zone divider lines (thirds + lanes) ────────────────────────
    for xv in [33.3, 66.6]:
        fig.add_shape(type="line", x0=xv, y0=0, x1=xv, y1=100,
                      line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"))
    for yv in [33.3, 66.6]:
        fig.add_shape(type="line", x0=0, y0=yv, x1=100, y1=yv,
                      line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot"))

    # ─── Summary box ─────────────────────────────────────────────────
    total_buildup = max(len(def_passes), 1)
    buildup_success = round(succ_def["outcome"].sum() / total_buildup * 100, 1)
    fig.add_annotation(
        x=50, y=-6,
        text=f"Build-up passes/m: {round(len(def_passes)/nm, 1)} | "
             f"Success: {buildup_success}% | "
             f"Exits to mid third/m: {round(len(exits)/nm, 1)}",
        showarrow=False, font=dict(color="#BDC3C7", size=10),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SHOT MAP
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_shot_map(shots_df, title="Shot Map"):
    fig = go.Figure(layout=_pitch_layout(title, half=True))
    if shots_df.empty:
        return fig

    goals = shots_df[shots_df["event"] == "Goal"]
    non_goals = shots_df[shots_df["event"] != "Goal"]

    def _hover(row):
        body = "Header" if row.get("is_Head") else ("Left" if row.get("is_Left footed") else "Right foot")
        sit = "Open play"
        if row.get("is_From corner"): sit = "From corner"
        elif row.get("is_Free kick"): sit = "Free kick"
        elif row.get("is_Penalty"): sit = "Penalty"
        xg = row.get("xg", 0)
        return f"<b>{row.get('player_name', '?')}</b><br>{row.get('event')}<br>xG: {xg:.3f}<br>{body} | {sit}<br>{row.get('_match_label', '')}"

    if len(non_goals) > 0:
        hovers = non_goals.apply(_hover, axis=1)
        sizes = non_goals["xg"].clip(0.02, 1) * 40 + 6
        fig.add_trace(go.Scatter(
            x=non_goals["x"], y=non_goals["y"], mode="markers",
            marker=dict(size=sizes, color="#E8443A", opacity=0.8,
                        line=dict(color="white", width=1)),
            hovertext=hovers, hoverinfo="text", name="Shot",
        ))
    if len(goals) > 0:
        hovers = goals.apply(_hover, axis=1)
        sizes = goals["xg"].clip(0.02, 1) * 40 + 8
        fig.add_trace(go.Scatter(
            x=goals["x"], y=goals["y"], mode="markers",
            marker=dict(size=sizes, color="#2ECC71", opacity=0.95,
                        symbol="star", line=dict(color="white", width=1.5)),
            hovertext=hovers, hoverinfo="text", name="Goal",
        ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PASS MAP — vectorized, no per-row annotation loop
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_pass_map(passes_df, title="Pass Map", progressive_only=False, max_passes=250):
    """Vectorized pass map using line traces instead of per-row annotations."""
    fig = go.Figure(layout=_pitch_layout(title))

    if progressive_only:
        passes_df = passes_df[
            (passes_df["Pass End X"] - passes_df["x"]) >= 10
        ]

    valid = passes_df[
        passes_df["x"].notna() & passes_df["Pass End X"].notna()
    ]

    if len(valid) > max_passes:
        valid = valid.sample(max_passes, random_state=42)

    succ = valid[valid["outcome"] == 1]
    fail = valid[valid["outcome"] == 0]

    # Draw passes as line segments (vectorized — much faster than annotations)
    for df_slice, color, name in [(succ, "rgba(46,134,171,0.35)", "Successful"),
                                   (fail, "rgba(231,76,60,0.35)", "Failed")]:
        if df_slice.empty:
            continue
        xs, ys = [], []
        for _, p in df_slice.iterrows():
            xs.extend([p["x"], p["Pass End X"], None])
            ys.extend([p["y"], p["Pass End Y"], None])

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=1),
            hoverinfo="skip", name=name,
        ))

        # Origin dots with hover
        hovers = df_slice.apply(
            lambda r: f"<b>{r.get('player_name', '?')}</b><br>{'Long' if r.get('is_Long ball') else 'Short'} pass<br>{'✅' if r['outcome']==1 else '❌'}",
            axis=1
        )
        dot_color = "#2E86AB" if "Successful" in name else "#E74C3C"
        fig.add_trace(go.Scatter(
            x=df_slice["x"], y=df_slice["y"], mode="markers",
            marker=dict(size=3, color=dot_color, opacity=0.5),
            hovertext=hovers, hoverinfo="text", showlegend=False,
        ))

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DEFENSIVE ACTION MAP
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_defensive_map(team_events, title="Defensive Actions"):
    fig = go.Figure(layout=_pitch_layout(title))

    event_styles = {
        "Tackle": ("#E74C3C", "triangle-up", "Tackle"),
        "Interception": ("#3498DB", "square", "Interception"),
        "Ball recovery": ("#2ECC71", "circle", "Recovery"),
        "Clearance": ("#F39C12", "diamond", "Clearance"),
    }
    for etype, (color, symbol, label) in event_styles.items():
        ev = team_events[(team_events["event"] == etype) & team_events["x"].notna()]
        if ev.empty:
            continue
        hovers = ev.apply(
            lambda r: f"<b>{r.get('player_name', '?')}</b><br>{label}<br>{'Won' if r['outcome']==1 else 'Lost'}<br>Min: {int(r.get('time_min',0))}",
            axis=1
        )
        fig.add_trace(go.Scatter(
            x=ev["x"], y=ev["y"], mode="markers",
            marker=dict(size=9, color=color, symbol=symbol, opacity=0.7,
                        line=dict(color="white", width=0.8)),
            hovertext=hovers, hoverinfo="text", name=label,
        ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# AVERAGE POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_avg_positions(avg_pos_df, title="Average Positions"):
    fig = go.Figure(layout=_pitch_layout(title))
    if avg_pos_df.empty:
        return fig

    max_ev = avg_pos_df["events"].max()
    sizes = (avg_pos_df["events"] / max(max_ev, 1)) * 35 + 12

    hovers = avg_pos_df.apply(
        lambda r: f"<b>{r['player_name']}</b><br>#{int(r['jersey']) if pd.notna(r.get('jersey')) else '?'} | {r.get('position','?')}<br>Events: {r['events']}",
        axis=1
    )
    fig.add_trace(go.Scatter(
        x=avg_pos_df["avg_x"], y=avg_pos_df["avg_y"],
        mode="markers+text",
        marker=dict(size=sizes, color="#2E86AB", opacity=0.85,
                    line=dict(color="white", width=2)),
        text=avg_pos_df.apply(lambda r: f"{int(r['jersey']) if pd.notna(r.get('jersey')) else ''}", axis=1),
        textfont=dict(color="white", size=10), textposition="middle center",
        hovertext=hovers, hoverinfo="text", name="Players", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=avg_pos_df["avg_x"], y=avg_pos_df["avg_y"] + 3,
        mode="text",
        text=avg_pos_df["player_name"].apply(lambda n: _sp(n, "")),
        textfont=dict(color="white", size=9), hoverinfo="skip", showlegend=False,
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE VISUALIZERS (kept from v2, no changes needed)
# ═══════════════════════════════════════════════════════════════════════════════

_EVENT_COLORS = {
    "Pass": "#2E86AB", "Take On": "#9B59B6", "Ball recovery": "#2ECC71",
    "Tackle": "#E74C3C", "Interception": "#3498DB", "Clearance": "#F39C12",
    "Miss": "#E8443A", "Saved Shot": "#E8443A", "Goal": "#FFD700",
    "Foul": "#F39C12", "Aerial": "#8E44AD", "Ball touch": "#95A5A6",
    "Dispossessed": "#C0392B", "Out": "#7F8C8D",
}


def visualize_sequence(seq, title="Possession Sequence"):
    fig = go.Figure(layout=_pitch_layout(title, height=580, width=850))
    steps = seq["steps"]
    if not steps:
        return fig

    for step in steps:
        if step["event"] == "Pass" and step["end_x"] is not None:
            color = "rgba(46,134,171,0.6)" if step["outcome"] == 1 else "rgba(231,76,60,0.6)"
            fig.add_annotation(
                x=step["end_x"], y=step["end_y"], ax=step["x"], ay=step["y"],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=2 if step["outcome"] == 1 else 1.5, arrowcolor=color,
            )

    xs = [s["x"] for s in steps if s["x"] is not None]
    ys = [s["y"] for s in steps if s["y"] is not None]
    colors = [_EVENT_COLORS.get(s["event"], "#95A5A6") for s in steps if s["x"] is not None]
    symbols = []
    for s in steps:
        if s["x"] is None: continue
        if s["event"] == "Goal": symbols.append("star")
        elif s["event"] in ("Miss", "Saved Shot"): symbols.append("x")
        elif s["event"] in ("Tackle", "Interception", "Ball recovery"): symbols.append("diamond")
        elif s["event"] == "Take On": symbols.append("triangle-up")
        else: symbols.append("circle")

    hovers = []
    for i, s in enumerate(steps):
        if s["x"] is None: continue
        hovers.append(f"<b>Step {i+1}: {s['event']}</b> {'✅' if s['outcome']==1 else '❌'}<br>Player: {_sf(s['player'], '?')} ({_sf(s['position'], '?')})<br>({s['x']:.0f}, {s['y']:.0f})")

    sizes = [14 if s["event"] in ("Goal", "Miss", "Saved Shot") else 10 for s in steps if s["x"] is not None]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors, symbol=symbols,
                    line=dict(color="white", width=1.5), opacity=0.9),
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))
    label_xs = [s["x"] for s in steps if s["x"] is not None]
    label_ys = [s["y"] + 3 for s in steps if s["y"] is not None]
    label_texts = [f"{i+1}. {_sp(s['player'])}" for i, s in enumerate(steps) if s["x"] is not None]
    fig.add_trace(go.Scatter(
        x=label_xs, y=label_ys, mode="text",
        text=label_texts, textfont=dict(color="white", size=8),
        hoverinfo="skip", showlegend=False,
    ))
    return fig


def visualize_sequence_set(sequences, title="Build-Up Patterns", max_display=50):
    fig = go.Figure(layout=_pitch_layout(title))
    seqs = sequences[:max_display]
    if not seqs:
        return fig

    for seq in seqs:
        for step in seq["steps"]:
            if step["event"] == "Pass" and step["end_x"] is not None and step["x"] is not None:
                color = "rgba(46,200,113,0.2)" if seq["contains_shot"] else "rgba(46,134,171,0.15)"
                if seq["contains_goal"]:
                    color = "rgba(255,215,0,0.4)"
                fig.add_annotation(
                    x=step["end_x"], y=step["end_y"], ax=step["x"], ay=step["y"],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1,
                    arrowcolor=color,
                )

    start_xs = [s["start_x"] for s in seqs if s["start_x"] is not None]
    start_ys = [s["start_y"] for s in seqs if s["start_y"] is not None]
    start_hovers = [f"<b>{s['start_player']}</b><br>{s.get('pattern','?')}<br>End: {s['end_type']}" for s in seqs if s["start_x"] is not None]
    fig.add_trace(go.Scatter(
        x=start_xs, y=start_ys, mode="markers",
        marker=dict(size=8, color="#2ECC71", symbol="circle", line=dict(color="white", width=1), opacity=0.7),
        hovertext=start_hovers, hoverinfo="text", name="Start",
    ))

    end_xs = [s["end_x"] for s in seqs if s["end_x"] is not None]
    end_ys = [s["end_y"] for s in seqs if s["end_y"] is not None]
    end_colors = ["#FFD700" if s["contains_goal"] else "#E8443A" if s["contains_shot"] else "#F39C12" for s in seqs if s["end_x"] is not None]
    fig.add_trace(go.Scatter(
        x=end_xs, y=end_ys, mode="markers",
        marker=dict(size=9, color=end_colors, symbol="x", line=dict(color="white", width=1), opacity=0.8),
        name="End",
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE ENTRY MAP
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_zone_entries(passes_df, title="Final-Third Entries"):
    fig = go.Figure(layout=_pitch_layout(title))
    entries = passes_df[
        (passes_df["x"] <= 66.6) & (passes_df["Pass End X"] > 66.6) &
        (passes_df["outcome"] == 1) & passes_df["x"].notna() & passes_df["Pass End X"].notna()
    ]
    if entries.empty:
        return fig
    # Use line segments instead of annotations for speed
    xs, ys = [], []
    for _, p in entries.iterrows():
        xs.extend([p["x"], p["Pass End X"], None])
        ys.extend([p["y"], p["Pass End Y"], None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="rgba(232,68,58,0.4)", width=1),
        hoverinfo="skip", name="Entry pass",
    ))
    hovers = entries.apply(
        lambda r: f"<b>{r.get('player_name','?')}</b><br>{'Long' if r.get('is_Long ball') else 'Short'} pass",
        axis=1
    )
    fig.add_trace(go.Scatter(
        x=entries["x"], y=entries["y"], mode="markers",
        marker=dict(size=5, color="#E8443A", opacity=0.5),
        hovertext=hovers, hoverinfo="text", name="Origin",
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CORNER DELIVERY MAP
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_corner_map(corners_df, title="Corner Deliveries"):
    fig = go.Figure(layout=_pitch_layout(title, half=True))
    targets = corners_df[corners_df["Pass End X"].notna()]
    if targets.empty:
        return fig
    swing = targets.apply(
        lambda r: "Inswing" if r.get("is_Inswinger") else ("Outswing" if r.get("is_Outswinger") else "Straight"), axis=1
    )
    colors = swing.map({"Inswing": "#2ECC71", "Outswing": "#E74C3C", "Straight": "#F39C12"})
    hovers = targets.apply(
        lambda r: f"<b>{r.get('player_name','?')}</b><br>{'✅' if r['outcome']==1 else '❌'}<br>Target: ({r['Pass End X']:.0f}, {r['Pass End Y']:.0f})",
        axis=1
    )
    fig.add_trace(go.Scatter(
        x=targets["Pass End X"], y=targets["Pass End Y"], mode="markers",
        marker=dict(size=12, color=colors, opacity=0.8, line=dict(color="white", width=1)),
        hovertext=hovers, hoverinfo="text", name="Delivery",
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BAR / LANE CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_bar(data_dict, title="", ylabel="", color_seq=None):
    if color_seq is None:
        color_seq = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"]
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    colors = [color_seq[i % len(color_seq)] for i in range(len(labels))]
    fig = go.Figure(data=[go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v}" for v in values], textposition="outside", textfont=dict(color="white"),
        hovertemplate="%{x}: %{y}<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14), x=0.5),
        plot_bgcolor="#1B2A4A", paper_bgcolor=BG_COLOR,
        yaxis=dict(title=ylabel, color="white", gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(color="white"), height=400, margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def interactive_lane_dist(left, center, right, title="Lane Distribution"):
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[""], x=[left], name=f"Left {left}%", orientation="h", marker_color="#3498DB",
                         text=f"Left {left}%", textposition="inside", textfont=dict(color="white")))
    fig.add_trace(go.Bar(y=[""], x=[center], name=f"Center {center}%", orientation="h", marker_color="#2ECC71",
                         text=f"Center {center}%", textposition="inside", textfont=dict(color="white")))
    fig.add_trace(go.Bar(y=[""], x=[right], name=f"Right {right}%", orientation="h", marker_color="#E74C3C",
                         text=f"Right {right}%", textposition="inside", textfont=dict(color="white")))
    fig.update_layout(
        barmode="stack",
        title=dict(text=title, font=dict(color="white", size=13), x=0.5),
        plot_bgcolor="#1B2A4A", paper_bgcolor=BG_COLOR,
        xaxis=dict(range=[0, 100], showticklabels=False), yaxis=dict(showticklabels=False),
        height=120, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.5, font=dict(color="white", size=10)),
    )
    return fig
