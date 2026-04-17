"""
Player Radar Charts — Position-specific KPI radars using Plotly.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from config import RADAR_KPIS, get_position_group, POSITION_GROUP_ORDER, SHOT_EVENTS

BG_COLOR = "#0e1117"


def compute_player_radar_data(team_events, num_matches):
    """
    Compute per-player metrics needed for radar charts.
    Returns a DataFrame with one row per player and all metrics per-match normalised.
    """
    # Get most common position per player
    valid_events = team_events[team_events["player_name"].str.len() > 0]
    pos_counts = (
        valid_events[valid_events["position"].str.len() > 0]
        .groupby(["player_name", "position"])
        .size()
        .reset_index(name="cnt")
    )
    if len(pos_counts) > 0:
        best_pos = pos_counts.sort_values("cnt", ascending=False).drop_duplicates("player_name")
        player_pos_map = dict(zip(best_pos["player_name"], best_pos["position"]))
    else:
        player_pos_map = {}

    unique_players = valid_events["player_name"].unique()
    results = []

    for pname in unique_players:
        pos = player_pos_map.get(pname, "Unknown")
        grp = get_position_group(pos)

        pe = team_events[team_events["player_name"] == pname]
        passes = pe[pe["event"] == "Pass"]
        shots = pe[pe["event"].isin(SHOT_EVENTS)]
        goals = pe[pe["event"] == "Goal"]
        tackles = pe[pe["event"] == "Tackle"]
        interceptions = pe[pe["event"] == "Interception"]
        recoveries = pe[pe["event"] == "Ball recovery"]
        aerials = pe[pe["event"] == "Aerial"]
        take_ons = pe[pe["event"] == "Take On"]
        clearances = pe[pe["event"] == "Clearance"]
        dispossessed = pe[pe["event"] == "Dispossessed"]

        total_passes = max(len(passes), 1)
        succ_passes = passes["outcome"].sum()

        # Progressive passes
        prog_mask = (passes["Pass End X"] - passes["x"]) >= 10
        prog_passes = passes[prog_mask.fillna(False)]

        # Line breaking: from mid to final third
        lb_mask = (passes["x"] <= 66.6) & (passes["Pass End X"] > 66.6) & (passes["outcome"] == 1)
        lb_passes = passes[lb_mask]

        # Crosses
        crosses = passes[passes["is_Cross"]]

        # Through balls
        through_balls = passes[passes["is_Through ball"]]

        # Long balls
        long_balls = passes[passes["is_Long ball"]]

        # Key passes (assists + intentional assists)
        key_passes = passes[passes["is_Assist"] | passes["is_Intentional Assist"]]

        # GK-specific
        gk_passes_total = len(passes) if grp == "GK" else 0
        gk_short = len(passes[~passes["is_Long ball"] & ~passes["is_Goal Kick"] & ~passes["is_GK hoof"]]) if grp == "GK" else 0
        gk_long = len(passes[passes["is_Long ball"] | passes["is_Goal Kick"] | passes["is_GK hoof"]]) if grp == "GK" else 0

        # GK saves/claims
        saves = pe[pe["event"] == "Save"]
        claims = pe[pe["event"] == "Claim"]

        total_shots = max(len(shots), 1)
        total_xg = shots["xg"].sum() if "xg" in shots.columns else 0
        shots_on_target = len(shots[shots["event"].isin(["Goal", "Saved Shot"])])

        # Matches played (estimate from unique match_ids)
        matches_played = pe["match_id"].nunique() if "match_id" in pe.columns else num_matches
        matches_played = max(matches_played, 1)

        results.append({
            "player": pname,
            "position": pos,
            "pos_group": grp,
            "matches": matches_played,
            "events": len(pe),
            # Per-match metrics
            "passes_pm": round(len(passes) / matches_played, 1),
            "pass_accuracy": round(succ_passes / total_passes * 100, 1),
            "progressive_passes_pm": round(len(prog_passes) / matches_played, 1),
            "line_breaking_pm": round(len(lb_passes) / matches_played, 1),
            "crosses_pm": round(len(crosses) / matches_played, 1),
            "through_balls_pm": round(len(through_balls) / matches_played, 1),
            "long_ball_pm": round(len(long_balls) / matches_played, 1),
            "key_passes_pm": round(len(key_passes) / matches_played, 2),
            "shots_pm": round(len(shots) / matches_played, 1),
            "goals_pm": round(len(goals) / matches_played, 2),
            "xg_pm": round(total_xg / matches_played, 2),
            "xg_per_shot": round(total_xg / total_shots, 3) if len(shots) > 0 else 0,
            "shot_on_target_pct": round(shots_on_target / total_shots * 100, 1) if len(shots) > 0 else 0,
            "tackles_pm": round(len(tackles) / matches_played, 1),
            "tackle_success": round(tackles["outcome"].sum() / max(len(tackles), 1) * 100, 1),
            "interceptions_pm": round(len(interceptions) / matches_played, 1),
            "recoveries_pm": round(len(recoveries) / matches_played, 1),
            "clearances_pm": round(len(clearances) / matches_played, 1),
            "aerials_won_pct": round(aerials["outcome"].sum() / max(len(aerials), 1) * 100, 1),
            "take_ons_pm": round(len(take_ons) / matches_played, 1),
            "take_on_success": round(take_ons["outcome"].sum() / max(len(take_ons), 1) * 100, 1),
            "progressive_carries_pm": round(len(take_ons[take_ons["outcome"] == 1]) / matches_played, 1),
            "dispossessed_pm": round(len(dispossessed) / matches_played, 1),
            "avg_x": round(pe["x"].mean(), 1) if pe["x"].notna().any() else 50,
            "avg_y": round(pe["y"].mean(), 1) if pe["y"].notna().any() else 50,
            # GK specific
            "short_dist_pct": round(gk_short / max(gk_passes_total, 1) * 100, 1),
            "long_dist_pct": round(gk_long / max(gk_passes_total, 1) * 100, 1),
            "saves_pm": round(len(saves) / matches_played, 1),
            "claims_pm": round(len(claims) / matches_played, 1),
        })

    return pd.DataFrame(results)


def _normalize_for_radar(values, all_values):
    """Normalize values to 0-100 percentile range for radar display."""
    if len(all_values) <= 1:
        return [50] * len(values)
    mi, ma = min(all_values), max(all_values)
    if ma == mi:
        return [50] * len(values)
    return [round((v - mi) / (ma - mi) * 100, 1) for v in values]


def plot_player_radar(player_row, all_players_df, title=None):
    """
    Generate a radar chart for a single player based on their position group's KPIs.
    Values are percentile-normalised against all players in the same position group.
    """
    grp = player_row["pos_group"]
    kpis = RADAR_KPIS.get(grp, RADAR_KPIS.get("Interior/CM"))

    if not kpis:
        return go.Figure()

    kpi_keys = [k[0] for k in kpis]
    kpi_labels = [k[1] for k in kpis]

    # Get values for this player
    raw_values = [player_row.get(k, 0) for k in kpi_keys]

    # Get all values in this pos group for percentile normalization
    same_group = all_players_df[all_players_df["pos_group"] == grp]
    norm_values = []
    for k, raw in zip(kpi_keys, raw_values):
        all_vals = same_group[k].dropna().tolist() if k in same_group.columns else [0]
        if not all_vals:
            all_vals = [0]
        mi, ma = min(all_vals), max(all_vals)
        if ma == mi:
            norm_values.append(50)
        else:
            norm_values.append(round((raw - mi) / (ma - mi) * 100, 1))

    # Close the polygon
    norm_values.append(norm_values[0])
    kpi_labels_closed = kpi_labels + [kpi_labels[0]]

    player_name = player_row.get("player", "Unknown")
    if title is None:
        title = f"{player_name} — {grp} Radar"

    # Hover: show both raw and percentile
    hover_texts = []
    for i, (label, raw, norm) in enumerate(zip(kpi_labels, raw_values, norm_values[:-1])):
        hover_texts.append(f"<b>{label}</b><br>Value: {raw}<br>Percentile: {norm:.0f}%")
    hover_texts.append(hover_texts[0])  # close

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatterpolar(
        r=norm_values,
        theta=kpi_labels_closed,
        fill="toself",
        fillcolor="rgba(46,134,171,0.3)",
        line=dict(color="#2E86AB", width=2),
        hovertext=hover_texts,
        hoverinfo="text",
        name=player_name,
    ))

    # Reference line at 50%
    fig.add_trace(go.Scatterpolar(
        r=[50] * (len(kpi_labels) + 1),
        theta=kpi_labels_closed,
        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14), x=0.5),
        polar=dict(
            bgcolor="#1B2A4A",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(color="white", gridcolor="rgba(255,255,255,0.15)"),
        ),
        paper_bgcolor=BG_COLOR,
        showlegend=False,
        height=450, width=450,
        margin=dict(l=60, r=60, t=60, b=40),
    )

    return fig


def plot_player_comparison_radar(player_rows, all_players_df, title="Player Comparison"):
    """
    Overlay multiple players on the same radar chart.
    All must be from the same position group.
    """
    if not player_rows:
        return go.Figure()

    grp = player_rows[0]["pos_group"]
    kpis = RADAR_KPIS.get(grp, RADAR_KPIS.get("Interior/CM"))
    kpi_keys = [k[0] for k in kpis]
    kpi_labels = [k[1] for k in kpis]

    same_group = all_players_df[all_players_df["pos_group"] == grp]

    colors = ["#2E86AB", "#E8443A", "#2ECC71", "#F39C12", "#9B59B6"]
    fig = go.Figure()

    for idx, pr in enumerate(player_rows):
        raw_values = [pr.get(k, 0) for k in kpi_keys]
        norm_values = []
        for k, raw in zip(kpi_keys, raw_values):
            all_vals = same_group[k].dropna().tolist() if k in same_group.columns else [0]
            mi, ma = min(all_vals + [0]), max(all_vals + [0])
            if ma == mi:
                norm_values.append(50)
            else:
                norm_values.append(round((raw - mi) / (ma - mi) * 100, 1))

        norm_values.append(norm_values[0])
        labels_closed = kpi_labels + [kpi_labels[0]]

        color = colors[idx % len(colors)]
        pname = pr.get("player", f"Player {idx+1}")

        fig.add_trace(go.Scatterpolar(
            r=norm_values,
            theta=labels_closed,
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            line=dict(color=color, width=2),
            name=pname,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14), x=0.5),
        polar=dict(
            bgcolor="#1B2A4A",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(color="white", gridcolor="rgba(255,255,255,0.15)"),
        ),
        paper_bgcolor=BG_COLOR,
        legend=dict(font=dict(color="white", size=11)),
        height=500, width=550,
        margin=dict(l=60, r=60, t=60, b=40),
    )

    return fig
