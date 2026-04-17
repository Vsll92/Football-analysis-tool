"""
Metrics Engine — computes all tactical metrics from Opta eventing data.

Each function returns a dict or DataFrame of metrics for a given team
across a set of matches.
"""

import numpy as np
import pandas as pd
from config import (
    THIRDS_X, LANES_Y, LANES_Y_3, BOX_X, BOX_Y, ZONE_14_X, ZONE_14_Y,
    SHOT_EVENTS, FORMATION_MAP, get_position_group,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_third(x):
    if pd.isna(x): return None
    if x <= 33.3: return "defensive"
    if x <= 66.6: return "middle"
    return "final"


def _get_lane_5(y):
    if pd.isna(y): return None
    if y >= 66.6: return "left"
    if y >= 55: return "left_hs"
    if y >= 36.8: return "center"
    if y >= 21.1: return "right_hs"
    return "right"


def _get_lane_3(y):
    if pd.isna(y): return None
    if y >= 66.6: return "left"
    if y >= 33.3: return "center"
    return "right"


def _in_box(x, y):
    return (x >= BOX_X[0]) & (y >= BOX_Y[0]) & (y <= BOX_Y[1])


def _in_zone14(x, y):
    return (x >= ZONE_14_X[0]) & (x < ZONE_14_X[1]) & (y >= ZONE_14_Y[0]) & (y <= ZONE_14_Y[1])


def _is_progressive_pass(x_start, x_end, y_start=None, y_end=None):
    """A pass is progressive if it moves ≥10% toward opponent goal in x."""
    if pd.isna(x_start) or pd.isna(x_end):
        return False
    return (x_end - x_start) >= 10.0


def _safe_per90(value, num_matches):
    if num_matches == 0:
        return 0
    return round(value / num_matches, 2)


def _pct(num, denom):
    if denom == 0:
        return 0
    return round(100 * num / denom, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: GENERAL FEATURES / TEAM IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_general_features(team_events, opp_events, num_matches):
    """Compute general identity metrics: possession, style, directness."""
    passes = team_events[team_events["event"] == "Pass"]
    opp_passes = opp_events[opp_events["event"] == "Pass"]

    total_passes = len(passes)
    total_opp_passes = len(opp_passes)
    possession_pct = _pct(total_passes, total_passes + total_opp_passes)

    # Pass success
    pass_success = _pct(passes["outcome"].sum(), total_passes)

    # Long ball %
    long_balls = passes["is_Long ball"].sum()
    long_ball_pct = _pct(long_balls, total_passes)

    # Average pass length
    avg_pass_length = passes["Length"].mean() if len(passes) > 0 else 0

    # Forward pass % (passes that go forward in x)
    fwd_passes = passes[passes["Pass End X"] > passes["x"]]
    fwd_pass_pct = _pct(len(fwd_passes), total_passes)

    # Progressive passes
    prog_mask = passes.apply(
        lambda r: _is_progressive_pass(r["x"], r["Pass End X"]), axis=1
    )
    progressive_passes = prog_mask.sum()

    # Width: % of passes in wide lanes
    wide_mask = passes["y"].apply(lambda v: v >= 66.6 or v <= 33.3 if not pd.isna(v) else False)
    width_pct = _pct(wide_mask.sum(), total_passes)

    # Cross count
    crosses = passes["is_Cross"].sum()

    # Through balls
    through_balls = passes["is_Through ball"].sum()

    # Switch of play
    switches = passes["is_Switch of play"].sum()

    # Shots
    shots = team_events[team_events["event"].isin(SHOT_EVENTS)]
    goals = team_events[team_events["event"] == "Goal"]
    total_xg = shots["xg"].sum() if "xg" in shots.columns else 0

    # ─── Defensive identity ─────────────────────────────────────────────
    opp_pass_successful = opp_passes[opp_passes["outcome"] == 1]
    team_def_actions = team_events[
        team_events["event"].isin(["Tackle", "Interception", "Foul"])
    ]
    # PPDA = opp successful passes / team defensive actions in opp half
    opp_passes_opp_half = opp_pass_successful[opp_pass_successful["x"] >= 33.3]
    team_def_opp_half = team_def_actions[team_def_actions["x"] <= 66.6]
    ppda = round(len(opp_passes_opp_half) / max(len(team_def_opp_half), 1), 2)

    # Defensive action height
    def_actions_all = team_events[
        team_events["event"].isin(["Tackle", "Interception", "Ball recovery", "Clearance"])
    ]
    def_height = round(100 - def_actions_all["x"].mean(), 1) if len(def_actions_all) > 0 else 50

    # High regains (in opponent third, x <= 33.3 for defending team)
    high_recoveries = team_events[
        (team_events["event"] == "Ball recovery") & (team_events["x"] >= 66.6)
    ]
    high_regain_pct = _pct(
        len(high_recoveries),
        len(team_events[team_events["event"] == "Ball recovery"])
    )

    return {
        "possession_pct": possession_pct,
        "passes_per_match": _safe_per90(total_passes, num_matches),
        "pass_success_pct": pass_success,
        "long_ball_pct": long_ball_pct,
        "avg_pass_length": round(avg_pass_length, 1),
        "fwd_pass_pct": fwd_pass_pct,
        "progressive_passes_pm": _safe_per90(progressive_passes, num_matches),
        "width_pct": width_pct,
        "crosses_pm": _safe_per90(crosses, num_matches),
        "through_balls_pm": _safe_per90(through_balls, num_matches),
        "switches_pm": _safe_per90(switches, num_matches),
        "shots_pm": _safe_per90(len(shots), num_matches),
        "goals_pm": _safe_per90(len(goals), num_matches),
        "xg_pm": _safe_per90(total_xg, num_matches),
        "ppda": ppda,
        "def_action_height": def_height,
        "high_regain_pct": high_regain_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OFFENSIVE PHASE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_buildup(team_events, num_matches):
    """3.1 Build-up phase analysis."""
    passes = team_events[team_events["event"] == "Pass"]

    # GK distribution
    gk_passes = passes[passes["position"] == "GK"]
    if len(gk_passes) == 0:
        gk_passes = passes[passes["Player Position"] == 1.0]

    gk_total = len(gk_passes)
    gk_short = len(gk_passes[~gk_passes["is_Long ball"] & ~gk_passes["is_Goal Kick"] & ~gk_passes["is_GK hoof"]])
    gk_long = len(gk_passes[gk_passes["is_Long ball"] | gk_passes["is_Goal Kick"] | gk_passes["is_GK hoof"]])
    gk_short_pct = _pct(gk_short, gk_total)
    gk_long_pct = _pct(gk_long, gk_total)

    # Build-up zone (passes originating in defensive third)
    buildup_passes = passes[passes["x"] <= 33.3]
    buildup_total = len(buildup_passes)

    # Lane distribution of build-up passes
    buildup_lanes = buildup_passes["y"].apply(_get_lane_3)
    lane_dist = buildup_lanes.value_counts(normalize=True).to_dict()

    # First receiver from GK
    # Can't easily track sequences in this structure, so use zone data
    buildup_success = buildup_passes[buildup_passes["Pass End X"] > 33.3]
    buildup_success_pct = _pct(len(buildup_success), buildup_total)

    # Build-up pass outcome
    buildup_pass_success = _pct(buildup_passes["outcome"].sum(), buildup_total)

    # Turnovers in build-up
    buildup_turnovers = buildup_passes[buildup_passes["outcome"] == 0]

    # CB passing
    cb_passes = passes[passes["position"] == "CB"]
    cb_prog = cb_passes[cb_passes.apply(
        lambda r: _is_progressive_pass(r["x"], r["Pass End X"]), axis=1
    )]

    return {
        "gk_short_pct": gk_short_pct,
        "gk_long_pct": gk_long_pct,
        "gk_passes_pm": _safe_per90(gk_total, num_matches),
        "buildup_passes_pm": _safe_per90(buildup_total, num_matches),
        "buildup_lane_left": round(lane_dist.get("left", 0) * 100, 1),
        "buildup_lane_center": round(lane_dist.get("center", 0) * 100, 1),
        "buildup_lane_right": round(lane_dist.get("right", 0) * 100, 1),
        "buildup_success_pct": buildup_success_pct,
        "buildup_pass_accuracy": buildup_pass_success,
        "buildup_turnovers_pm": _safe_per90(len(buildup_turnovers), num_matches),
        "cb_progressive_passes_pm": _safe_per90(len(cb_prog), num_matches),
    }


def compute_progression(team_events, num_matches):
    """3.2 Progression through middle third."""
    passes = team_events[team_events["event"] == "Pass"]

    # Progressive passes (≥10 units forward in x)
    prog_mask = passes.apply(
        lambda r: _is_progressive_pass(r["x"], r["Pass End X"]), axis=1
    )
    prog_passes = passes[prog_mask]

    # Middle-third passes
    mid_passes = passes[(passes["x"] > 33.3) & (passes["x"] <= 66.6)]

    # Line-breaking: passes from middle third into final third
    line_breaking = mid_passes[mid_passes["Pass End X"] > 66.6]

    # Progression by lane
    prog_lanes = prog_passes["y"].apply(_get_lane_3)
    prog_lane_dist = prog_lanes.value_counts(normalize=True).to_dict()

    # Progression by position group
    prog_by_pos = prog_passes.groupby("position").size().to_dict()

    # Carries (Take On events that progress)
    take_ons = team_events[team_events["event"] == "Take On"]
    successful_take_ons = take_ons[take_ons["outcome"] == 1]

    return {
        "progressive_passes_pm": _safe_per90(len(prog_passes), num_matches),
        "line_breaking_passes_pm": _safe_per90(len(line_breaking), num_matches),
        "prog_lane_left": round(prog_lane_dist.get("left", 0) * 100, 1),
        "prog_lane_center": round(prog_lane_dist.get("center", 0) * 100, 1),
        "prog_lane_right": round(prog_lane_dist.get("right", 0) * 100, 1),
        "prog_by_position": prog_by_pos,
        "take_ons_pm": _safe_per90(len(take_ons), num_matches),
        "take_on_success_pct": _pct(len(successful_take_ons), len(take_ons)),
        "mid_third_pass_accuracy": _pct(mid_passes["outcome"].sum(), len(mid_passes)),
    }


def compute_final_third(team_events, num_matches):
    """3.3 Final-third entry and chance creation."""
    passes = team_events[team_events["event"] == "Pass"]
    shots = team_events[team_events["event"].isin(SHOT_EVENTS)]
    goals = team_events[team_events["event"] == "Goal"]

    # Final-third entries: passes that start before and end in final third
    ft_entries = passes[
        (passes["x"] <= 66.6) & (passes["Pass End X"] > 66.6) & (passes["outcome"] == 1)
    ]
    ft_entries_pm = _safe_per90(len(ft_entries), num_matches)

    # Box entries
    box_entries = passes[
        (passes["outcome"] == 1) &
        (passes["Pass End X"] >= BOX_X[0]) &
        (passes["Pass End Y"] >= BOX_Y[0]) &
        (passes["Pass End Y"] <= BOX_Y[1])
    ]
    box_entries_pm = _safe_per90(len(box_entries), num_matches)

    # Entry lane distribution
    entry_lanes = ft_entries["y"].apply(_get_lane_3)
    entry_lane_dist = entry_lanes.value_counts(normalize=True).to_dict()

    # Entry method
    entry_cross = ft_entries["is_Cross"].sum()
    entry_through = ft_entries["is_Through ball"].sum()
    entry_long = ft_entries["is_Long ball"].sum()
    entry_short = len(ft_entries) - entry_cross - entry_through - entry_long
    total_entries = max(len(ft_entries), 1)

    # Crosses from final third
    ft_passes = passes[passes["x"] > 66.6]
    crosses_ft = ft_passes["is_Cross"].sum()

    # Shots analysis
    total_shots = len(shots)
    total_xg = shots["xg"].sum() if "xg" in shots.columns else 0
    shots_on_target = len(shots[shots["event"].isin(["Goal", "Saved Shot"])])
    headers = shots["is_Head"].sum()

    # Shot zones
    shot_zones = {}
    for _, s in shots.iterrows():
        if pd.notna(s["x"]) and pd.notna(s["y"]):
            in_box = _in_box(s["x"], s["y"])
            in_z14 = _in_zone14(s["x"], s["y"])
            if in_box:
                shot_zones["box"] = shot_zones.get("box", 0) + 1
            elif in_z14:
                shot_zones["zone_14"] = shot_zones.get("zone_14", 0) + 1
            else:
                shot_zones["outside"] = shot_zones.get("outside", 0) + 1

    # Big chances
    big_chances = shots["is_Big Chance"].sum()

    return {
        "ft_entries_pm": ft_entries_pm,
        "box_entries_pm": box_entries_pm,
        "entry_lane_left": round(entry_lane_dist.get("left", 0) * 100, 1),
        "entry_lane_center": round(entry_lane_dist.get("center", 0) * 100, 1),
        "entry_lane_right": round(entry_lane_dist.get("right", 0) * 100, 1),
        "entry_cross_pct": _pct(entry_cross, total_entries),
        "entry_through_pct": _pct(entry_through, total_entries),
        "entry_long_pct": _pct(entry_long, total_entries),
        "entry_short_pct": _pct(entry_short, total_entries),
        "crosses_ft_pm": _safe_per90(crosses_ft, num_matches),
        "shots_pm": _safe_per90(total_shots, num_matches),
        "xg_pm": _safe_per90(total_xg, num_matches),
        "shots_on_target_pct": _pct(shots_on_target, total_shots),
        "header_shots_pct": _pct(headers, total_shots),
        "shot_zones": shot_zones,
        "big_chances_pm": _safe_per90(big_chances, num_matches),
        "goals_pm": _safe_per90(len(goals), num_matches),
        "xg_per_shot": round(total_xg / max(total_shots, 1), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DEFENSIVE PHASE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_defensive(team_events, opp_events, num_matches):
    """Defensive phase metrics: pressing, block, vulnerabilities."""

    # ─── Pressing ──────────────────────────────────────────────────────
    tackles = team_events[team_events["event"] == "Tackle"]
    interceptions = team_events[team_events["event"] == "Interception"]
    recoveries = team_events[team_events["event"] == "Ball recovery"]
    clearances = team_events[team_events["event"] == "Clearance"]
    fouls = team_events[team_events["event"] == "Foul"]
    aerials = team_events[team_events["event"] == "Aerial"]

    all_def = pd.concat([tackles, interceptions, recoveries, clearances])

    # Defensive action zones
    def_action_third = all_def["x"].apply(lambda v: 100 - v if pd.notna(v) else None).apply(_get_third)
    def_zone_dist = def_action_third.value_counts(normalize=True).to_dict()

    # Average defensive line height (100 - x because defending toward own goal)
    avg_def_height = round(100 - all_def["x"].mean(), 1) if len(all_def) > 0 else 50

    # High press regains
    high_press = recoveries[recoveries["x"] >= 66.6]
    mid_press = recoveries[(recoveries["x"] >= 33.3) & (recoveries["x"] < 66.6)]

    # PPDA
    opp_passes = opp_events[opp_events["event"] == "Pass"]
    opp_passes_success = opp_passes[opp_passes["outcome"] == 1]
    opp_in_own_half = opp_passes_success[opp_passes_success["x"] >= 33.3]
    team_def_opp_half = all_def[all_def["x"] >= 33.3]  # team actions in opp half (from defending perspective use >=33.3 in attacking coords, but for PPDA it's the opponent's half)
    # Correct PPDA: opponent passes allowed per our defensive action in their half
    ppda = round(len(opp_in_own_half) / max(len(team_def_opp_half), 1), 2)

    # ─── Opponent shots conceded ──────────────────────────────────────
    opp_shots = opp_events[opp_events["event"].isin(SHOT_EVENTS)]
    opp_xg = opp_shots["xg"].sum() if "xg" in opp_shots.columns else 0
    opp_goals = opp_events[opp_events["event"] == "Goal"]

    # Shots conceded by zone
    opp_shot_lanes = opp_shots["y"].apply(_get_lane_3)
    opp_shot_lane_dist = opp_shot_lanes.value_counts(normalize=True).to_dict()

    # Box entries conceded
    opp_passes_succ = opp_passes[opp_passes["outcome"] == 1]
    opp_box_entries = opp_passes_succ[
        (opp_passes_succ["Pass End X"] >= BOX_X[0]) &
        (opp_passes_succ["Pass End Y"] >= BOX_Y[0]) &
        (opp_passes_succ["Pass End Y"] <= BOX_Y[1])
    ]

    # Tackle success
    tackle_success = _pct(tackles["outcome"].sum(), len(tackles))

    # Aerial success
    aerial_success = _pct(aerials["outcome"].sum(), len(aerials))

    return {
        "ppda": ppda,
        "avg_def_height": avg_def_height,
        "tackles_pm": _safe_per90(len(tackles), num_matches),
        "tackle_success_pct": tackle_success,
        "interceptions_pm": _safe_per90(len(interceptions), num_matches),
        "recoveries_pm": _safe_per90(len(recoveries), num_matches),
        "clearances_pm": _safe_per90(len(clearances), num_matches),
        "fouls_pm": _safe_per90(len(fouls), num_matches),
        "aerials_pm": _safe_per90(len(aerials), num_matches),
        "aerial_success_pct": aerial_success,
        "high_press_recoveries_pm": _safe_per90(len(high_press), num_matches),
        "def_zone_defensive": round(def_zone_dist.get("defensive", 0) * 100, 1),
        "def_zone_middle": round(def_zone_dist.get("middle", 0) * 100, 1),
        "def_zone_final": round(def_zone_dist.get("final", 0) * 100, 1),
        "shots_conceded_pm": _safe_per90(len(opp_shots), num_matches),
        "xg_conceded_pm": _safe_per90(opp_xg, num_matches),
        "goals_conceded_pm": _safe_per90(len(opp_goals), num_matches),
        "opp_box_entries_pm": _safe_per90(len(opp_box_entries), num_matches),
        "opp_shot_lane_left": round(opp_shot_lane_dist.get("left", 0) * 100, 1),
        "opp_shot_lane_center": round(opp_shot_lane_dist.get("center", 0) * 100, 1),
        "opp_shot_lane_right": round(opp_shot_lane_dist.get("right", 0) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 & 6: TRANSITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_transitions(team_events, opp_events, combined_match, num_matches, team_name):
    """Offensive and defensive transition metrics."""

    recoveries = team_events[team_events["event"] == "Ball recovery"].copy()

    # ─── Offensive Transition ─────────────────────────────────────────
    # Recovery zones
    rec_thirds = recoveries["x"].apply(_get_third)
    rec_third_dist = rec_thirds.value_counts(normalize=True).to_dict()
    rec_lanes = recoveries["y"].apply(_get_lane_3)
    rec_lane_dist = rec_lanes.value_counts(normalize=True).to_dict()

    # Fast breaks (shots marked as fast break)
    shots = team_events[team_events["event"].isin(SHOT_EVENTS)]
    fast_break_shots = shots[shots["is_Fast break"]].copy()
    fast_break_xg = fast_break_shots["xg"].sum() if "xg" in fast_break_shots.columns else 0

    # ─── Defensive Transition ─────────────────────────────────────────
    # Turnovers: team's failed passes, dispossessions
    turnovers = team_events[
        ((team_events["event"] == "Pass") & (team_events["outcome"] == 0)) |
        (team_events["event"] == "Dispossessed")
    ]
    turnover_thirds = turnovers["x"].apply(_get_third)
    turnover_third_dist = turnover_thirds.value_counts(normalize=True).to_dict()

    # Counterpress: team recoveries within short time after losing ball
    # Approximate: recoveries in middle/final third
    counterpress_approx = recoveries[recoveries["x"] >= 50]

    # Opponent fast break shots (transition against)
    opp_shots = opp_events[opp_events["event"].isin(SHOT_EVENTS)]
    opp_fb_shots = opp_shots[opp_shots["is_Fast break"]].copy()
    opp_transition_xg = opp_fb_shots["xg"].sum() if "xg" in opp_fb_shots.columns else 0

    return {
        # Offensive transition
        "recovery_third_def": round(rec_third_dist.get("defensive", 0) * 100, 1),
        "recovery_third_mid": round(rec_third_dist.get("middle", 0) * 100, 1),
        "recovery_third_final": round(rec_third_dist.get("final", 0) * 100, 1),
        "recovery_lane_left": round(rec_lane_dist.get("left", 0) * 100, 1),
        "recovery_lane_center": round(rec_lane_dist.get("center", 0) * 100, 1),
        "recovery_lane_right": round(rec_lane_dist.get("right", 0) * 100, 1),
        "fast_break_shots_pm": _safe_per90(len(fast_break_shots), num_matches),
        "fast_break_xg_pm": _safe_per90(fast_break_xg, num_matches),
        "recoveries_pm": _safe_per90(len(recoveries), num_matches),
        # Defensive transition
        "turnovers_pm": _safe_per90(len(turnovers), num_matches),
        "turnover_def_third": round(turnover_third_dist.get("defensive", 0) * 100, 1),
        "turnover_mid_third": round(turnover_third_dist.get("middle", 0) * 100, 1),
        "turnover_final_third": round(turnover_third_dist.get("final", 0) * 100, 1),
        "counterpress_recoveries_pm": _safe_per90(len(counterpress_approx), num_matches),
        "opp_transition_shots_pm": _safe_per90(len(opp_fb_shots), num_matches),
        "opp_transition_xg_pm": _safe_per90(opp_transition_xg, num_matches),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SET PIECES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_set_pieces(team_events, opp_events, num_matches):
    """Offensive and defensive set-piece analysis."""
    passes = team_events[team_events["event"] == "Pass"]
    shots = team_events[team_events["event"].isin(SHOT_EVENTS)]
    opp_shots = opp_events[opp_events["event"].isin(SHOT_EVENTS)]

    # ─── Offensive Corners ────────────────────────────────────────────
    corners = passes[passes["is_Corner taken"]]
    corner_count = len(corners)
    corner_inswing = corners["is_Inswinger"].sum()
    corner_outswing = corners["is_Outswinger"].sum()
    corner_straight = corners["is_Straight"].sum()
    corner_short = corners[corners["Length"] < 15] if "Length" in corners.columns else pd.DataFrame()

    # Shots from corners (team shots marked from_corner)
    shots_from_corner = shots[shots["is_From corner"]]
    goals_from_corner = team_events[(team_events["event"] == "Goal") & (team_events["is_From corner"])]
    xg_corners = shots_from_corner["xg"].sum() if "xg" in shots_from_corner.columns else 0

    # Corner delivery zones (Pass End X/Y)
    corner_targets = corners[corners["Pass End X"].notna()]

    # ─── Offensive Free Kicks ─────────────────────────────────────────
    fk_passes = passes[passes["is_Free kick taken"]]
    direct_fk_shots = shots[shots["is_Free kick"]]

    # ─── Defensive Set Pieces ─────────────────────────────────────────
    opp_shots_corner = opp_shots[opp_shots["is_From corner"]]
    opp_xg_corners = opp_shots_corner["xg"].sum() if "xg" in opp_shots_corner.columns else 0
    opp_shots_fk = opp_shots[opp_shots["is_Free kick"]]

    # Throw-ins
    throw_ins = passes[passes["is_Throw In"]]

    return {
        # Offensive
        "corners_pm": _safe_per90(corner_count, num_matches),
        "corner_inswing_pct": _pct(corner_inswing, max(corner_count, 1)),
        "corner_outswing_pct": _pct(corner_outswing, max(corner_count, 1)),
        "corner_straight_pct": _pct(corner_straight, max(corner_count, 1)),
        "corner_short_pct": _pct(len(corner_short), max(corner_count, 1)),
        "shots_from_corners_pm": _safe_per90(len(shots_from_corner), num_matches),
        "goals_from_corners": len(goals_from_corner),
        "xg_from_corners_pm": _safe_per90(xg_corners, num_matches),
        "free_kicks_pm": _safe_per90(len(fk_passes), num_matches),
        "direct_fk_shots_pm": _safe_per90(len(direct_fk_shots), num_matches),
        "throw_ins_pm": _safe_per90(len(throw_ins), num_matches),
        # Defensive
        "opp_shots_from_corners_pm": _safe_per90(len(opp_shots_corner), num_matches),
        "opp_xg_from_corners_pm": _safe_per90(opp_xg_corners, num_matches),
        "opp_shots_from_fk_pm": _safe_per90(len(opp_shots_fk), num_matches),
        # Delivery data for viz
        "_corner_deliveries": corner_targets[["Pass End X", "Pass End Y"]].values.tolist() if len(corner_targets) > 0 else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PLAYER REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_player_stats(team_events, num_matches):
    """Per-player stats broken down by role. Uses most common position per player."""
    # Get most common position per player (avoid duplicates from multi-position)
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
        pgrp = get_position_group(pos)
        pevents = team_events[team_events["player_name"] == pname]

        passes = pevents[pevents["event"] == "Pass"]
        shots = pevents[pevents["event"].isin(SHOT_EVENTS)]
        tackles = pevents[pevents["event"] == "Tackle"]
        interceptions = pevents[pevents["event"] == "Interception"]
        recoveries = pevents[pevents["event"] == "Ball recovery"]
        aerials = pevents[pevents["event"] == "Aerial"]
        take_ons = pevents[pevents["event"] == "Take On"]
        dispossessed = pevents[pevents["event"] == "Dispossessed"]

        total_passes = len(passes)
        prog_passes = passes[passes.apply(
            lambda r: _is_progressive_pass(r["x"], r["Pass End X"]), axis=1
        )]
        crosses = passes[passes["is_Cross"]]
        through_balls = passes[passes["is_Through ball"]]
        key_passes = passes[passes["is_Assist"] | passes["is_Intentional Assist"]]

        xg_total = shots["xg"].sum() if "xg" in shots.columns else 0
        goals = pevents[pevents["event"] == "Goal"]

        results.append({
            "player": pname,
            "position": pos,
            "pos_group": pgrp,
            "passes": total_passes,
            "pass_accuracy": _pct(passes["outcome"].sum(), total_passes),
            "progressive_passes": len(prog_passes),
            "crosses": len(crosses),
            "through_balls": len(through_balls),
            "key_passes": len(key_passes),
            "shots": len(shots),
            "goals": len(goals),
            "xg": round(xg_total, 2),
            "tackles": len(tackles),
            "tackle_success": _pct(tackles["outcome"].sum(), len(tackles)),
            "interceptions": len(interceptions),
            "recoveries": len(recoveries),
            "aerials_won": aerials["outcome"].sum() if len(aerials) > 0 else 0,
            "aerials_total": len(aerials),
            "take_ons": len(take_ons),
            "take_on_success": _pct(take_ons["outcome"].sum(), len(take_ons)),
            "dispossessed": len(dispossessed),
            "avg_x": round(pevents["x"].mean(), 1) if pevents["x"].notna().any() else 0,
            "avg_y": round(pevents["y"].mean(), 1) if pevents["y"].notna().any() else 0,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_formations(team_events, match_df, team_name, num_matches):
    """Most used formations."""
    team_matches = match_df[
        (match_df["home_team"] == team_name) | (match_df["away_team"] == team_name)
    ]
    formations = []
    for _, m in team_matches.iterrows():
        if m["home_team"] == team_name:
            formations.append(m["home_formation"])
        else:
            formations.append(m["away_formation"])

    form_counts = pd.Series(formations).value_counts()
    return form_counts.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# PASS NETWORK & HEATMAP DATA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pass_network(team_events):
    """Compute average positions and pass connections for pass network viz."""
    passes = team_events[
        (team_events["event"] == "Pass") & (team_events["outcome"] == 1)
    ].copy()

    # Average positions (exclude empty player names)
    valid = team_events[team_events["player_name"].str.len() > 0]
    avg_pos = valid.groupby("player_name").agg(
        avg_x=("x", "mean"),
        avg_y=("y", "mean"),
        events=("event", "count"),
        position=("position", "first"),
        jersey=("Jersey Number", "first"),
    ).reset_index()
    avg_pos = avg_pos[avg_pos["events"] >= 5]  # minimum involvement

    # Pass connections — need to find receiver
    # Since we don't have explicit receiver, we approximate using Related event ID
    # For now, just return average positions
    return avg_pos


def compute_action_zones(events, event_types=None):
    """Compute action density for heatmap by zone."""
    if event_types:
        events = events[events["event"].isin(event_types)]

    events = events[events["x"].notna() & events["y"].notna()]
    if len(events) == 0:
        return np.zeros((10, 10))

    # Create 10x10 grid
    heatmap = np.zeros((10, 10))
    for _, e in events.iterrows():
        xi = min(int(e["x"] / 10), 9)
        yi = min(int(e["y"] / 10), 9)
        heatmap[yi, xi] += 1

    return heatmap


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ACROSS WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_window_comparison(team_events_dict, opp_events_dict, match_counts):
    """
    Compare key metrics across match windows (last 3, 5, 10).
    team_events_dict: {"Last 3": df, "Last 5": df, ...}
    """
    comparison = {}
    for window, tevents in team_events_dict.items():
        oevents = opp_events_dict[window]
        nm = match_counts[window]
        general = compute_general_features(tevents, oevents, nm)
        ft = compute_final_third(tevents, nm)
        defense = compute_defensive(tevents, oevents, nm)
        comparison[window] = {
            "possession": general["possession_pct"],
            "passes_pm": general["passes_per_match"],
            "pass_accuracy": general["pass_success_pct"],
            "ppda": defense["ppda"],
            "shots_pm": ft["shots_pm"],
            "xg_pm": ft["xg_pm"],
            "goals_pm": ft["goals_pm"],
            "shots_conceded_pm": defense["shots_conceded_pm"],
            "xg_conceded_pm": defense["xg_conceded_pm"],
            "ft_entries_pm": ft["ft_entries_pm"],
            "box_entries_pm": ft["box_entries_pm"],
            "progressive_passes_pm": general["progressive_passes_pm"],
        }
    return pd.DataFrame(comparison).T
