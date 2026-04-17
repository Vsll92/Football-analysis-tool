"""
Possession Sequence Builder and Build-Up Pattern Classifier.

Builds possession chains from Opta event data, classifies build-up patterns,
and computes per-sequence statistics for interactive exploration.
"""

import pandas as pd
import numpy as np
from config import STOPPAGE_EVENTS, SEQUENCE_END_TYPES


def _safe_str(value, fallback=""):
    """Safely convert a value to string. Handles NaN, None, float, 'nan'."""
    if value is None:
        return fallback
    if isinstance(value, float) and (np.isnan(value) or pd.isna(value)):
        return fallback
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return fallback
    return s


def _safe_player(value, short=False):
    """Get a clean player name. If short=True, return surname only."""
    name = _safe_str(value, "")
    if not name:
        return ""
    if short:
        parts = name.split()
        return parts[-1] if parts else ""
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def _get_third(x):
    if pd.isna(x): return "unknown"
    if x <= 33.3: return "defensive"
    if x <= 66.6: return "middle"
    return "final"


def _get_lane(y):
    if pd.isna(y): return "unknown"
    if y >= 66.6: return "left"
    if y >= 33.3: return "center"
    return "right"


def _classify_end(events_list):
    """Classify how a sequence ended."""
    if not events_list:
        return "unknown"
    last = events_list[-1]
    ev = last["event"]

    if ev == "Goal":
        return "goal"
    if ev in ("Miss", "Saved Shot"):
        return "shot"
    if ev == "Foul":
        return "foul_won"
    if ev == "Corner Awarded":
        return "corner_won"
    if ev == "Offside Pass" or ev == "Offside provoked":
        return "offside"
    if ev == "Out":
        return "out_of_play"
    if ev == "Dispossessed":
        return "dispossessed"

    # Failed pass = turnover
    if ev == "Pass" and last["outcome"] == 0:
        return "failed_pass"
    if ev == "Take On" and last["outcome"] == 0:
        return "failed_dribble"
    if ev == "Blocked Pass":
        return "pass_blocked"

    # Check if last action was a successful event but possession changed
    if last["outcome"] == 1 and ev == "Pass":
        return "continued"  # possession didn't end here

    return "other"


def build_sequences(match_events, team_name):
    """
    Build possession sequences for a specific team from match events.

    Returns list of dicts, each containing:
    - events: list of event dicts with all columns
    - start_x, start_y, end_x, end_y
    - start_player, end_player
    - start_zone, end_zone
    - length (number of events)
    - end_type
    - passes_in_seq, successful_passes
    - reached_final_third, reached_box
    - contains_shot, contains_goal
    - match_id, period, start_time, end_time
    """
    df = match_events.sort_values(["period_id", "abs_time", "event_id"]).reset_index(drop=True)

    sequences = []
    current_events = []
    current_team = None

    for _, row in df.iterrows():
        ev = row["event"]

        # Skip stoppages
        if ev in STOPPAGE_EVENTS or pd.isna(row.get("team_name")):
            if current_events and current_team == team_name:
                sequences.append(_finalize_sequence(current_events, row.get("match_id")))
            current_events = []
            current_team = None
            continue

        # Team change = new sequence
        if row["team_name"] != current_team:
            if current_events and current_team == team_name:
                sequences.append(_finalize_sequence(current_events, row.get("match_id")))
            current_events = [row.to_dict()]
            current_team = row["team_name"]
        else:
            current_events.append(row.to_dict())

    # Final sequence
    if current_events and current_team == team_name:
        sequences.append(_finalize_sequence(current_events, match_events["match_id"].iloc[0] if len(match_events) > 0 else ""))

    return sequences


def _finalize_sequence(events_list, match_id):
    """Convert a raw event list into a structured sequence dict."""
    first = events_list[0]
    last = events_list[-1]

    # Track which zones were reached
    max_x = max(e.get("x", 0) or 0 for e in events_list)
    pass_end_xs = [e.get("Pass End X", 0) or 0 for e in events_list if e.get("event") == "Pass"]
    if pass_end_xs:
        max_x = max(max_x, max(pass_end_xs))

    # Count passes
    passes = [e for e in events_list if e.get("event") == "Pass"]
    succ_passes = [e for e in passes if e.get("outcome") == 1]

    # Shots
    shots = [e for e in events_list if e.get("event") in ("Miss", "Goal", "Saved Shot")]
    goals = [e for e in events_list if e.get("event") == "Goal"]

    # Get all player names in sequence (for the chain)
    players_chain = []
    for e in events_list:
        pname = e.get("player_name")
        if pname and isinstance(pname, str) and pname != "nan" and (not players_chain or players_chain[-1] != pname):
            players_chain.append(pname)

    # Build step details
    steps = []
    for i, e in enumerate(events_list):
        step = {
            "idx": i,
            "event": _safe_str(e.get("event"), ""),
            "player": _safe_player(e.get("player_name")),
            "position": _safe_str(e.get("position"), ""),
            "x": e.get("x"),
            "y": e.get("y"),
            "end_x": e.get("Pass End X"),
            "end_y": e.get("Pass End Y"),
            "outcome": e.get("outcome", 0),
            "time_min": e.get("time_min", 0),
            "time_sec": e.get("time_sec", 0),
            "is_long": bool(e.get("is_Long ball", False)),
            "is_cross": bool(e.get("is_Cross", False)),
            "is_through": bool(e.get("is_Through ball", False)),
            "is_switch": bool(e.get("is_Switch of play", False)),
            "zone": _safe_str(e.get("Zone"), ""),
        }
        steps.append(step)

    # Determine start player position
    start_pos = _safe_str(first.get("position"), "")
    start_player_pos = first.get("Player Position", 0)
    is_gk_start = (start_pos == "GK") or (start_player_pos == 1.0)

    return {
        "match_id": match_id,
        "period": first.get("period_id", 1),
        "start_time": f"{int(first.get('time_min', 0))}:{int(first.get('time_sec', 0)):02d}",
        "start_time_sec": first.get("abs_time", 0),
        "end_time": f"{int(last.get('time_min', 0))}:{int(last.get('time_sec', 0)):02d}",
        "start_x": first.get("x"),
        "start_y": first.get("y"),
        "end_x": last.get("x"),
        "end_y": last.get("y"),
        "start_player": _safe_player(first.get("player_name")),
        "start_position": start_pos,
        "is_gk_start": is_gk_start,
        "end_player": _safe_player(last.get("player_name")),
        "start_third": _get_third(first.get("x")),
        "start_lane": _get_lane(first.get("y")),
        "end_third": _get_third(last.get("x")),
        "end_type": _classify_end(events_list),
        "length": len(events_list),
        "passes": len(passes),
        "successful_passes": len(succ_passes),
        "reached_final_third": max_x > 66.6,
        "reached_box": max_x > 83.0,
        "contains_shot": len(shots) > 0,
        "contains_goal": len(goals) > 0,
        "players_chain": players_chain,
        "steps": steps,
        "max_x_reached": round(max_x, 1),
        "xg": sum(e.get("xg", 0) or 0 for e in events_list),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_buildup_pattern(seq):
    """
    Classify a sequence into a build-up pattern type.
    Returns a pattern label.
    """
    if seq["length"] <= 1:
        return "single_action"

    steps = seq["steps"]
    first_event = steps[0]["event"]
    first_pos = steps[0]["position"]
    is_gk = seq["is_gk_start"]

    # GK-initiated patterns
    if is_gk and first_event == "Pass":
        if steps[0].get("is_long"):
            return "gk_long_ball"
        else:
            # Short build-up from GK
            if seq["reached_final_third"]:
                return "gk_short_to_final_third"
            elif seq["length"] >= 4:
                return "gk_short_buildup"
            else:
                return "gk_short_recycle"

    # Starts from defensive third
    if seq["start_third"] == "defensive":
        if seq["reached_final_third"]:
            # Determine route
            lanes_used = set()
            for s in steps:
                if s["y"] is not None:
                    lanes_used.add(_get_lane(s["y"]))
            if len(lanes_used) >= 2 and "left" in lanes_used and "right" in lanes_used:
                return "switch_play_buildup"
            return "buildup_to_final_third"
        elif seq["length"] >= 5:
            return "patient_buildup"
        else:
            return "short_buildup"

    # Recovery-initiated (ball recovery / interception)
    if first_event in ("Ball recovery", "Interception", "Tackle"):
        if seq["start_third"] == "final":
            return "high_press_recovery"
        elif seq["start_third"] == "middle":
            if seq["contains_shot"]:
                return "mid_recovery_to_shot"
            return "mid_recovery_possession"
        else:
            return "low_recovery"

    # Middle-third starts
    if seq["start_third"] == "middle":
        if seq["reached_final_third"]:
            return "progression_to_final_third"
        return "middle_third_possession"

    # Final-third starts
    if seq["start_third"] == "final":
        if seq["contains_shot"]:
            return "final_third_chance"
        return "final_third_possession"

    return "other"


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN-LEVEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_sequences(sequences):
    """Convert list of sequence dicts to a summary DataFrame."""
    if not sequences:
        return pd.DataFrame()

    rows = []
    for seq in sequences:
        seq["pattern"] = classify_buildup_pattern(seq)
        rows.append({
            "match_id": seq["match_id"],
            "period": seq["period"],
            "start_time": seq["start_time"],
            "start_player": seq["start_player"],
            "start_position": seq["start_position"],
            "is_gk_start": seq["is_gk_start"],
            "start_third": seq["start_third"],
            "start_lane": seq["start_lane"],
            "end_third": seq["end_third"],
            "end_type": seq["end_type"],
            "pattern": seq["pattern"],
            "length": seq["length"],
            "passes": seq["passes"],
            "successful_passes": seq["successful_passes"],
            "reached_ft": seq["reached_final_third"],
            "reached_box": seq["reached_box"],
            "contains_shot": seq["contains_shot"],
            "contains_goal": seq["contains_goal"],
            "max_x": seq["max_x_reached"],
            "xg": seq["xg"],
            "players": " → ".join(str(p) for p in seq["players_chain"][:8] if p and str(p) != "nan"),
            "num_players": len(seq["players_chain"]),
        })
    return pd.DataFrame(rows)


def pattern_summary(seq_df):
    """Summarize patterns with counts and success rates."""
    if seq_df.empty:
        return pd.DataFrame()

    summary = seq_df.groupby("pattern").agg(
        count=("pattern", "size"),
        avg_length=("length", "mean"),
        avg_passes=("passes", "mean"),
        pass_accuracy=("successful_passes", lambda x: round(x.sum() / max(seq_df.loc[x.index, "passes"].sum(), 1) * 100, 1)),
        ft_reached_pct=("reached_ft", lambda x: round(x.mean() * 100, 1)),
        box_reached_pct=("reached_box", lambda x: round(x.mean() * 100, 1)),
        shot_pct=("contains_shot", lambda x: round(x.mean() * 100, 1)),
        goal_pct=("contains_goal", lambda x: round(x.mean() * 100, 1)),
        total_xg=("xg", "sum"),
    ).reset_index()

    summary = summary.sort_values("count", ascending=False)
    return summary


def end_type_summary(seq_df):
    """How sequences end."""
    if seq_df.empty:
        return pd.DataFrame()
    return seq_df["end_type"].value_counts().reset_index().rename(
        columns={"index": "end_type", "end_type": "End Type", "count": "Count"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER-LEVEL SEQUENCE STATS
# ═══════════════════════════════════════════════════════════════════════════════

def player_sequence_stats(sequences):
    """
    Compute per-player involvement in sequences:
    - How often they start a sequence
    - How often they appear in sequences reaching final third
    - Ball losses (ending a sequence with turnover)
    - Pass success within sequences
    """
    player_data = {}

    for seq in sequences:
        pattern = classify_buildup_pattern(seq)

        for i, step in enumerate(seq["steps"]):
            pname = step["player"]
            if not pname:
                continue

            if pname not in player_data:
                player_data[pname] = {
                    "player": pname,
                    "position": step["position"],
                    "sequences_involved": 0,
                    "sequences_started": 0,
                    "passes_in_seq": 0,
                    "successful_passes_in_seq": 0,
                    "progressive_passes": 0,
                    "failed_passes": 0,
                    "ball_losses": 0,
                    "in_ft_sequences": 0,
                    "in_shot_sequences": 0,
                    "in_goal_sequences": 0,
                    "tackles_won": 0,
                    "interceptions": 0,
                    "recoveries": 0,
                    "take_ons_attempted": 0,
                    "take_ons_won": 0,
                    "dispossessed": 0,
                    "shots": 0,
                    "goals": 0,
                    "xg": 0.0,
                }

            pd_entry = player_data[pname]
            pd_entry["sequences_involved"] += 1

            if i == 0:
                pd_entry["sequences_started"] += 1

            if seq["reached_final_third"]:
                pd_entry["in_ft_sequences"] += 1
            if seq["contains_shot"]:
                pd_entry["in_shot_sequences"] += 1
            if seq["contains_goal"]:
                pd_entry["in_goal_sequences"] += 1

            ev = step["event"]
            outcome = step["outcome"]

            if ev == "Pass":
                pd_entry["passes_in_seq"] += 1
                if outcome == 1:
                    pd_entry["successful_passes_in_seq"] += 1
                    # Progressive?
                    if step["end_x"] and step["x"]:
                        if (step["end_x"] - step["x"]) >= 10:
                            pd_entry["progressive_passes"] += 1
                else:
                    pd_entry["failed_passes"] += 1
                    # If last event = this is a ball loss
                    if i == len(seq["steps"]) - 1:
                        pd_entry["ball_losses"] += 1

            elif ev == "Tackle":
                if outcome == 1:
                    pd_entry["tackles_won"] += 1
            elif ev == "Interception":
                pd_entry["interceptions"] += 1
            elif ev == "Ball recovery":
                pd_entry["recoveries"] += 1
            elif ev == "Take On":
                pd_entry["take_ons_attempted"] += 1
                if outcome == 1:
                    pd_entry["take_ons_won"] += 1
            elif ev == "Dispossessed":
                pd_entry["dispossessed"] += 1
                if i == len(seq["steps"]) - 1:
                    pd_entry["ball_losses"] += 1
            elif ev in ("Miss", "Saved Shot"):
                pd_entry["shots"] += 1
            elif ev == "Goal":
                pd_entry["shots"] += 1
                pd_entry["goals"] += 1

    result = pd.DataFrame(player_data.values())
    if len(result) > 0:
        result["pass_accuracy"] = (result["successful_passes_in_seq"] /
                                   result["passes_in_seq"].replace(0, 1) * 100).round(1)
        result["take_on_pct"] = (result["take_ons_won"] /
                                 result["take_ons_attempted"].replace(0, 1) * 100).round(1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD-UP FLOW (for interactive pitch)
# ═══════════════════════════════════════════════════════════════════════════════

def get_buildup_sequences(sequences, min_length=3, pattern_filter=None,
                          start_third_filter=None, end_type_filter=None):
    """
    Filter sequences for interactive exploration.
    Returns the full sequence objects with steps for pitch animation.
    """
    filtered = []
    for seq in sequences:
        seq["pattern"] = classify_buildup_pattern(seq)

        if seq["length"] < min_length:
            continue
        if pattern_filter and seq["pattern"] not in pattern_filter:
            continue
        if start_third_filter and seq["start_third"] not in start_third_filter:
            continue
        if end_type_filter and seq["end_type"] not in end_type_filter:
            continue
        filtered.append(seq)

    return filtered


def get_all_patterns(sequences):
    """Get list of all unique pattern types found in sequences."""
    patterns = set()
    for seq in sequences:
        seq["pattern"] = classify_buildup_pattern(seq)
        patterns.add(seq["pattern"])
    return sorted(patterns)


# ─── Pattern display names ────────────────────────────────────────────────────
PATTERN_LABELS = {
    "gk_long_ball": "🧤 GK Long Ball",
    "gk_short_to_final_third": "🧤 GK Short → Final Third",
    "gk_short_buildup": "🧤 GK Short Build-up",
    "gk_short_recycle": "🧤 GK Short Recycle",
    "buildup_to_final_third": "📐 Build-up → Final Third",
    "switch_play_buildup": "🔄 Switch of Play Build-up",
    "patient_buildup": "⏳ Patient Build-up",
    "short_buildup": "📏 Short Build-up",
    "high_press_recovery": "🔴 High Press Recovery",
    "mid_recovery_to_shot": "⚡ Mid Recovery → Shot",
    "mid_recovery_possession": "🔵 Mid Recovery Possession",
    "low_recovery": "🟢 Low Recovery",
    "progression_to_final_third": "📈 Progression → Final Third",
    "middle_third_possession": "🟡 Middle Third Possession",
    "final_third_chance": "🎯 Final Third Chance",
    "final_third_possession": "🔶 Final Third Possession",
    "single_action": "1️⃣ Single Action",
    "other": "❓ Other",
}

END_TYPE_LABELS = {
    "goal": "⚽ Goal",
    "shot": "🎯 Shot (no goal)",
    "foul_won": "🟨 Foul Won",
    "corner_won": "🚩 Corner Won",
    "offside": "🚫 Offside",
    "out_of_play": "↗️ Out of Play",
    "dispossessed": "💥 Dispossessed",
    "failed_pass": "❌ Failed Pass",
    "failed_dribble": "❌ Failed Dribble",
    "pass_blocked": "🛡️ Pass Blocked",
    "continued": "➡️ Continued",
    "other": "❓ Other",
}
