"""
Configuration constants for Opponent Analysis App.
"""

# ─── Formation Code Mapping (Opta ID → human-readable) ────────────────────────
FORMATION_MAP = {
    2: "4-4-2",
    3: "4-1-2-1-2",
    4: "4-3-3",
    5: "4-5-1",
    6: "4-4-1-1",
    7: "4-1-4-1",
    8: "4-2-3-1",
    10: "5-3-2",
    11: "5-4-1",
    12: "3-5-2",
    13: "3-4-3",
    15: "4-2-2-2",
    16: "3-5-1-1",
    17: "3-4-2-1",
    18: "3-4-1-2",
    19: "3-1-4-2",
    21: "4-1-3-2",
    23: "4-3-1-2",
}

# ─── Position Groupings ───────────────────────────────────────────────────────
POSITION_GROUPS = {
    "GK": ["GK"],
    "CB": ["CB"],
    "FB/WB": ["RB", "LB", "RWB", "LWB"],
    "Pivot/DM": ["CDM", "DM"],
    "Interior/CM": ["MC", "CM"],
    "AM/10": ["CAM", "AM"],
    "Winger": ["RW", "LW", "RM", "LM"],
    "Striker": ["CF", "ST", "SS"],
}

POSITION_GROUP_ORDER = ["GK", "CB", "FB/WB", "Pivot/DM", "Interior/CM", "AM/10", "Winger", "Striker"]

# ─── Radar KPIs per Position Group ─────────────────────────────────────────────
# Each list defines which stats appear on that group's radar
RADAR_KPIS = {
    "GK": [
        ("pass_accuracy", "Pass Accuracy %"),
        ("short_dist_pct", "Short Distribution %"),
        ("long_dist_pct", "Long Distribution %"),
        ("passes_pm", "Passes / Match"),
        ("claims_pm", "Claims / Match"),
        ("saves_pm", "Saves / Match"),
    ],
    "CB": [
        ("pass_accuracy", "Pass Accuracy %"),
        ("progressive_passes_pm", "Progressive Passes / M"),
        ("tackles_pm", "Tackles / M"),
        ("interceptions_pm", "Interceptions / M"),
        ("aerials_won_pct", "Aerial Win %"),
        ("recoveries_pm", "Recoveries / M"),
        ("clearances_pm", "Clearances / M"),
        ("long_ball_pm", "Long Balls / M"),
    ],
    "FB/WB": [
        ("pass_accuracy", "Pass Accuracy %"),
        ("progressive_passes_pm", "Progressive Passes / M"),
        ("crosses_pm", "Crosses / M"),
        ("tackles_pm", "Tackles / M"),
        ("interceptions_pm", "Interceptions / M"),
        ("take_on_success", "Take-On Success %"),
        ("recoveries_pm", "Recoveries / M"),
        ("avg_x", "Avg Height (x)"),
    ],
    "Pivot/DM": [
        ("pass_accuracy", "Pass Accuracy %"),
        ("progressive_passes_pm", "Progressive Passes / M"),
        ("line_breaking_pm", "Line-Break Passes / M"),
        ("tackles_pm", "Tackles / M"),
        ("interceptions_pm", "Interceptions / M"),
        ("recoveries_pm", "Recoveries / M"),
        ("aerials_won_pct", "Aerial Win %"),
        ("passes_pm", "Passes / M"),
    ],
    "Interior/CM": [
        ("pass_accuracy", "Pass Accuracy %"),
        ("progressive_passes_pm", "Progressive Passes / M"),
        ("key_passes_pm", "Key Passes / M"),
        ("shots_pm", "Shots / M"),
        ("xg_pm", "xG / M"),
        ("tackles_pm", "Tackles / M"),
        ("take_on_success", "Take-On Success %"),
        ("passes_pm", "Passes / M"),
    ],
    "AM/10": [
        ("key_passes_pm", "Key Passes / M"),
        ("progressive_passes_pm", "Progressive Passes / M"),
        ("shots_pm", "Shots / M"),
        ("xg_pm", "xG / M"),
        ("goals_pm", "Goals / M"),
        ("through_balls_pm", "Through Balls / M"),
        ("take_on_success", "Take-On Success %"),
        ("pass_accuracy", "Pass Accuracy %"),
    ],
    "Winger": [
        ("shots_pm", "Shots / M"),
        ("xg_pm", "xG / M"),
        ("goals_pm", "Goals / M"),
        ("crosses_pm", "Crosses / M"),
        ("key_passes_pm", "Key Passes / M"),
        ("take_on_success", "Take-On Success %"),
        ("take_ons_pm", "Take-Ons / M"),
        ("progressive_carries_pm", "Progressive Carries / M"),
    ],
    "Striker": [
        ("shots_pm", "Shots / M"),
        ("xg_pm", "xG / M"),
        ("goals_pm", "Goals / M"),
        ("aerials_won_pct", "Aerial Win %"),
        ("key_passes_pm", "Key Passes / M"),
        ("take_on_success", "Take-On Success %"),
        ("xg_per_shot", "xG per Shot"),
        ("shot_on_target_pct", "Shot on Target %"),
    ],
}

# ─── Sequence End Classification ──────────────────────────────────────────────
SEQUENCE_END_TYPES = {
    "Goal": "Goal",
    "Miss": "Shot Off Target",
    "Saved Shot": "Shot On Target",
    "Foul": "Foul Won",
    "Corner Awarded": "Corner Won",
    "Out": "Out of Play",
    "Offside Pass": "Offside",
    "Dispossessed": "Dispossessed",
}

STOPPAGE_EVENTS = {
    "Start", "End", "Start delay", "End delay", "Referee Drop Ball",
    "Player Off", "Player on", "Team setp up", "Formation change",
    "Injury Time Announcement", "Collection End", "Unknown",
    "Deleted event", "Card", "Contentious referee decision",
}

def get_position_group(pos):
    if pos is None or str(pos) == "nan":
        return "Unknown"
    for group, positions in POSITION_GROUPS.items():
        if pos in positions:
            return group
    return "Unknown"


# ─── Pitch Zone Definitions (Opta 100×100 coordinate system) ──────────────────
# X: 0=own goal line, 100=opponent goal line
# Y: 0=right touchline (from TV camera), 100=left touchline

THIRDS_X = {
    "defensive": (0, 33.3),
    "middle": (33.3, 66.6),
    "final": (66.6, 100),
}

LANES_Y = {
    "left": (66.6, 100),       # left from attacking perspective
    "left_hs": (55, 66.6),     # left half-space
    "center": (36.8, 55),      # central corridor
    "right_hs": (21.1, 36.8),  # right half-space
    "right": (0, 21.1),        # right channel
}

# Wide lanes (3-lane split)
LANES_Y_3 = {
    "left": (66.6, 100),
    "center": (33.3, 66.6),
    "right": (0, 33.3),
}

# Box zones
BOX_X = (83.0, 100.0)
BOX_Y = (21.1, 78.9)

ZONE_14_X = (72, 83)
ZONE_14_Y = (30, 70)

# ─── Event Type Constants ──────────────────────────────────────────────────────
SHOT_EVENTS = ["Miss", "Goal", "Saved Shot"]
DEFENSIVE_ACTIONS = ["Tackle", "Interception", "Foul", "Clearance"]
PRESSING_ACTIONS = ["Tackle", "Interception", "Ball recovery"]

# ─── Match Window Labels ──────────────────────────────────────────────────────
MATCH_WINDOWS = {
    "Last 3": 3,
    "Last 5": 5,
    "Last 10": 10,
    "All Available": 999,
}

# ─── xG Model Features ────────────────────────────────────────────────────────
XG_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "is_header",
    "is_right_foot",
    "is_left_foot",
    "is_penalty",
    "is_direct_fk",
    "is_from_corner",
    "is_fast_break",
    "is_big_chance",
    "shot_x",
    "shot_y",
    "distance_sq",
    "angle_sq",
]

# ─── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1B2A4A",
    "secondary": "#2E86AB",
    "accent": "#E8443A",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "light": "#ECF0F1",
    "dark": "#2C3E50",
    "pitch": "#2D8C3C",
    "pitch_lines": "#FFFFFF",
    "heatmap_low": "#313695",
    "heatmap_mid": "#FFFFBF",
    "heatmap_high": "#A50026",
}

# ─── Macro Category Mapping ───────────────────────────────────────────────────
MACRO_CATEGORIES = {
    "possession": "Possession",
    "shot": "Shot",
    "defending": "Defending",
    "dribble_duel": "Duel",
    "foul_card": "Foul/Card",
    "goalkeeper": "Goalkeeper",
    "stoppage_restart": "Set Piece / Restart",
    "offside": "Offside",
    "match_admin": "Admin",
    "feed_meta": "Meta",
}
