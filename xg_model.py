"""
Expected Goals (xG) Model.

Based on a logistic regression approach inspired by StatsBomb's public methodology.
Uses shot location, angle, body part, and game situation as features.
Pre-trained coefficients are used so no training data is needed at runtime.
"""

import numpy as np
import pandas as pd


# ─── Pre-calibrated coefficients ──────────────────────────────────────────────
# These approximate a well-known open-source xG model calibrated on large
# datasets. The intercept and feature weights produce xG values consistent
# with expected ranges (penalties ~0.76, close-range headers ~0.20-0.40, etc.)

_INTERCEPT = -0.80

_COEFFICIENTS = {
    "distance_to_goal":  -0.15,
    "angle_to_goal":      2.00,
    "distance_sq":        0.0,
    "angle_sq":          -1.00,
    "is_header":         -0.40,
    "is_right_foot":      0.0,
    "is_left_foot":       0.0,
    "is_penalty":         3.00,    # boosted to ~0.76 via clamp
    "is_direct_fk":      -0.80,
    "is_from_corner":    -0.15,
    "is_fast_break":      0.25,
    "is_big_chance":      1.10,
    "shot_x_centered":   -0.02,    # more central = higher xG
}


def _sigmoid(z):
    z = np.clip(z, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


def compute_shot_features(shots_df):
    """
    Compute xG features from Opta shot event data.

    Opta coordinates: x ∈ [0, 100], y ∈ [0, 100]
    Goal is at x=100, y=50 (center of goal).
    Pitch is 105m × 68m.
    """
    df = shots_df.copy()

    # Convert to meters (Opta 100 = full pitch)
    x_m = df["x"].fillna(50) * 1.05      # 0–105m
    y_m = df["y"].fillna(50) * 0.68       # 0–68m

    # Goal center at (105, 34)
    goal_x = 105.0
    goal_y = 34.0
    goal_post_left = 34.0 - 3.66    # ~30.34m
    goal_post_right = 34.0 + 3.66   # ~37.66m

    # Distance to goal center
    dx = goal_x - x_m
    dy = goal_y - y_m
    df["distance_to_goal"] = np.sqrt(dx ** 2 + dy ** 2)

    # Angle to goal (visible goal mouth angle in radians)
    # Using the two posts
    d_left = np.sqrt((goal_x - x_m) ** 2 + (goal_post_left - y_m) ** 2)
    d_right = np.sqrt((goal_x - x_m) ** 2 + (goal_post_right - y_m) ** 2)
    goal_width = 7.32  # meters
    # Cosine rule
    cos_angle = (d_left ** 2 + d_right ** 2 - goal_width ** 2) / (2 * d_left * d_right)
    cos_angle = np.clip(cos_angle, -1, 1)
    df["angle_to_goal"] = np.arccos(cos_angle)

    df["distance_sq"] = df["distance_to_goal"] ** 2
    df["angle_sq"] = df["angle_to_goal"] ** 2

    # How central (0 = dead center, higher = wider)
    df["shot_x_centered"] = np.abs(y_m - goal_y)

    # Boolean features
    df["is_header"] = df.get("is_Head", pd.Series(False, index=df.index)).astype(float)
    df["is_right_foot"] = df.get("is_Right footed", pd.Series(False, index=df.index)).astype(float)
    df["is_left_foot"] = df.get("is_Left footed", pd.Series(False, index=df.index)).astype(float)
    df["is_penalty"] = df.get("is_Penalty", pd.Series(False, index=df.index)).astype(float)
    df["is_direct_fk"] = df.get("is_Free kick", pd.Series(False, index=df.index)).astype(float)
    df["is_from_corner"] = df.get("is_From corner", pd.Series(False, index=df.index)).astype(float)
    df["is_fast_break"] = df.get("is_Fast break", pd.Series(False, index=df.index)).astype(float)
    df["is_big_chance"] = df.get("is_Big Chance", pd.Series(False, index=df.index)).astype(float)

    return df


def calculate_xg(shots_df):
    """
    Calculate xG for each shot in the DataFrame.
    Returns the DataFrame with an 'xg' column added.
    """
    if shots_df.empty:
        shots_df["xg"] = pd.Series(dtype=float)
        return shots_df

    df = compute_shot_features(shots_df)

    # Compute log-odds
    z = np.full(len(df), _INTERCEPT)
    for feat, coef in _COEFFICIENTS.items():
        if feat in df.columns:
            z += coef * df[feat].fillna(0).values

    df["xg"] = _sigmoid(z)

    # Clamp penalties
    penalty_mask = df["is_penalty"] == 1.0
    df.loc[penalty_mask, "xg"] = 0.76

    return df


def add_xg_to_events(events_df):
    """Add xG column to events DataFrame (only for shot events)."""
    shot_mask = events_df["event"].isin(["Miss", "Goal", "Saved Shot"])
    xg_values = pd.Series(0.0, index=events_df.index)

    if shot_mask.any():
        shots = events_df.loc[shot_mask].copy()
        shots = calculate_xg(shots)
        xg_values.loc[shot_mask] = shots["xg"].values

    events_df = events_df.copy()
    events_df["xg"] = xg_values
    return events_df
