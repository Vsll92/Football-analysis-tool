"""
Data loading and preprocessing for Opta eventing data.
Handles folder scanning, CSV loading, match metadata extraction, and team filtering.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import streamlit as st
from config import FORMATION_MAP


# ─── Discover Available Leagues/Seasons ────────────────────────────────────────

def discover_datasets(data_root="data"):
    """Scan the data root for league-season folders."""
    datasets = []
    if not os.path.isdir(data_root):
        return datasets
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full):
            csvs = glob.glob(os.path.join(full, "*.csv"))
            if csvs:
                datasets.append({
                    "folder": entry,
                    "path": full,
                    "num_files": len(csvs),
                    "label": entry.replace("_", " "),
                })
    return datasets


# ─── Parse Filename ────────────────────────────────────────────────────────────

def parse_filename(filename):
    """Extract week, home team, away team, match_id from filename pattern:
       {week}_{home}_{away}_{opta_id}_with_categories.csv
    """
    base = os.path.basename(filename).replace("_with_categories.csv", "")
    parts = base.split("_")
    if len(parts) < 3:
        return None
    week = int(parts[0])
    # The opta_id is the last part — everything between first _ and last _ is teams
    opta_id = parts[-1]
    teams_str = "_".join(parts[1:-1])
    return {
        "week": week,
        "teams_raw": teams_str,
        "opta_id": opta_id,
        "filename": os.path.basename(filename),
    }


# ─── Load Single Match ────────────────────────────────────────────────────────

def load_match_csv(filepath):
    """Load a single match CSV into a DataFrame with basic cleaning."""
    df = pd.read_csv(filepath, low_memory=False)

    # Basic type fixes
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce").fillna(0).astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce").fillna(0)
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce").fillna(0)
    df["Pass End X"] = pd.to_numeric(df["Pass End X"], errors="coerce")
    df["Pass End Y"] = pd.to_numeric(df["Pass End Y"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df["Angle"] = pd.to_numeric(df["Angle"], errors="coerce")
    df["Goal Mouth Y Coordinate"] = pd.to_numeric(df["Goal Mouth Y Coordinate"], errors="coerce")
    df["Goal Mouth Z Coordinate"] = pd.to_numeric(df["Goal Mouth Z Coordinate"], errors="coerce")
    df["Jersey Number"] = pd.to_numeric(df["Jersey Number"], errors="coerce")
    df["Team Formation"] = pd.to_numeric(df["Team Formation"], errors="coerce")

    # Map formation codes
    df["formation_name"] = df["Team Formation"].map(FORMATION_MAP)

    # ─── Normalize player text fields (prevent NaN floats downstream) ─
    for col in ["player_name", "team_name", "position"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
            df[col] = df[col].replace({"nan": "", "None": "", "NaN": ""})

    # Compute absolute time for ordering
    df["abs_time"] = df["time_min"] * 60 + df["time_sec"]

    # Boolean qualifier flags (non-null = True)
    bool_flags = [
        "Long ball", "Cross", "Head pass", "Through ball", "Free kick taken",
        "Corner taken", "Head", "Right footed", "Left footed", "Other body part",
        "Regular play", "Fast break", "Set piece", "From corner", "Free kick",
        "Penalty", "Big Chance", "Assist", "Intentional Assist", "Attacking Pass",
        "Switch of play", "Throw In", "Volley", "Half Volley", "Leading to attempt",
        "Leading to goal", "Keeper Throw", "Goal Kick", "Flick-on", "Lay-off",
        "Chipped", "Blocked", "Inswinger", "Outswinger", "Straight",
        "own goal", "GK hoof",
    ]
    for flag in bool_flags:
        if flag in df.columns:
            df[f"is_{flag}"] = df[flag].notna()
        else:
            df[f"is_{flag}"] = False

    return df


# ─── Load All Matches for a Dataset ──────────────────────────────────────────

@st.cache_data(show_spinner="Loading match data…")
def load_dataset(dataset_path):
    """Load all match CSVs in a dataset folder, compute xG, return combined DataFrame + match index."""
    from xg_model import add_xg_to_events

    csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    all_dfs = []
    match_index = []

    for fpath in csv_files:
        meta = parse_filename(fpath)
        if meta is None:
            continue
        df = load_match_csv(fpath)

        # Extract actual team names from data
        teams = df["team_name"].dropna().unique().tolist()
        home_team = df.loc[df["team_position"] == "home", "team_name"].dropna().unique()
        away_team = df.loc[df["team_position"] == "away", "team_name"].dropna().unique()
        home_team = home_team[0] if len(home_team) > 0 else teams[0]
        away_team = away_team[0] if len(away_team) > 0 else (teams[1] if len(teams) > 1 else "Unknown")

        match_id = df["match_id"].dropna().unique()
        match_id = match_id[0] if len(match_id) > 0 else meta["opta_id"]
        local_date = df["local_date"].dropna().unique()
        local_date = local_date[0] if len(local_date) > 0 else "Unknown"

        # Get starting formations
        home_form = df.loc[
            (df["team_name"] == home_team) & (df["event"] == "Team setp up"), "Team Formation"
        ].dropna()
        away_form = df.loc[
            (df["team_name"] == away_team) & (df["event"] == "Team setp up"), "Team Formation"
        ].dropna()

        # Get goals
        goals_home = len(df[(df["team_name"] == home_team) & (df["event"] == "Goal") & (~df["is_own goal"])])
        goals_away = len(df[(df["team_name"] == away_team) & (df["event"] == "Goal") & (~df["is_own goal"])])
        # Add own goals the other way
        goals_home += len(df[(df["team_name"] == away_team) & (df["event"] == "Goal") & (df["is_own goal"])])
        goals_away += len(df[(df["team_name"] == home_team) & (df["event"] == "Goal") & (df["is_own goal"])])

        match_info = {
            "match_id": match_id,
            "week": meta["week"],
            "date": local_date,
            "home_team": home_team,
            "away_team": away_team,
            "score_home": goals_home,
            "score_away": goals_away,
            "home_formation": FORMATION_MAP.get(int(home_form.iloc[0]), "Unknown") if len(home_form) > 0 else "Unknown",
            "away_formation": FORMATION_MAP.get(int(away_form.iloc[0]), "Unknown") if len(away_form) > 0 else "Unknown",
            "filename": meta["filename"],
        }
        match_index.append(match_info)

        # Tag each row with match metadata
        df["_week"] = meta["week"]
        df["_date"] = local_date
        df["_home_team"] = home_team
        df["_away_team"] = away_team
        df["_match_label"] = f"W{meta['week']} {home_team} vs {away_team}"

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    match_df = pd.DataFrame(match_index)

    # Compute xG INSIDE the cache so it only runs once per dataset
    if not combined.empty:
        combined = add_xg_to_events(combined)

    return combined, match_df


# ─── Filter Data for a Team ──────────────────────────────────────────────────

def get_team_matches(match_df, team_name):
    """Get match rows where team played (home or away), sorted by week desc."""
    mask = (match_df["home_team"] == team_name) | (match_df["away_team"] == team_name)
    return match_df[mask].sort_values("week", ascending=False).reset_index(drop=True)


def get_team_events(combined_df, team_name, match_ids=None):
    """Get all events for a specific team, optionally filtered by match IDs."""
    mask = combined_df["team_name"] == team_name
    if match_ids is not None:
        mask = mask & combined_df["match_id"].isin(match_ids)
    return combined_df[mask].copy()


def get_match_events(combined_df, match_ids):
    """Get all events (both teams) for specific matches."""
    return combined_df[combined_df["match_id"].isin(match_ids)].copy()


def get_opponent_events(combined_df, team_name, match_ids=None):
    """Get opponent events in matches where team_name played."""
    if match_ids is None:
        return pd.DataFrame()
    mask = (~(combined_df["team_name"] == team_name)) & (combined_df["match_id"].isin(match_ids))
    return combined_df[mask].copy()


# ─── Build Possession Sequences ──────────────────────────────────────────────

def build_possession_sequences(match_df):
    """
    Build possession sequences from event data within a single match.
    A new sequence starts when the team changes or after a stoppage.
    """
    stoppage_events = {"Start", "End", "Start delay", "End delay", "Referee Drop Ball",
                       "Player Off", "Player on", "Team setp up", "Formation change",
                       "Injury Time Announcement", "Collection End"}

    match_df = match_df.sort_values(["period_id", "abs_time", "event_id"]).reset_index(drop=True)

    sequences = []
    seq_id = 0
    current_team = None
    seq_events = []

    for _, row in match_df.iterrows():
        if row["event"] in stoppage_events or pd.isna(row["team_name"]):
            if seq_events:
                sequences.append({"seq_id": seq_id, "team": current_team, "events": seq_events})
                seq_id += 1
                seq_events = []
                current_team = None
            continue

        if row["team_name"] != current_team:
            if seq_events:
                sequences.append({"seq_id": seq_id, "team": current_team, "events": seq_events})
                seq_id += 1
            current_team = row["team_name"]
            seq_events = [row]
        else:
            seq_events.append(row)

    if seq_events:
        sequences.append({"seq_id": seq_id, "team": current_team, "events": seq_events})

    return sequences
