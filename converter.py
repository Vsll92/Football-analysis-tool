"""
Converter: PerformFeeds matchevent JSON → App's Opta-style CSV.
Handles the full 250-column schema with qualifier unpacking, category enrichment,
and formation/position extraction.
"""

import json
import os
import re
import logging
import pandas as pd
import numpy as np
from .config import (
    EVENT_TYPE_MAP, QUALIFIER_ID_MAP, EVENT_CATEGORY_MAP,
    DEFAULT_CATEGORY, COMPETITIONS,
)

logger = logging.getLogger(__name__)

# Formation code mapping (same as app's config.py)
FORMATION_MAP = {
    2: 442, 3: 41212, 4: 433, 5: 451, 6: 4411, 7: 4141, 8: 4231,
    10: 532, 11: 541, 12: 352, 13: 343, 15: 4222, 16: 3511, 17: 3421,
    18: 3412, 19: 3142, 21: 4132, 23: 4312,
}

# Position mapping from Opta position strings
POSITION_MAP = {
    "Goalkeeper": "GK", "Defender": "CB", "Midfielder": "MC",
    "Forward": "CF", "Substitute": "SUB",
}


def load_json(path):
    """Load a PerformFeeds JSON file, handling common wrapper formats."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle different wrapper formats
    if isinstance(data, dict):
        # Direct matchevent response
        if "matchInfo" in data and "liveData" in data:
            return data
        # JSONP callback wrapper
        if "body" in data:
            return data["body"]
        # Array wrapper
        if "matchEventsData" in data:
            return data["matchEventsData"]
    if isinstance(data, list) and len(data) > 0:
        return data[0]

    return data


def convert_match(json_path, output_path, comp_config=None, season_config=None):
    """
    Convert a single match JSON to the app's CSV format.
    Returns dict with match metadata or None on failure.
    """
    try:
        data = load_json(json_path)
    except Exception as e:
        logger.error(f"Failed to load JSON {json_path}: {e}")
        return None

    match_info = data.get("matchInfo", {})
    live_data = data.get("liveData", {})
    events = live_data.get("event", [])

    if not events:
        logger.warning(f"No events in {json_path}")
        return None

    # ─── Extract match metadata ──────────────────────────────────────
    match_id = match_info.get("id", "")
    description = match_info.get("description", "")

    # Competition info
    competition = match_info.get("competition", {})
    competition_id = competition.get("id", comp_config.get("opta_competition_id", "") if comp_config else "")
    competition_name = competition.get("name", comp_config.get("competition_name", "") if comp_config else "")
    competition_known_name = competition.get("knownName", comp_config.get("competition_known_name", "") if comp_config else "")
    competition_code = competition.get("code", comp_config.get("competition_code", "") if comp_config else "")
    # Sponsor name may not be in JSON
    competition_sponsor_name = competition.get("sponsorName", "")

    # Contestants (teams)
    contestants = match_info.get("contestant", [])
    teams = {}
    for c in contestants:
        pos = c.get("position", "")  # "home" or "away"
        teams[pos] = {
            "id": c.get("id", ""),
            "name": c.get("name", ""),
            "code": c.get("code", c.get("shortName", "")),
        }

    home_team = teams.get("home", {})
    away_team = teams.get("away", {})

    # Venue
    venue = match_info.get("venue", {})
    venue_id = venue.get("id", "")
    venue_name = venue.get("longName", venue.get("shortName", ""))

    # Date / time
    local_date = match_info.get("localDate", match_info.get("date", ""))
    local_time = match_info.get("localTime", match_info.get("time", ""))

    # Week / matchday
    week = match_info.get("week", match_info.get("matchDay", ""))

    # Coverage
    coverage_level = match_info.get("coverageLevel", "")
    number_of_periods = match_info.get("numberOfPeriods", 2)
    period_length = match_info.get("periodLength", 45)

    # ─── Build lineup lookup (player_id → info) ─────────────────────
    lineups = live_data.get("lineUp", [])
    player_lookup = {}
    for lineup in lineups:
        team_id = lineup.get("contestantId", "")
        team_pos = "home" if team_id == home_team.get("id") else "away"
        for player in lineup.get("player", []):
            pid = player.get("playerId", "")
            pname = player.get("matchName", player.get("shortFirstName", "") + " " + player.get("shortLastName", ""))
            position = player.get("position", "")
            jersey = player.get("shirtNumber", "")
            player_lookup[pid] = {
                "name": pname.strip(),
                "position": POSITION_MAP.get(position, position),
                "jersey": jersey,
                "team_id": team_id,
                "team_position": team_pos,
            }

    # ─── Convert events ──────────────────────────────────────────────
    rows = []
    unmapped_qualifiers = set()
    unmapped_events = set()

    for ev in events:
        type_id = ev.get("typeId", 0)
        event_name = EVENT_TYPE_MAP.get(type_id, f"Unknown_{type_id}")
        if event_name.startswith("Unknown_"):
            unmapped_events.add(type_id)
            event_name = "Unknown"

        contestant_id = ev.get("contestantId", "")
        player_id = ev.get("playerId", "")
        player_info = player_lookup.get(player_id, {})

        # Determine team
        if contestant_id == home_team.get("id"):
            team_name = home_team.get("name", "")
            team_code = home_team.get("code", "")
            team_position = "home"
        elif contestant_id == away_team.get("id"):
            team_name = away_team.get("name", "")
            team_code = away_team.get("code", "")
            team_position = "away"
        else:
            team_name = ""
            team_code = ""
            team_position = ""

        # Base row
        row = {
            "general_id": ev.get("id", ""),
            "event_id": ev.get("eventId", ev.get("id", "")),
            "event": event_name,
            "type_id": type_id,
            "period_id": ev.get("periodId", ""),
            "time_min": ev.get("timeMin", 0),
            "time_sec": ev.get("timeSec", 0),
            "contestant_id": contestant_id,
            "team_name": team_name,
            "team_code": team_code,
            "team_position": team_position,
            "player_id": player_id,
            "player_name": player_info.get("name", ""),
            "x": ev.get("x"),
            "y": ev.get("y"),
            "outcome": ev.get("outcome", ""),
            "timeStamp": ev.get("timeStamp", ""),
            "lastModified": ev.get("lastModified", ""),
            "match_id": match_id,
            "coverage_level": coverage_level,
            "local_date": local_date,
            "local_time": local_time,
            "week": week,
            "number_of_periods": number_of_periods,
            "period_length": period_length,
            "description": description,
        }

        # ─── Unpack qualifiers ───────────────────────────────────────
        qualifiers = ev.get("qualifier", [])
        represented_parts = []
        non_represented_parts = []

        for q in qualifiers:
            qid = q.get("qualifierId", 0)
            qval = q.get("value", "")

            col_name = QUALIFIER_ID_MAP.get(qid)
            if col_name:
                # Set the column value
                if qval and qval not in ("", "N/A"):
                    row[col_name] = qval
                else:
                    row[col_name] = "Si"  # flag-type qualifier
                represented_parts.append(f"{col_name}: {qval if qval else 'Si'}")
            else:
                non_represented_parts.append(f"ID: {qid}, Value: {qval}")
                unmapped_qualifiers.add(qid)

        row["represented_qualifiers"] = "; ".join(represented_parts) if represented_parts else ""
        row["non_represented_qualifiers"] = "; ".join(non_represented_parts) if non_represented_parts else ""

        # ─── Category enrichment ─────────────────────────────────────
        macro, cat = EVENT_CATEGORY_MAP.get(event_name, DEFAULT_CATEGORY)
        row["macro_category"] = macro
        row["categorias"] = cat

        # ─── Competition metadata ────────────────────────────────────
        row["competition_id"] = competition_id
        row["competition_name"] = competition_name
        row["competition_known_name"] = competition_known_name
        row["competition_sponsor_name"] = competition_sponsor_name
        row["competition_code"] = competition_code
        row["venue_id"] = venue_id
        row["venue_long_name"] = venue_name

        # ─── Formation / position from lineup ────────────────────────
        # formation column = human-readable formation number
        if "Team Formation" in row:
            try:
                form_code = int(row["Team Formation"])
                row["formation"] = FORMATION_MAP.get(form_code, form_code)
            except (ValueError, TypeError):
                row["formation"] = ""
        else:
            row["formation"] = ""

        row["position"] = player_info.get("position", "")

        rows.append(row)

    if unmapped_qualifiers:
        logger.info(f"Unmapped qualifier IDs: {sorted(unmapped_qualifiers)}")
    if unmapped_events:
        logger.warning(f"Unmapped event type IDs: {sorted(unmapped_events)}")

    # ─── Build DataFrame with full schema ────────────────────────────
    df = pd.DataFrame(rows)

    # Ensure all expected columns exist using reindex (avoids fragmentation)
    expected_cols = _get_full_column_order()
    df = df.reindex(columns=expected_cols + [c for c in df.columns if c not in expected_cols])

    # ─── Sort events ─────────────────────────────────────────────────
    df = df.sort_values(["period_id", "time_min", "time_sec", "event_id"]).reset_index(drop=True)

    # ─── Save ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "match_id": match_id,
        "home_team": home_team.get("name", ""),
        "away_team": away_team.get("name", ""),
        "week": week,
        "date": local_date,
        "events": len(df),
        "output_path": output_path,
    }


def _get_full_column_order():
    """Return the canonical column order matching the app's schema."""
    return [
        "general_id", "event_id", "event", "type_id", "period_id", "time_min", "time_sec",
        "contestant_id", "team_name", "team_code", "team_position", "player_id", "player_name",
        "x", "y", "outcome", "timeStamp", "lastModified", "match_id", "coverage_level",
        "local_date", "local_time", "week", "number_of_periods", "period_length", "description",
        "represented_qualifiers", "non_represented_qualifiers",
        "Long ball", "Cross", "Head pass", "Through ball", "Free kick taken", "Corner taken",
        "Players caught offside", "Goal disallowed", "Penalty", "Hand", "6-seconds violation",
        "Dangerous play", "Foul", "Last line", "Head",
        "Small box-centre", "Box-centre", "Out of box-centre", "35+ centre",
        "Right footed", "Other body part", "Regular play", "Fast break", "Set piece",
        "From corner", "Free kick", "Unnamed: 54", "own goal", "Assisted", "Involved",
        "Yellow Card", "Second yellow", "Red Card",
        "Referee abuse", "Argument", "Fight", "Time wasting", "Excessive celebration",
        "Crowd interaction", "Other reason", "Injury", "Tactical",
        "Player Position", "Temperature", "Conditions", "Field Pitch", "Lightings",
        "Attendance figure", "Official position", "Official Id", "Injured player id",
        "End cause", "Related event ID", "Zone", "End type", "Jersey Number",
        "Small box-right", "Small box-left", "Box-deep right", "Box-right", "Box-left",
        "Box-deep left", "Out of box-deep right", "Out of box-right", "Out of box-left",
        "Out of box-deep left", "35+ right", "35+ left",
        "Left footed", "Left", "High", "Right", "Low Left", "High Left", "Low Centre",
        "High Centre", "Low Right", "High Right", "Blocked",
        "Close Left", "Close Right", "Close High", "Close Left and High", "Close Right and High",
        "High claim", "1 on 1", "Deflected save", "Dive and deflect", "Catch", "Dive and catch",
        "Def block", "Back pass", "Corner situation", "Direct free", "Six Yard Blocked",
        "Saved Off Line", "Goal Mouth Y Coordinate", "Goal Mouth Z Coordinate",
        "Attacking Pass", "Throw In", "Volley", "Overhead", "Half Volley", "Diving Header",
        "Scramble", "Strong", "Weak", "Rising", "Dipping", "Lob", "One Bounce", "Few Bounces",
        "Swerve Left", "Swerve Right", "Swerve Moving", "Keeper Throw", "Goal Kick",
        "Direction of play", "Punch", "Team Formation", "Team Player Formation",
        "Dive", "Deflection", "Far Wide Left", "Far Wide Right",
        "Keeper Touched", "Keeper Saved", "Hit Woodwork", "Own Player",
        "Pass End X", "Pass End Y", "Deleted Event Type", "Formation slot",
        "Blocked X Coordinate", "Blocked Y Coordinate", "Not past goal line",
        "Intentional Assist", "Chipped", "Lay-off", "Launch",
        "Persistent Infringement", "Foul and Abusive Language", "Throw In set piece",
        "Encroachment", "Leaving field", "Entering field", "Spitting",
        "Professional foul", "Handling on the line", "Out of play",
        "Flick-on", "Leading to attempt", "Leading to goal", "Rescinded Card",
        "No impact on timing", "Parried safe", "Parried danger", "Fingertip",
        "Caught", "Collected", "Standing", "Diving", "Stooping", "Reaching",
        "Hands", "Feet", "Dissent", "Blocked cross", "Scored", "Saved", "Missed",
        "Player Not Visible", "From shot off target", "Off the ball foul",
        "Block by hand", "Captain", "Pull Back", "Switch of play", "Team kit",
        "GK hoof", "Gk kick from hands",
        "Referee stop", "Referee delay", "Weather problem", "Crowd trouble", "Fire",
        "Object thrown on pitch", "Spectator on pitch", "Awaiting officials decision",
        "Referee Injury", "Game end", "Assist", "Overrun", "Length", "Angle",
        "Big Chance", "Individual Play",
        "2nd related event ID", "2nd assisted", "2nd assist",
        "Players on both posts", "Player on near post", "Player on far post",
        "No players on posts", "Inswinger", "Outswinger", "Straight",
        "Suspended", "Resume", "Own shot blocked", "Post match complete",
        "competition_id", "competition_name", "competition_known_name",
        "competition_sponsor_name", "competition_code",
        "venue_id", "venue_long_name",
        "formation", "position", "macro_category", "categorias",
    ]


def build_output_filename(match_meta):
    """Build the app's expected filename: {week}_{Home}_{Away}_{matchId}_with_categories.csv"""
    week = match_meta.get("week", 0)
    home = match_meta.get("home_team", "Unknown").replace(" ", "_")
    away = match_meta.get("away_team", "Unknown").replace(" ", "_")
    mid = match_meta.get("match_id", "unknown")
    # Clean special chars from team names for filesystem safety
    home = re.sub(r'[^\w\-]', '_', home)
    away = re.sub(r'[^\w\-]', '_', away)
    return f"{week}_{home}_{away}_{mid}_with_categories.csv"
