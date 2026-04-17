# ⚽ Opponent Analysis Engine v2

Professional football opponent analysis tool built on **Opta eventing data** with **Streamlit** and **Plotly**.

Generates fully interactive scouting reports following a professional 9-section structure.

---

## Quick Start

```bash
pip install -r requirements.txt
# Copy your Opta CSVs into data/France_League_1_25-26/ (or any league-season folder)
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Report Sections (9 Tabs)

| Tab | What It Covers |
|-----|---------------|
| **📋 Executive Summary** | Formation, style tag, PPDA, xG/xGA, coaching summary, window comparison table |
| **🏟️ Team Identity** | Possession profile, directness, pressing behaviour, interactive heatmaps |
| **⚔️ Offensive Phase** | Build-up (GK distribution, lane split), progression (line-breaking, carries), final third (shot map, xG/shot, entry methods) |
| **🛡️ Defensive Phase** | PPDA, press height, tackle/aerial success, defensive action map, shots conceded map |
| **⚡ Transitions** | Recovery zones, fast-break xG, turnovers, counterpress, opponent transition threat |
| **🚩 Set Pieces** | Corner delivery map (inswing/outswing/straight), taker table, offensive & defensive xG from set plays |
| **🔄 Build-Up Patterns** | Possession sequence classification, step-by-step interactive pitch, player-level ball loss / progression tables |
| **👤 Players & Radars** | Position-specific KPI radars (8 templates: GK→Striker), percentile normalisation, multi-player comparison overlay |
| **🎯 Coaching Conclusions** | Auto-generated press/deny/attack/transition actions with specific player targets |

---

## Key Features

### Fully Interactive (Plotly)
Every chart supports hover tooltips (player name, xG, body part, pass type, outcome), zoom, pan, and legend toggling. No static images.

### Build-Up Pattern Analysis (Tab 7)
This is the core tactical intelligence feature:

- **Pattern classification** — 15+ categories: GK short build-up, GK long ball, switch of play, high press recovery, progression to final third, etc.
- **Sequence end tracking** — How each possession ends: goal, shot, failed pass, dispossessed, foul won, corner won, offside
- **Player-level sequence stats** — Who passes most, who loses the ball most, who drives play to the final third. Key for identifying press targets and tight-marking candidates.
- **Interactive sequence explorer** — Filter by pattern type, start zone, end type, and minimum length. Pick any individual sequence to view it step-by-step on the pitch with arrows, player names, and event markers.
- **Overlay view** — See all matching sequences on one pitch to identify common routes and end zones.

### Position-Specific Player Radars (Tab 8)
Eight different KPI templates tailored to each role:

| Position | Key KPIs |
|----------|----------|
| **GK** | Pass accuracy, short/long distribution %, saves, claims |
| **CB** | Progressive passes, tackles, interceptions, aerial win %, clearances, long balls |
| **FB/WB** | Progressive passes, crosses, tackles, take-on success, average height |
| **Pivot/DM** | Pass accuracy, progressive passes, line-breaking, tackles, interceptions, recoveries |
| **Interior/CM** | Progressive passes, key passes, shots, xG, tackles, take-on success |
| **AM/10** | Key passes, through balls, shots, goals, xG, take-on success |
| **Winger** | Shots, goals, xG, crosses, key passes, take-ons, progressive carries |
| **Striker** | Shots, goals, xG, xG/shot, aerial win %, shot on target % |

All values are percentile-normalised against squad players in the same role (50% = squad average).

### xG Model
Built-in StatsBomb-style logistic regression. Calibrated: mean 0.12, close-range ~0.39, long-range ~0.01, penalties clamped at 0.76.

### Multi-Window Comparison
Compare Last 3 / Last 5 / Last 10 matches to detect tactical shifts (formation changes, pressing intensity, attack route changes).

---

## Architecture

```
opponent_analysis/
├── app.py                 # Main Streamlit app (9 tabs)
├── config.py              # Formation map, zones, position groups, radar KPI templates
├── data_loader.py         # CSV loading, match indexing, team filtering
├── xg_model.py            # Expected Goals model
├── metrics_engine.py      # All tactical metric computations
├── pitch_viz.py           # Matplotlib pitch (kept for compatibility)
├── interactive_viz.py     # Plotly interactive: pitch, shot maps, heatmaps, sequence viewer
├── patterns.py            # Possession sequence builder, pattern classifier, player seq stats
├── player_radar.py        # Position-specific radar charts with percentile normalisation
├── requirements.txt
└── data/
    ├── France_League_1_25-26/
    │   ├── 1_TeamA_TeamB_{opta_id}_with_categories.csv
    │   └── ...
    └── (add more league-season folders here)
```

### Adding New Leagues / Seasons

1. Create a folder in `data/` named like `{Country}_{League}_{Season}`
2. Place all Opta match CSVs inside (same format)
3. Restart — new dataset appears in the sidebar dropdown

### CSV File Naming Convention

```
{matchweek}_{HomeTeam}_{AwayTeam}_{OptaMatchID}_with_categories.csv
```

### Required Columns

Core: `event`, `team_name`, `team_position`, `player_name`, `x`, `y`, `outcome`, `Pass End X`, `Pass End Y`, `position`, `Team Formation`, `match_id`, `local_date`, `week`, `Player Position`, `period_id`, `time_min`, `time_sec`, `event_id`

Qualifiers: `Cross`, `Long ball`, `Through ball`, `Corner taken`, `Free kick taken`, `Head`, `Right footed`, `Left footed`, `Penalty`, `Big Chance`, `Fast break`, `From corner`, `Free kick`, `Assist`, `Inswinger`, `Outswinger`, `Straight`, `Switch of play`, `Goal Kick`, `GK hoof`, `Throw In`, `own goal`
