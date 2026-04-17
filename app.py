"""
Opponent Analysis — Professional Football Scouting App (v2)
Built on Opta Eventing Data — Fully Interactive with Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Opponent Analysis", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

from config import FORMATION_MAP, MATCH_WINDOWS, COLORS, get_position_group, POSITION_GROUP_ORDER, RADAR_KPIS
from data_loader import (
    discover_datasets, load_dataset, get_team_matches,
    get_team_events, get_opponent_events, get_match_events,
)
from metrics_engine import (
    compute_general_features, compute_buildup, compute_progression,
    compute_final_third, compute_defensive, compute_transitions,
    compute_set_pieces, compute_player_stats, compute_formations,
    compute_pass_network, compute_window_comparison,
)
from patterns import (
    build_sequences, aggregate_sequences, pattern_summary, end_type_summary,
    player_sequence_stats, get_buildup_sequences, get_all_patterns,
    classify_buildup_pattern, PATTERN_LABELS, END_TYPE_LABELS,
)
from interactive_viz import (
    interactive_shot_map, interactive_heatmap, interactive_pass_map,
    interactive_defensive_map, interactive_avg_positions,
    visualize_sequence, visualize_sequence_set, interactive_zone_entries,
    interactive_corner_map, interactive_bar, interactive_lane_dist,
    interactive_buildup_pitch, interactive_recovery_map,
)
from player_radar import compute_player_radar_data, plot_player_radar, plot_player_comparison_radar

st.markdown("""
<style>
    .main .block-container{padding-top:.8rem;max-width:1300px}
    .mc{background:linear-gradient(135deg,#1B2A4A 0%,#2C3E50 100%);border-radius:10px;padding:14px 18px;text-align:center;border:1px solid #34495e;margin-bottom:6px}
    .mc .v{font-size:26px;font-weight:700;color:#2ECC71}
    .mc .l{font-size:11px;color:#BDC3C7;margin-top:2px}
    .sh{background:linear-gradient(90deg,#1B2A4A,#2E86AB);padding:10px 18px;border-radius:8px;margin:18px 0 12px;border-left:4px solid #E8443A}
    .sh h2{margin:0;color:white;font-size:19px}
    .eb{background:#1B2A4A;border:1px solid #2E86AB;border-radius:10px;padding:18px;color:white;margin:8px 0}
    .eb h3{color:#2ECC71;margin-top:0}
    .mr{background:#1B2A4A;border-radius:6px;padding:7px 13px;margin:3px 0;border-left:3px solid #2E86AB;font-size:13px;color:#ECF0F1}
    .ib{background:#1B2A4A;border:1px solid #34495e;border-radius:8px;padding:12px 16px;color:#BDC3C7;font-size:12px;margin:6px 0}
    .stTabs [data-baseweb="tab-list"]{gap:3px}
    .stTabs [data-baseweb="tab"]{background-color:#1B2A4A;color:white;border-radius:6px 6px 0 0;padding:7px 14px}
    .stTabs [aria-selected="true"]{background-color:#2E86AB !important;color:white !important}
</style>""", unsafe_allow_html=True)

def mc(v, l):
    return f'<div class="mc"><div class="v">{v}</div><div class="l">{l}</div></div>'
def sh(t, i="📋"):
    st.markdown(f'<div class="sh"><h2>{i} {t}</h2></div>', unsafe_allow_html=True)
def ib(txt):
    st.markdown(f'<div class="ib">ℹ️ {txt}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ Opponent Analysis")
    st.markdown("---")
    datasets = discover_datasets("data")
    if not datasets:
        st.error("No data in `data/` folder.")
        st.stop()
    ds_idx = st.selectbox("📂 League / Season", range(len(datasets)),
                          format_func=lambda i: f"{datasets[i]['label']} ({datasets[i]['num_files']} matches)")
    selected_ds = datasets[ds_idx]
    combined_df, match_df = load_dataset(selected_ds["path"])
    if combined_df.empty:
        st.error("No valid match data."); st.stop()
    all_teams = sorted(set(match_df["home_team"].tolist() + match_df["away_team"].tolist()))
    selected_team = st.selectbox("🎯 Select Opponent", all_teams)
    team_matches = get_team_matches(match_df, selected_team)
    total_available = len(team_matches)
    st.caption(f"**{total_available} matches available**")
    window_option = st.selectbox("📊 Analysis Window", ["Last 3","Last 5","Last 10","All Available"], index=1)
    window_n = MATCH_WINDOWS[window_option]
    primary_matches = team_matches.head(min(window_n, total_available))
    primary_match_ids = primary_matches["match_id"].tolist()
    st.markdown("---")
    st.markdown(f"### Matches ({window_option})")
    for _, m in primary_matches.iterrows():
        is_home = m["home_team"]==selected_team
        score = f"{m['score_home']}–{m['score_away']}"
        opp = m["away_team"] if is_home else m["home_team"]
        ha = "H" if is_home else "A"
        form = m["home_formation"] if is_home else m["away_formation"]
        st.markdown(f'<div class="mr">W{m["week"]} | {ha} | <b>{score}</b> vs {opp}<br/><small>{m["date"]} | {form}</small></div>', unsafe_allow_html=True)
    compare_windows = st.checkbox("📈 Compare Windows (3/5/10)", value=True)

num_matches = len(primary_matches)
team_events = get_team_events(combined_df, selected_team, primary_match_ids)
opp_events = get_opponent_events(combined_df, selected_team, primary_match_ids)
match_events = get_match_events(combined_df, primary_match_ids)

st.markdown(f"# 🎯 Opponent Report: **{selected_team}**")
st.caption(f"{selected_ds['label']} — {window_option} ({num_matches} matches)")

tabs = st.tabs(["📋 Summary","🏟️ Identity","⚔️ Offensive","🛡️ Defensive","⚡ Transitions","🚩 Set Pieces","🔄 Build-Up Patterns","👤 Players & Radars","🎯 Coaching"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    sh("Executive Summary","📋")
    general = compute_general_features(team_events, opp_events, num_matches)
    ft = compute_final_third(team_events, num_matches)
    defense = compute_defensive(team_events, opp_events, num_matches)
    formations = compute_formations(team_events, match_df, selected_team, num_matches)
    buildup = compute_buildup(team_events, num_matches)
    top_formation = max(formations, key=formations.get) if formations else "?"
    style_tag = "Possession" if general["possession_pct"]>=55 else ("Direct" if general["long_ball_pct"]>=20 else "Balanced")
    press_tag = "High press" if general["ppda"]<=9 else ("Medium" if general["ppda"]<=12 else "Low block")
    max_lane = max([("Left",buildup["buildup_lane_left"]),("Center",buildup["buildup_lane_center"]),("Right",buildup["buildup_lane_right"])], key=lambda x:x[1])
    c=st.columns(4); c[0].markdown(mc(top_formation,"Formation"),True); c[1].markdown(mc(f"{general['possession_pct']}%","Possession"),True); c[2].markdown(mc(f"{ft['xg_pm']}","xG / Match"),True); c[3].markdown(mc(f"{defense['xg_conceded_pm']}","xGA / Match"),True)
    c=st.columns(4); c[0].markdown(mc(style_tag,"Style"),True); c[1].markdown(mc(f"{general['ppda']}","PPDA"),True); c[2].markdown(mc(press_tag,"Defensive Style"),True); c[3].markdown(mc(f"{max_lane[0]}","Main Attack Route"),True)
    form_str = ", ".join(f"{k}: {v}x" for k,v in sorted(formations.items(), key=lambda x:-x[1])[:3])
    vuln = f"Conceding {defense['xg_conceded_pm']} xGA/m — {defense['opp_box_entries_pm']} box entries/m" if defense['xg_conceded_pm']>=1.0 else "Low xGA — compact defensive block"
    st.markdown(f"""<div class="eb"><h3>Coaching Summary</h3><ul>
    <li><b>Formation:</b> {top_formation} ({form_str})</li>
    <li><b>Style:</b> {style_tag} — {general['passes_per_match']} passes/m, {general['pass_success_pct']}% accuracy</li>
    <li><b>Main threat:</b> {max_lane[0]} channel — {ft['ft_entries_pm']} FT entries/m, {ft['shots_pm']} shots/m</li>
    <li><b>Press:</b> {press_tag} (PPDA {general['ppda']}) — def height {general['def_action_height']}</li>
    <li><b>Vulnerability:</b> {vuln}</li>
    <li><b>Output:</b> {ft['xg_pm']} xG/m, {ft['goals_pm']} goals/m</li></ul></div>""", True)
    if compare_windows and total_available>=3:
        st.markdown("### 📈 Window Comparison")
        ib("Compare tactical trends across different match windows.")
        w2c,ow,wmc={},{},{}
        for wl,wn in [("Last 3",3),("Last 5",5),("Last 10",10)]:
            wm=team_matches.head(min(wn,total_available))
            if len(wm)==0: continue
            wmids=wm["match_id"].tolist(); w2c[wl]=get_team_events(combined_df,selected_team,wmids); ow[wl]=get_opponent_events(combined_df,selected_team,wmids); wmc[wl]=len(wm)
        if len(w2c)>=2:
            comp=compute_window_comparison(w2c,ow,wmc)
            st.dataframe(comp.style.format("{:.2f}").background_gradient(cmap="YlOrRd",axis=0), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TEAM IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    sh("General Features / Team Identity","🏟️")
    general = compute_general_features(team_events, opp_events, num_matches)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### With the Ball")
        st.dataframe(pd.DataFrame({"Metric":["Possession","Passes/M","Pass Success","Forward Pass %","Long Ball %","Avg Pass Length","Prog. Passes/M","Width Usage","Crosses/M","Through Balls/M","Switches/M"],"Value":[f"{general['possession_pct']}%",general['passes_per_match'],f"{general['pass_success_pct']}%",f"{general['fwd_pass_pct']}%",f"{general['long_ball_pct']}%",f"{general['avg_pass_length']}m",general['progressive_passes_pm'],f"{general['width_pct']}%",general['crosses_pm'],general['through_balls_pm'],general['switches_pm']]}),hide_index=True,use_container_width=True)
    with c2:
        st.markdown("### Without the Ball")
        st.dataframe(pd.DataFrame({"Metric":["PPDA","Def Action Height","High Regain %","Shots/M","Goals/M","xG/M"],"Value":[general['ppda'],general['def_action_height'],f"{general['high_regain_pct']}%",general['shots_pm'],general['goals_pm'],general['xg_pm']]}),hide_index=True,use_container_width=True)
    ib("Heatmaps are interactive — zoom, pan, and hover for details.")
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(interactive_heatmap(team_events,f"{selected_team} — All Actions"),use_container_width=True)
    with c2: st.plotly_chart(interactive_heatmap(team_events[team_events["event"]=="Pass"],f"{selected_team} — Pass Zones"),use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: OFFENSIVE PHASE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    sh("Offensive Phase","⚔️")
    buildup=compute_buildup(team_events,num_matches); progression=compute_progression(team_events,num_matches); final_third=compute_final_third(team_events,num_matches)
    otabs=st.tabs(["Build-up","Progression","Final Third"])
    with otabs[0]:
        st.markdown("### 3.1 Build-up Phase"); ib("How they initiate possession from GK and CBs. The pitch below shows GK distribution arrows, first-receiver positions, and build-up exit lanes with percentages.")
        c=st.columns(3); c[0].markdown(mc(f"{buildup['gk_short_pct']}%","GK Short Dist."),True); c[1].markdown(mc(f"{buildup['gk_long_pct']}%","GK Long Dist."),True); c[2].markdown(mc(f"{buildup['buildup_success_pct']}%","Build-up → Mid Third"),True)
        c=st.columns(3); c[0].markdown(mc(f"{buildup['buildup_pass_accuracy']}%","Build-up Pass Acc."),True); c[1].markdown(mc(f"{buildup['buildup_turnovers_pm']}","Turnovers/M"),True); c[2].markdown(mc(f"{buildup['cb_progressive_passes_pm']}","CB Prog. Passes/M"),True)
        st.plotly_chart(interactive_buildup_pitch(team_events, num_matches, f"{selected_team} — Build-Up Structure"), True)
        st.plotly_chart(interactive_lane_dist(buildup["buildup_lane_left"],buildup["buildup_lane_center"],buildup["buildup_lane_right"],"Build-up Lanes"),True)
    with otabs[1]:
        st.markdown("### 3.2 Progression"); ib("How they move into advanced territory.")
        c=st.columns(3); c[0].markdown(mc(f"{progression['progressive_passes_pm']}","Prog. Passes/M"),True); c[1].markdown(mc(f"{progression['line_breaking_passes_pm']}","Line-Break/M"),True); c[2].markdown(mc(f"{progression['take_on_success_pct']}%","Take-On Success"),True)
        st.plotly_chart(interactive_lane_dist(progression["prog_lane_left"],progression["prog_lane_center"],progression["prog_lane_right"],"Progression Lanes"),True)
        st.plotly_chart(interactive_pass_map(team_events[team_events["event"]=="Pass"],f"{selected_team} — Progressive Passes",progressive_only=True),True)
        if progression["prog_by_position"]:
            st.plotly_chart(interactive_bar(progression["prog_by_position"],"Progressive Passes by Position","Count"),True)
    with otabs[2]:
        st.markdown("### 3.3 Final Third & Chances"); ib("Hover on shots for xG, body part, situation.")
        c=st.columns(4); c[0].markdown(mc(f"{final_third['ft_entries_pm']}","FT Entries/M"),True); c[1].markdown(mc(f"{final_third['box_entries_pm']}","Box Entries/M"),True); c[2].markdown(mc(f"{final_third['shots_pm']}","Shots/M"),True); c[3].markdown(mc(f"{final_third['xg_pm']}","xG/M"),True)
        c=st.columns(4); c[0].markdown(mc(f"{final_third['goals_pm']}","Goals/M"),True); c[1].markdown(mc(f"{final_third['xg_per_shot']}","xG/Shot"),True); c[2].markdown(mc(f"{final_third['shots_on_target_pct']}%","On Target %"),True); c[3].markdown(mc(f"{final_third['big_chances_pm']}","Big Chances/M"),True)
        st.plotly_chart(interactive_lane_dist(final_third["entry_lane_left"],final_third["entry_lane_center"],final_third["entry_lane_right"],"FT Entry Lanes"),True)
        st.plotly_chart(interactive_bar({"Short":final_third["entry_short_pct"],"Cross":final_third["entry_cross_pct"],"Through":final_third["entry_through_pct"],"Long":final_third["entry_long_pct"]},"FT Entry Method (%)"),True)
        shots=team_events[team_events["event"].isin(["Miss","Goal","Saved Shot"])]
        st.plotly_chart(interactive_shot_map(shots,f"{selected_team} — Shot Map"),True)
        st.plotly_chart(interactive_zone_entries(team_events[team_events["event"]=="Pass"],f"{selected_team} — FT Entries"),True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: DEFENSIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    sh("Defensive Phase","🛡️"); defense=compute_defensive(team_events,opp_events,num_matches)
    ib("Hover on markers for player name, action type, outcome.")
    c=st.columns(4); c[0].markdown(mc(f"{defense['ppda']}","PPDA"),True); c[1].markdown(mc(f"{defense['avg_def_height']}","Def Height"),True); c[2].markdown(mc(f"{defense['shots_conceded_pm']}","Shots Conceded/M"),True); c[3].markdown(mc(f"{defense['xg_conceded_pm']}","xGA/M"),True)
    c=st.columns(4); c[0].markdown(mc(f"{defense['tackles_pm']}","Tackles/M"),True); c[1].markdown(mc(f"{defense['tackle_success_pct']}%","Tackle Success"),True); c[2].markdown(mc(f"{defense['interceptions_pm']}","Interceptions/M"),True); c[3].markdown(mc(f"{defense['aerial_success_pct']}%","Aerial Success"),True)
    st.plotly_chart(interactive_bar({"Def Third":defense["def_zone_defensive"],"Mid Third":defense["def_zone_middle"],"Final Third":defense["def_zone_final"]},"Defensive Action Zones (%)"),True)
    st.plotly_chart(interactive_lane_dist(defense["opp_shot_lane_left"],defense["opp_shot_lane_center"],defense["opp_shot_lane_right"],"Shots Conceded — Lane Origin"),True)
    st.plotly_chart(interactive_defensive_map(team_events,f"{selected_team} — Defensive Actions"),True)
    opp_shots=opp_events[opp_events["event"].isin(["Miss","Goal","Saved Shot"])]
    st.plotly_chart(interactive_shot_map(opp_shots,f"Shots CONCEDED by {selected_team}"),True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: TRANSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    sh("Transitions","⚡"); trans=compute_transitions(team_events,opp_events,match_events,num_matches,selected_team)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### ⚡ Offensive Transition")
        st.dataframe(pd.DataFrame({"Metric":["Recoveries/M","Recovery Def Third","Recovery Mid Third","Recovery Final Third","Fast-Break Shots/M","Fast-Break xG/M"],"Value":[trans['recoveries_pm'],f"{trans['recovery_third_def']}%",f"{trans['recovery_third_mid']}%",f"{trans['recovery_third_final']}%",trans['fast_break_shots_pm'],trans['fast_break_xg_pm']]}),hide_index=True,use_container_width=True)
        st.plotly_chart(interactive_lane_dist(trans["recovery_lane_left"],trans["recovery_lane_center"],trans["recovery_lane_right"],"Recovery Lanes"),True)
    with c2:
        st.markdown("### 🛡️ Defensive Transition")
        st.dataframe(pd.DataFrame({"Metric":["Turnovers/M","Turnover Def Third","Turnover Mid Third","Turnover Final Third","Counter-press Rec./M","Opp Transition Shots/M","Opp Transition xG/M"],"Value":[trans['turnovers_pm'],f"{trans['turnover_def_third']}%",f"{trans['turnover_mid_third']}%",f"{trans['turnover_final_third']}%",trans['counterpress_recoveries_pm'],trans['opp_transition_shots_pm'],trans['opp_transition_xg_pm']]}),hide_index=True,use_container_width=True)
    st.plotly_chart(interactive_recovery_map(team_events, f"{selected_team} — Recovery Zones"), True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: SET PIECES
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    sh("Set Pieces","🚩"); sp=compute_set_pieces(team_events,opp_events,num_matches)
    ib("Corner map: inswing=green, outswing=red, straight=orange. Hover for taker and outcome.")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### Offensive")
        st.dataframe(pd.DataFrame({"Metric":["Corners/M","Inswing %","Outswing %","Short Corner %","Shots from Corners/M","Goals from Corners","xG from Corners/M","Free Kicks/M","Direct FK Shots/M"],"Value":[sp['corners_pm'],f"{sp['corner_inswing_pct']}%",f"{sp['corner_outswing_pct']}%",f"{sp['corner_short_pct']}%",sp['shots_from_corners_pm'],sp['goals_from_corners'],sp['xg_from_corners_pm'],sp['free_kicks_pm'],sp['direct_fk_shots_pm']]}),hide_index=True,use_container_width=True)
    with c2:
        st.markdown("### Defensive")
        st.dataframe(pd.DataFrame({"Metric":["Opp Shots from Corners/M","Opp xG from Corners/M","Opp Shots from FK/M"],"Value":[sp['opp_shots_from_corners_pm'],sp['opp_xg_from_corners_pm'],sp['opp_shots_from_fk_pm']]}),hide_index=True,use_container_width=True)
    corners=team_events[(team_events["event"]=="Pass")&(team_events["is_Corner taken"])]
    if len(corners)>0:
        st.plotly_chart(interactive_corner_map(corners,f"{selected_team} — Corner Deliveries"),True)
        st.markdown("#### Corner Takers")
        takers=corners[corners["player_name"].str.len()>0].groupby("player_name").agg(taken=("event","size"),success=("outcome","sum"),inswing=("is_Inswinger","sum"),outswing=("is_Outswinger","sum")).reset_index().sort_values("taken",ascending=False)
        takers["acc %"]=(takers["success"]/takers["taken"]*100).round(1)
        takers.columns=["Player","Taken","Successful","Inswing","Outswing","Acc %"]
        st.dataframe(takers,hide_index=True,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: BUILD-UP PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    sh("Build-Up Patterns & Sequence Analysis","🔄")
    ib("Every possession sequence broken down: how they start, progress, and end. Use filters to isolate patterns and explore individual sequences step-by-step.")
    all_sequences=[]
    for mid in primary_match_ids:
        me=combined_df[combined_df["match_id"]==mid]
        seqs=build_sequences(me,selected_team)
        all_sequences.extend(seqs)
    seq_df=aggregate_sequences(all_sequences)
    if seq_df.empty:
        st.warning("No sequences found.")
    else:
        st.markdown("### Pattern Summary")
        psummary=pattern_summary(seq_df)
        psummary["Pattern"]=psummary["pattern"].map(PATTERN_LABELS).fillna(psummary["pattern"])
        pcols=["Pattern","count","avg_length","avg_passes","pass_accuracy","ft_reached_pct","box_reached_pct","shot_pct","goal_pct","total_xg"]
        pdisp=psummary[pcols].copy()
        pdisp.columns=["Pattern","Count","Avg Len","Avg Passes","Pass Acc %","→FT %","→Box %","→Shot %","→Goal %","Total xG"]
        st.dataframe(pdisp.style.format({"Avg Len":"{:.1f}","Avg Passes":"{:.1f}","Total xG":"{:.2f}"}).background_gradient(subset=["→FT %","→Box %"],cmap="Greens"),hide_index=True,use_container_width=True)

        st.markdown("### How Sequences End")
        end_counts=seq_df["end_type"].value_counts()
        end_labels={k:END_TYPE_LABELS.get(k,k) for k in end_counts.index}
        st.plotly_chart(interactive_bar(dict(zip([end_labels[k] for k in end_counts.index],end_counts.values)),"Sequence End Types","Count"),True)

        st.markdown("### Player Stats in Possession Sequences")
        ib("Who passes most, loses the ball most, drives sequences to final third. Key for identifying press targets and tight-marking candidates.")
        pss=player_sequence_stats(all_sequences)
        if not pss.empty:
            pss_d=pss[pss["player"].str.len() > 0].sort_values("passes_in_seq",ascending=False)
            st.dataframe(pss_d[["player","position","sequences_involved","sequences_started","passes_in_seq","successful_passes_in_seq","pass_accuracy","failed_passes","ball_losses","progressive_passes","in_ft_sequences","in_shot_sequences","recoveries","tackles_won","interceptions","take_ons_attempted","take_ons_won","dispossessed","shots","goals"]].rename(columns={"player":"Player","position":"Pos","sequences_involved":"Seq","sequences_started":"Started","passes_in_seq":"Passes","successful_passes_in_seq":"Succ","pass_accuracy":"Acc %","failed_passes":"Failed","ball_losses":"Losses","progressive_passes":"Prog","in_ft_sequences":"In FT","in_shot_sequences":"In Shot","recoveries":"Rec","tackles_won":"Tkl","interceptions":"Int","take_ons_attempted":"TO","take_ons_won":"TO Won","dispossessed":"Disp","shots":"Shots","goals":"Goals"}).style.background_gradient(subset=["Losses"],cmap="Reds").background_gradient(subset=["Prog"],cmap="Greens"),hide_index=True,use_container_width=True)
            c1,c2,c3=st.columns(3)
            with c1:
                st.markdown("#### 🔑 Most Passes")
                st.dataframe(pss_d[["player","passes_in_seq","pass_accuracy"]].head(8).rename(columns={"player":"Player","passes_in_seq":"Passes","pass_accuracy":"Acc %"}),hide_index=True,use_container_width=True)
            with c2:
                st.markdown("#### ⚠️ Most Ball Losses")
                st.dataframe(pss_d.sort_values("ball_losses",ascending=False)[["player","ball_losses","failed_passes"]].head(8).rename(columns={"player":"Player","ball_losses":"Losses","failed_passes":"Failed"}),hide_index=True,use_container_width=True)
            with c3:
                st.markdown("#### 📈 Most Prog. Passes")
                st.dataframe(pss_d.sort_values("progressive_passes",ascending=False)[["player","progressive_passes","in_ft_sequences"]].head(8).rename(columns={"player":"Player","progressive_passes":"Prog","in_ft_sequences":"In FT"}),hide_index=True,use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Interactive Sequence Explorer")
        ib("Filter by pattern type, start zone, and end type. Then pick a sequence to view step-by-step on the pitch with player names, pass arrows, and event markers.")
        avail_patterns=get_all_patterns(all_sequences)
        pattern_options=["All"]+[PATTERN_LABELS.get(p,p) for p in avail_patterns]
        pattern_keys=["All"]+avail_patterns
        fc1,fc2,fc3,fc4=st.columns(4)
        with fc1: sel_pat_label=st.selectbox("Pattern Type",pattern_options); sel_pat=pattern_keys[pattern_options.index(sel_pat_label)]
        with fc2: sel_start=st.selectbox("Start Zone",["All","defensive","middle","final"])
        with fc3: sel_end=st.selectbox("End Type",["All","goal","shot","failed_pass","dispossessed","foul_won","corner_won","out_of_play"])
        with fc4: min_len=st.slider("Min Length",1,20,3)
        pf=[sel_pat] if sel_pat!="All" else None; sf=[sel_start] if sel_start!="All" else None; ef=[sel_end] if sel_end!="All" else None
        filtered=get_buildup_sequences(all_sequences,min_length=min_len,pattern_filter=pf,start_third_filter=sf,end_type_filter=ef)
        st.caption(f"**{len(filtered)} sequences match**")
        if filtered:
            st.markdown("#### All Matching Sequences (overlay)")
            st.plotly_chart(visualize_sequence_set(filtered,f"Filtered: {sel_pat_label} ({len(filtered)})"),True)
            st.markdown("#### Step-by-Step Viewer")
            seq_labels=[f"#{i+1} | {s['start_time']} | {s['start_player'] or '?'}→{s['end_player'] or '?'} | {PATTERN_LABELS.get(s.get('pattern',''),s.get('pattern',''))} | {END_TYPE_LABELS.get(s['end_type'],s['end_type'])} | {s['length']}steps" for i,s in enumerate(filtered)]
            sel_idx=st.selectbox("Pick a sequence",range(len(seq_labels)),format_func=lambda i:seq_labels[i])
            chosen=filtered[sel_idx]
            st.plotly_chart(visualize_sequence(chosen,f"Sequence #{sel_idx+1}: {chosen['start_player'] or '?'} → {chosen['end_type']}"),True)
            step_rows=[]
            for s in chosen["steps"]:
                tags=[]
                if s.get("is_long"):tags.append("Long")
                if s.get("is_cross"):tags.append("Cross")
                if s.get("is_through"):tags.append("Through")
                if s.get("is_switch"):tags.append("Switch")
                step_rows.append({"Step":s["idx"]+1,"Event":s["event"],"Player":s["player"] or "?","Pos":s["position"],"X":round(s["x"],1) if s["x"] else "","Y":round(s["y"],1) if s["y"] else "","→X":round(s["end_x"],1) if s["end_x"] else "","→Y":round(s["end_y"],1) if s["end_y"] else "","Result":"✅" if s["outcome"]==1 else "❌","Tags":", ".join(tags),"Time":f"{int(s.get('time_min',0))}:{int(s.get('time_sec',0)):02d}"})
            st.dataframe(pd.DataFrame(step_rows),hide_index=True,use_container_width=True)
        else:
            st.info("No sequences match. Adjust filters.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: PLAYERS & RADARS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    sh("Player References & Radar Charts","👤")
    ib("Position-specific radar charts normalised against all squad players in the same role. 50% line = squad average.")
    radar_df=compute_player_radar_data(team_events,num_matches)
    player_stats=compute_player_stats(team_events,num_matches)
    if radar_df.empty:
        st.warning("No player data.")
    else:
        avg_pos=compute_pass_network(team_events)
        st.plotly_chart(interactive_avg_positions(avg_pos,f"{selected_team} — Average Positions"),True)
        pos_tabs=st.tabs(POSITION_GROUP_ORDER)
        dcm={"GK":["player","matches","passes_pm","pass_accuracy","short_dist_pct","long_dist_pct","saves_pm","claims_pm"],"CB":["player","matches","passes_pm","pass_accuracy","progressive_passes_pm","long_ball_pm","tackles_pm","interceptions_pm","aerials_won_pct","clearances_pm"],"FB/WB":["player","matches","passes_pm","pass_accuracy","progressive_passes_pm","crosses_pm","tackles_pm","interceptions_pm","take_on_success","avg_x"],"Pivot/DM":["player","matches","passes_pm","pass_accuracy","progressive_passes_pm","line_breaking_pm","tackles_pm","interceptions_pm","recoveries_pm"],"Interior/CM":["player","matches","passes_pm","pass_accuracy","progressive_passes_pm","key_passes_pm","shots_pm","xg_pm","tackles_pm"],"AM/10":["player","matches","key_passes_pm","progressive_passes_pm","shots_pm","goals_pm","xg_pm","through_balls_pm","take_on_success"],"Winger":["player","matches","shots_pm","goals_pm","xg_pm","crosses_pm","key_passes_pm","take_ons_pm","take_on_success"],"Striker":["player","matches","shots_pm","goals_pm","xg_pm","xg_per_shot","aerials_won_pct","key_passes_pm"]}
        for gi,grp in enumerate(POSITION_GROUP_ORDER):
            with pos_tabs[gi]:
                gp=radar_df[radar_df["pos_group"]==grp].sort_values("events",ascending=False)
                if gp.empty: st.info(f"No {grp} players."); continue
                st.markdown(f"### {grp} Players")
                dcols=[c for c in dcm.get(grp,["player","matches","passes_pm","shots_pm","goals_pm","xg_pm"]) if c in gp.columns]
                st.dataframe(gp[dcols],hide_index=True,use_container_width=True)
                if len(gp)>0:
                    st.markdown("#### Radar Charts")
                    pnames=gp["player"].tolist()
                    rc1,rc2=st.columns(2)
                    with rc1:
                        sp1=st.selectbox(f"Player 1 ({grp})",pnames,key=f"r1_{grp}")
                        r1=gp[gp["player"]==sp1].iloc[0].to_dict()
                        st.plotly_chart(plot_player_radar(r1,radar_df),use_container_width=True)
                    with rc2:
                        if len(pnames)>1:
                            sp2=st.selectbox(f"Player 2 ({grp})",pnames,index=min(1,len(pnames)-1),key=f"r2_{grp}")
                            r2=gp[gp["player"]==sp2].iloc[0].to_dict()
                            st.plotly_chart(plot_player_radar(r2,radar_df),use_container_width=True)
                    if len(pnames)>=2:
                        sc=st.multiselect(f"Compare {grp}",pnames,default=pnames[:min(3,len(pnames))],key=f"cmp_{grp}")
                        if len(sc)>=2:
                            cr=[gp[gp["player"]==p].iloc[0].to_dict() for p in sc]
                            st.plotly_chart(plot_player_comparison_radar(cr,radar_df,f"{grp} Comparison"),use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9: COACHING CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    sh("Tactical Conclusions — Coaching Actions","🎯")
    ib("Auto-generated recommendations linked to specific metrics. Player-focus items come from sequence analysis.")
    general=compute_general_features(team_events,opp_events,num_matches); buildup=compute_buildup(team_events,num_matches); progression=compute_progression(team_events,num_matches); final_third=compute_final_third(team_events,num_matches); defense=compute_defensive(team_events,opp_events,num_matches); trans=compute_transitions(team_events,opp_events,match_events,num_matches,selected_team); sp=compute_set_pieces(team_events,opp_events,num_matches)
    conclusions=[]
    if buildup["gk_short_pct"]>60: conclusions.append(("🔴 PRESS",f"GK builds short ({buildup['gk_short_pct']}%). Press GK & CBs to force long."))
    else: conclusions.append(("🔴 PRESS",f"GK goes long ({buildup['gk_long_pct']}%). Prepare second-ball situations."))
    ws="left" if buildup["buildup_lane_left"]<buildup["buildup_lane_right"] else "right"
    ss="left" if buildup["buildup_lane_left"]>buildup["buildup_lane_right"] else "right"
    conclusions.append(("🔴 PRESS",f"Force build-up toward {ws} — they prefer {ss} ({max(buildup['buildup_lane_left'],buildup['buildup_lane_right'])}%)."))
    if general["ppda"]<=9: conclusions.append(("🟡 DENY","High press team. Use switches to beat the press."))
    if progression["prog_lane_center"]>40: conclusions.append(("🟡 DENY",f"Central progression dominant ({progression['prog_lane_center']}%). Deny pivot between the lines."))
    else:
        dp="left" if progression["prog_lane_left"]>progression["prog_lane_right"] else "right"
        conclusions.append(("🟡 DENY",f"Progression mainly {dp} ({max(progression['prog_lane_left'],progression['prog_lane_right'])}%). Close that channel."))
    if defense["opp_box_entries_pm"]>8: conclusions.append(("🟢 ATTACK",f"Concede {defense['opp_box_entries_pm']} box entries/m. Attack centrally."))
    if defense["def_zone_final"]>25: conclusions.append(("🟢 ATTACK",f"Aggressive pressing ({defense['def_zone_final']}% final third). Play through for space behind."))
    if trans["turnover_mid_third"]>40: conclusions.append(("🟢 ATTACK",f"Mid-third turnovers ({trans['turnover_mid_third']}%). Counter from mid recoveries."))
    if trans["opp_transition_xg_pm"]>0.15: conclusions.append(("⚡ TRANSITION",f"Vulnerable in def transition ({trans['opp_transition_xg_pm']} xG from breaks/m). Attack quickly after winning."))
    if sp["xg_from_corners_pm"]>0.1: conclusions.append(("🚩 SET PIECE",f"Create {sp['xg_from_corners_pm']} xG from corners/m. Alert to routines."))
    if sp["opp_xg_from_corners_pm"]>0.1: conclusions.append(("🚩 SET PIECE",f"Concede {sp['opp_xg_from_corners_pm']} xG from corners/m. Target corners."))
    if all_sequences:
        pss=player_sequence_stats(all_sequences)
        pss=pss[pss["player"].str.len()>0] if not pss.empty else pss
        if not pss.empty and len(pss)>=1:
            tp=pss.sort_values("passes_in_seq",ascending=False).iloc[0]
            tl=pss.sort_values("ball_losses",ascending=False).iloc[0]
            tpr=pss.sort_values("progressive_passes",ascending=False).iloc[0]
            conclusions.append(("👤 PLAYER",f"<b>{tp['player']}</b> drives possession ({int(tp['passes_in_seq'])} seq. passes). Deny him space."))
            if tl["ball_losses"]>5: conclusions.append(("👤 PLAYER",f"<b>{tl['player']}</b> loses ball most ({int(tl['ball_losses'])}x). Press to force errors."))
            conclusions.append(("👤 PLAYER",f"<b>{tpr['player']}</b> is main progressor ({int(tpr['progressive_passes'])} prog. passes). Tight-mark."))
    cm={"🔴":"#E74C3C","🟡":"#F39C12","🟢":"#2ECC71","⚡":"#3498DB","🚩":"#9B59B6","👤":"#1ABC9C"}
    for tag,text in conclusions:
        c=cm.get(tag[:2],"#666")
        st.markdown(f'<div style="background:#1B2A4A;border-left:4px solid {c};padding:11px 16px;border-radius:6px;margin:6px 0;color:white"><b>{tag}</b>  {text}</div>',True)

st.markdown("---")
st.caption(f"Opponent Analysis Engine — {selected_ds['label']} — {num_matches} matches")
