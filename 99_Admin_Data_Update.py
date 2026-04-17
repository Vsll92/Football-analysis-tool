"""
Admin Data Update — Streamlit page for managing the data pipeline.
Provides UI for discovery, capture, conversion, and status monitoring.
"""

import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Admin — Data Update", page_icon="🔧", layout="wide")

from data_pipeline.config import COMPETITIONS
from data_pipeline.manifest import load_manifest, save_manifest
from data_pipeline.runner import (
    run_full_pipeline, run_update, run_conversion,
    get_pipeline_status,
)

st.markdown("# 🔧 Admin — Data Pipeline")
st.markdown("Manage match data ingestion: discover, download, convert, and deploy.")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPETITION / SEASON SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

comp_options = {k: v["label"] for k, v in COMPETITIONS.items()}
selected_comp = st.selectbox(
    "🏆 Competition",
    list(comp_options.keys()),
    format_func=lambda k: comp_options[k],
)

comp_config = COMPETITIONS[selected_comp]
season_options = list(comp_config["seasons"].keys())
selected_season = st.selectbox("📅 Season", season_options)

season_config = comp_config["seasons"][selected_season]
data_root = "data"

# Show configured URLs
with st.expander("🔗 Configured URLs (click to verify or override)"):
    results_url = st.text_input(
        "Results page URL",
        value=season_config.get("scoresway_results_url", ""),
        key="results_url",
        help="The Scoresway results page URL. Must contain the correct season hash."
    )
    base_url = st.text_input(
        "Base URL (for match links)",
        value=season_config.get("scoresway_base_url", ""),
        key="base_url",
    )
    if results_url != season_config.get("scoresway_results_url", ""):
        season_config["scoresway_results_url"] = results_url
        st.info("⚠️ URL overridden for this session. Edit `data_pipeline/config.py` to save permanently.")
    if base_url != season_config.get("scoresway_base_url", ""):
        season_config["scoresway_base_url"] = base_url

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📊 Current Status")
status = get_pipeline_status(selected_comp, selected_season, data_root)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tracked", status["total"])
c2.metric("Ready (in app)", status["ready"])
c3.metric("Downloaded", status["downloaded"])
c4.metric("Failed", status["failed"])

if status["last_updated"]:
    st.caption(f"Last updated: {status['last_updated']}")
else:
    st.caption("No data pipeline runs yet for this competition/season.")

if status["status_breakdown"]:
    with st.expander("Detailed status breakdown"):
        for s, count in sorted(status["status_breakdown"].items()):
            st.write(f"**{s}**: {count} matches")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### ⚡ Actions")

# Log output area
log_container = st.empty()
log_messages = []


def log_callback(msg):
    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    log_container.code("\n".join(log_messages[-30:]), language="text")


headless = st.checkbox("Run browser headless", value=True,
                       help="Uncheck to see the browser during scraping (useful for debugging)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🔄 Full Initial Load")
    st.caption("Discover all completed matches, download events, and convert to app format.")
    if st.button("▶️ Run Full Pipeline", type="primary", key="full"):
        with st.spinner("Running full pipeline... this may take a while"):
            try:
                result = run_full_pipeline(
                    selected_comp, selected_season,
                    data_root=data_root,
                    headless=headless,
                    progress_callback=log_callback,
                )
                st.success(f"Pipeline complete! Discovered: {result.get('discovered', 0)}, "
                          f"Captured: {result.get('captured', 0)}, "
                          f"Converted: {result.get('converted', 0)}")
                # Force Streamlit to clear the dataset cache so the main app reloads
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())

with col2:
    st.markdown("#### 📥 Update Latest")
    st.caption("Fetch only newly completed matches since last run.")
    if st.button("▶️ Update Latest Matches", key="update"):
        with st.spinner("Updating..."):
            try:
                result = run_update(
                    selected_comp, selected_season,
                    data_root=data_root,
                    headless=headless,
                    progress_callback=log_callback,
                )
                st.success(f"Update complete! New: {result.get('captured', 0)}, "
                          f"Converted: {result.get('converted', 0)}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Update failed: {e}")
                import traceback
                st.code(traceback.format_exc())

with col3:
    st.markdown("#### 🔧 Force Reconvert")
    st.caption("Re-convert all downloaded JSON files (useful after converter updates).")
    if st.button("▶️ Force Reconvert All", key="reconvert"):
        with st.spinner("Reconverting..."):
            try:
                converted, skipped, failed = run_conversion(
                    selected_comp, selected_season,
                    data_root=data_root,
                    force_reconvert=True,
                    progress_callback=log_callback,
                )
                st.success(f"Reconversion complete! Converted: {converted}, Failed: {failed}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Reconversion failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📋 Match Manifest")

manifest = load_manifest(data_root, selected_comp, selected_season)
matches = manifest.get("matches", {})

if matches:
    import pandas as pd
    rows = []
    for mid, info in matches.items():
        rows.append({
            "Match ID": mid[:12] + "..." if len(mid) > 12 else mid,
            "Week": info.get("week", ""),
            "Home": info.get("home_team", info.get("home", "")),
            "Away": info.get("away_team", info.get("away", "")),
            "Status": info.get("status", ""),
            "CSV File": info.get("csv_file", ""),
            "Events": info.get("events", ""),
            "Updated": info.get("updated", "")[:19] if info.get("updated") else "",
            "Error": info.get("error", ""),
        })

    df = pd.DataFrame(rows).sort_values("Week", ascending=False)

    # Color-code status
    def _style_status(val):
        colors = {"ready": "background-color: #1a5c2a", "downloaded": "background-color: #1a3a5c",
                  "failed": "background-color: #5c1a1a"}
        return colors.get(val, "")

    st.dataframe(
        df.style.applymap(_style_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No matches tracked yet. Run the pipeline to start.")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FOLDER STATUS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📁 App Data Folders")

import glob
data_folders = sorted(glob.glob(os.path.join(data_root, "*")))
for folder in data_folders:
    if os.path.isdir(folder) and not folder.endswith("raw"):
        csvs = glob.glob(os.path.join(folder, "*.csv"))
        st.write(f"📂 **{os.path.basename(folder)}** — {len(csvs)} match files")
