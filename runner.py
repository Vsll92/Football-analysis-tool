"""
Pipeline runner — orchestrates discovery → capture → convert → deploy.
Called from the Streamlit admin page or CLI.
"""

import os
import logging
import time
from datetime import datetime

from .config import COMPETITIONS
from .manifest import (
    load_manifest, save_manifest, get_match_status, set_match_status,
    get_missing_match_ids, raw_json_path,
)
from .converter import convert_match, build_output_filename

logger = logging.getLogger(__name__)


def run_discovery(comp_key, season_key, data_root="data", headless=True, progress_callback=None):
    """
    Discover all completed matches for a competition-season.
    Returns list of discovered match dicts.
    """
    comp = COMPETITIONS[comp_key]
    season = comp["seasons"][season_key]

    from .scraper import create_driver, discover_matches

    _log(progress_callback, f"Starting discovery for {comp['label']} {season_key}...")

    driver = None
    try:
        driver = create_driver(headless=headless)
        matches = discover_matches(driver, season, progress_callback=progress_callback)
        _log(progress_callback, f"Discovered {len(matches)} completed matches")
        return matches

    except Exception as e:
        _log(progress_callback, f"Discovery failed: {e}")
        import traceback
        _log(progress_callback, traceback.format_exc())
        return []

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def run_capture(comp_key, season_key, discovered_matches, data_root="data",
                headless=True, progress_callback=None):
    """
    Capture PerformFeeds JSON for newly discovered matches.
    Skips already-downloaded matches.
    Returns (captured, skipped, failed) counts.
    """
    comp = COMPETITIONS[comp_key]
    manifest = load_manifest(data_root, comp_key, season_key)

    # Find matches not yet downloaded
    already = {mid for mid, info in manifest["matches"].items()
               if info.get("status") in ("downloaded", "converted", "ready")}

    to_capture = [m for m in discovered_matches
                  if m.get("match_id") and m["match_id"] not in already]

    if not to_capture:
        _log(progress_callback, "No new matches to capture")
        return 0, len(discovered_matches), 0

    _log(progress_callback, f"Capturing {len(to_capture)} new matches...")

    from .scraper import create_driver, capture_match_events, _is_driver_alive

    captured, failed = 0, 0
    driver = None
    try:
        driver = create_driver(headless=headless)

        for i, match in enumerate(to_capture):
            mid = match["match_id"]
            _log(progress_callback, f"[{i+1}/{len(to_capture)}] {match.get('label', mid)}")

            # Recreate driver if session died
            if not _is_driver_alive(driver):
                _log(progress_callback, "  ⚠️ Browser session died — restarting...")
                try:
                    driver.quit()
                except Exception:
                    pass
                time.sleep(2)
                driver = create_driver(headless=headless)

            json_path = raw_json_path(data_root, comp_key, season_key, mid)

            try:
                success = capture_match_events(driver, match, json_path, progress_callback=progress_callback)
            except Exception as e:
                _log(progress_callback, f"  Capture exception: {e}")
                success = False

            if success:
                set_match_status(manifest, mid, "downloaded",
                                 week=match.get("week"), home=match.get("home"),
                                 away=match.get("away"), score=match.get("score"),
                                 url=match.get("match_url"))
                captured += 1
            else:
                set_match_status(manifest, mid, "failed",
                                 error="capture_failed",
                                 week=match.get("week"), home=match.get("home"),
                                 away=match.get("away"))
                failed += 1

            save_manifest(data_root, comp_key, season_key, manifest)
            time.sleep(2)  # Rate limiting

    except Exception as e:
        _log(progress_callback, f"Capture error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    skipped = len(discovered_matches) - len(to_capture)
    _log(progress_callback, f"Captured: {captured}, Skipped: {skipped}, Failed: {failed}")
    return captured, skipped, failed


def run_conversion(comp_key, season_key, data_root="data",
                   force_reconvert=False, progress_callback=None):
    """
    Convert all downloaded JSON files to the app's CSV format.
    Deploys to the app's data folder.
    Returns (converted, skipped, failed) counts.
    """
    comp = COMPETITIONS[comp_key]
    season = comp["seasons"][season_key]
    manifest = load_manifest(data_root, comp_key, season_key)

    # Find matches that need conversion
    to_convert = []
    for mid, info in manifest["matches"].items():
        status = info.get("status", "")
        if force_reconvert and status in ("downloaded", "converted", "ready"):
            to_convert.append(mid)
        elif status == "downloaded":
            to_convert.append(mid)

    if not to_convert:
        _log(progress_callback, "No matches to convert")
        return 0, 0, 0

    _log(progress_callback, f"Converting {len(to_convert)} matches...")

    output_folder = os.path.join(data_root, season["data_folder"])
    os.makedirs(output_folder, exist_ok=True)

    converted, failed = 0, 0
    for i, mid in enumerate(to_convert):
        json_path = raw_json_path(data_root, comp_key, season_key, mid)
        if not os.path.exists(json_path):
            logger.warning(f"JSON not found for {mid}: {json_path}")
            set_match_status(manifest, mid, "failed", error="json_missing")
            failed += 1
            continue

        _log(progress_callback, f"[{i+1}/{len(to_convert)}] Converting {mid}...")

        try:
            result = convert_match(json_path, "/tmp/temp_convert.csv",
                                   comp_config=comp, season_config=season)
            if result:
                # Build proper filename
                filename = build_output_filename(result)
                final_path = os.path.join(output_folder, filename)
                os.rename("/tmp/temp_convert.csv", final_path)

                set_match_status(manifest, mid, "ready",
                                 csv_file=filename,
                                 events=result.get("events", 0),
                                 home_team=result.get("home_team", ""),
                                 away_team=result.get("away_team", ""),
                                 week=result.get("week", ""))
                converted += 1
            else:
                set_match_status(manifest, mid, "failed", error="conversion_failed")
                failed += 1
        except Exception as e:
            logger.error(f"Conversion error for {mid}: {e}")
            set_match_status(manifest, mid, "failed", error=str(e)[:200])
            failed += 1

    save_manifest(data_root, comp_key, season_key, manifest)

    skipped = len(manifest["matches"]) - len(to_convert)
    _log(progress_callback, f"Converted: {converted}, Failed: {failed}")
    return converted, skipped, failed


def run_full_pipeline(comp_key, season_key, data_root="data",
                      headless=True, progress_callback=None):
    """Run the complete pipeline: discover → capture → convert."""
    _log(progress_callback, f"═══ Full pipeline: {comp_key} / {season_key} ═══")

    # Step 1: Discover
    _log(progress_callback, "── Step 1: Discovery ──")
    matches = run_discovery(comp_key, season_key, data_root, headless, progress_callback)

    if not matches:
        _log(progress_callback, "No matches discovered. Stopping.")
        return {"discovered": 0, "captured": 0, "converted": 0, "failed": 0}

    # Step 2: Capture
    _log(progress_callback, "── Step 2: Capture ──")
    captured, skipped, cap_failed = run_capture(
        comp_key, season_key, matches, data_root, headless, progress_callback
    )

    # Step 3: Convert
    _log(progress_callback, "── Step 3: Conversion ──")
    converted, conv_skipped, conv_failed = run_conversion(
        comp_key, season_key, data_root, progress_callback=progress_callback
    )

    result = {
        "discovered": len(matches),
        "captured": captured,
        "capture_skipped": skipped,
        "capture_failed": cap_failed,
        "converted": converted,
        "conversion_failed": conv_failed,
    }
    _log(progress_callback, f"═══ Pipeline complete: {result} ═══")
    return result


def run_update(comp_key, season_key, data_root="data",
               headless=True, progress_callback=None):
    """
    Incremental update: only fetch newly completed matches.
    Same as full pipeline but skips already-captured matches.
    """
    return run_full_pipeline(comp_key, season_key, data_root, headless, progress_callback)


def get_pipeline_status(comp_key, season_key, data_root="data"):
    """Get current status summary for a competition-season."""
    manifest = load_manifest(data_root, comp_key, season_key)
    matches = manifest.get("matches", {})

    status_counts = {}
    for mid, info in matches.items():
        s = info.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    return {
        "total": len(matches),
        "ready": status_counts.get("ready", 0),
        "downloaded": status_counts.get("downloaded", 0),
        "failed": status_counts.get("failed", 0),
        "last_updated": manifest.get("last_updated"),
        "status_breakdown": status_counts,
    }


def _log(callback, msg):
    """Log to both logger and optional Streamlit callback."""
    logger.info(msg)
    if callback:
        callback(msg)
