"""
Manifest manager — tracks which matches have been downloaded, converted, and are ready.
Stored as JSON per competition-season.
"""

import os
import json
from datetime import datetime


def _manifest_path(data_root, comp_key, season_key):
    folder = os.path.join(data_root, "raw", f"{comp_key}_{season_key}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "manifest.json")


def load_manifest(data_root, comp_key, season_key):
    path = _manifest_path(data_root, comp_key, season_key)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"matches": {}, "last_updated": None, "version": 1}


def save_manifest(data_root, comp_key, season_key, manifest):
    manifest["last_updated"] = datetime.utcnow().isoformat()
    path = _manifest_path(data_root, comp_key, season_key)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_match_status(manifest, match_id):
    """Get status of a match: None, 'downloaded', 'converted', 'ready'."""
    return manifest["matches"].get(match_id, {}).get("status")


def set_match_status(manifest, match_id, status, **extra):
    if match_id not in manifest["matches"]:
        manifest["matches"][match_id] = {}
    manifest["matches"][match_id]["status"] = status
    manifest["matches"][match_id]["updated"] = datetime.utcnow().isoformat()
    manifest["matches"][match_id].update(extra)


def get_downloaded_match_ids(manifest):
    return {mid for mid, info in manifest["matches"].items()
            if info.get("status") in ("downloaded", "converted", "ready")}


def get_missing_match_ids(manifest, all_ids):
    """Return match IDs that haven't been downloaded yet."""
    downloaded = get_downloaded_match_ids(manifest)
    return [mid for mid in all_ids if mid not in downloaded]


def raw_json_path(data_root, comp_key, season_key, match_id):
    folder = os.path.join(data_root, "raw", f"{comp_key}_{season_key}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{match_id}.json")
