"""
Selenium-based match discovery and PerformFeeds capture for Scoresway.

URL structure:
  Results:     .../ligue-1-2025-2026/{season_hash}/results
  Match stats: .../ligue-1-2025-2026/{season_hash}/match/view/{match_id}/player-stats

Workflow:
  1. Discovery: Navigate to /results page, collect all /match/view/{id} links
  2. Capture: For each match, go to /player-stats, extract event JSON from page source
"""

import os
import re
import json
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _get_webdriver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
    return webdriver, Options, By, WebDriverWait, EC, TimeoutException, StaleElementReferenceException


def create_driver(headless=True):
    """Create Chrome WebDriver with performance logging for network interception."""
    webdriver, Options, *_ = _get_webdriver()

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--max_old_space_size=4096")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

    # Enable performance logging to capture network requests
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(3)
    return driver


def _is_driver_alive(driver):
    """Check if the Chrome session is still valid."""
    try:
        _ = driver.window_handles
        return True
    except Exception:
        return False


def _dismiss_cookies(driver, By):
    """Try to dismiss cookie consent popups."""
    try:
        for selector in [
            "button[id*='accept']", "button[id*='consent']", "button[id*='agree']",
            "button[class*='accept']", "button[class*='consent']",
            "a[id*='accept']", "a[class*='accept']",
            "[data-testid*='accept']", "[data-testid*='consent']",
        ]:
            elems = driver.find_elements(By.CSS_SELECTOR, selector)
            for e in elems:
                if e.is_displayed():
                    e.click()
                    time.sleep(1)
                    return
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# DISCOVERY — find all completed matches from the /results page
# ═══════════════════════════════════════════════════════════════════════════════

def discover_matches(driver, season_config, progress_callback=None):
    """
    Discover all completed matches from the Scoresway results page.

    Steps:
    1. Load results page
    2. Expand all matchday accordions
    3. Collect all <a href="/match/view/..."> hrefs into plain strings
    4. For each unique match URL, determine completion from parent row text
    5. Return structured list
    """
    _, _, By, WebDriverWait, EC, TimeoutException, _ = _get_webdriver()

    results_url = season_config["scoresway_results_url"]
    base_url = season_config["scoresway_base_url"]

    _log(progress_callback, f"Loading results page: {results_url}")

    try:
        driver.get(results_url)
        time.sleep(5)
    except Exception as e:
        _log(progress_callback, f"Failed to load results page: {e}")
        return []

    _dismiss_cookies(driver, By)
    time.sleep(1)

    # ─── Step 1: Expand all matchday accordions ──────────────────────
    _log(progress_callback, "Expanding matchday accordions...")
    _expand_all_matchdays(driver, By, progress_callback)
    time.sleep(2)

    # Scroll to make sure all content is in the DOM
    _scroll_to_bottom(driver, max_scrolls=20, pause=1.0)
    time.sleep(1)

    # ─── Step 2: Extract all hrefs into plain Python strings ─────────
    # This avoids stale element issues — we extract everything now
    _log(progress_callback, "Extracting match URLs from page...")

    raw_hrefs = _extract_all_match_hrefs(driver, By, progress_callback)
    _log(progress_callback, f"Extracted {len(raw_hrefs)} unique /match/view/ URLs")

    if not raw_hrefs:
        _log(progress_callback, "⚠️ No match URLs found. Debug: saving page source...")
        _save_debug_page_source(driver, progress_callback)
        return []

    # Log samples
    for href in list(raw_hrefs)[:5]:
        _log(progress_callback, f"  Sample URL: {href}")

    # ─── Step 3: Determine match completion from row context ─────────
    _log(progress_callback, "Identifying completed matches from page text...")

    matches = _identify_completed_matches(driver, By, raw_hrefs, base_url, progress_callback)

    _log(progress_callback, f"Discovered {len(matches)} completed matches")

    # ─── Step 4: Fallback — if row-based check found nothing, use all URLs
    if not matches and raw_hrefs:
        _log(progress_callback, "⚠️ Row-based filtering found 0 matches. Using ALL discovered URLs as fallback.")
        for href in raw_hrefs:
            mid = _extract_match_id(href)
            if mid:
                matches.append({
                    "match_id": mid,
                    "match_url": href.split("/player-stats")[0].split("/summary")[0],
                    "stats_url": _to_stats_url(href, base_url, mid),
                    "week": None,
                    "label": mid,
                    "completed": True,  # assume completed since it's on the results page
                })
        _log(progress_callback, f"Fallback: {len(matches)} matches included")

    matches.sort(key=lambda m: (m.get("week") or 99, m["match_id"]))
    return matches


def _expand_all_matchdays(driver, By, progress_callback=None):
    """
    Expand all collapsed matchday accordion sections on the results page.
    Scoresway uses clickable headers like "Matchday 27" that toggle visibility.
    """
    expanded_count = 0

    # Strategy 1: Look for clickable matchday headers
    # Common selectors for accordion triggers on Scoresway
    selectors_to_try = [
        # Exact patterns from the screenshot
        "[class*='matchday'] h3", "[class*='matchday'] h4",
        "[class*='round'] h3", "[class*='round'] h4",
        "[class*='accordion']", "[class*='collaps']",
        "h3[class*='matchday']", "h4[class*='matchday']",
        # Generic clickable headers that might toggle content
        ".results-container h3", ".results h3",
        "[data-toggle]", "[aria-expanded]",
    ]

    for selector in selectors_to_try:
        try:
            headers = driver.find_elements(By.CSS_SELECTOR, selector)
            if headers:
                _log(progress_callback, f"  Found {len(headers)} accordion elements via '{selector}'")
                for h in headers:
                    try:
                        if h.is_displayed():
                            driver.execute_script("arguments[0].click();", h)
                            time.sleep(0.3)
                            expanded_count += 1
                    except Exception:
                        continue
                if expanded_count > 0:
                    break
        except Exception:
            continue

    # Strategy 2: Click any elements with aria-expanded="false"
    try:
        collapsed = driver.find_elements(By.CSS_SELECTOR, "[aria-expanded='false']")
        for elem in collapsed:
            try:
                driver.execute_script("arguments[0].click();", elem)
                time.sleep(0.3)
                expanded_count += 1
            except Exception:
                continue
    except Exception:
        pass

    # Strategy 3: Execute JS to force all hidden sections visible
    try:
        driver.execute_script("""
            // Force all potentially hidden match sections visible
            document.querySelectorAll('[style*="display: none"], [style*="display:none"]').forEach(el => {
                el.style.display = '';
            });
            document.querySelectorAll('[class*="collapse"]:not(.show)').forEach(el => {
                el.classList.add('show');
                el.style.display = '';
            });
            document.querySelectorAll('[hidden]').forEach(el => {
                el.removeAttribute('hidden');
            });
        """)
    except Exception:
        pass

    _log(progress_callback, f"  Expanded {expanded_count} accordion sections")


def _extract_all_match_hrefs(driver, By, progress_callback=None):
    """
    Extract all unique /match/view/ hrefs from the page as plain Python strings.
    Uses JavaScript for reliability — avoids stale element issues.
    """
    # Method 1: JavaScript extraction (most reliable)
    try:
        hrefs = driver.execute_script("""
            const links = document.querySelectorAll('a[href*="/match/view/"]');
            const urls = new Set();
            links.forEach(a => {
                const href = a.href || a.getAttribute('href');
                if (href && href.includes('/match/view/')) {
                    urls.add(href);
                }
            });
            return Array.from(urls);
        """)
        if hrefs:
            _log(progress_callback, f"  JS extraction: {len(hrefs)} unique URLs")
            return set(hrefs)
    except Exception as e:
        _log(progress_callback, f"  JS extraction failed: {e}")

    # Method 2: Selenium elements (fallback)
    hrefs = set()
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/match/view/']")
        _log(progress_callback, f"  Selenium found {len(elements)} anchor elements")
        for elem in elements:
            try:
                href = elem.get_attribute("href")
                if href and "/match/view/" in href:
                    hrefs.add(href)
            except Exception:
                continue
        _log(progress_callback, f"  Selenium extraction: {len(hrefs)} unique URLs")
    except Exception as e:
        _log(progress_callback, f"  Selenium extraction failed: {e}")

    return hrefs


def _identify_completed_matches(driver, By, hrefs, base_url, progress_callback=None):
    """
    For each unique match URL, determine if the match is completed.

    Strategy: Use JavaScript to get the text content of each match link's
    parent row, looking for 'FT' indicator or score pattern like "2 v 3".
    """
    matches = []
    seen_ids = set()
    rejected_count = 0

    # Build a mapping: match_id → href
    id_to_href = {}
    for href in hrefs:
        mid = _extract_match_id(href)
        if mid and mid not in id_to_href:
            id_to_href[mid] = href

    _log(progress_callback, f"  Unique match IDs: {len(id_to_href)}")

    # Use JavaScript to get row context for each match link
    try:
        row_data = driver.execute_script("""
            const results = [];
            const links = document.querySelectorAll('a[href*="/match/view/"]');
            const seenIds = new Set();

            links.forEach(a => {
                const href = a.href || a.getAttribute('href');
                if (!href) return;

                const match = href.match(/\\/match\\/view\\/([a-zA-Z0-9]+)/);
                if (!match) return;

                const matchId = match[1];
                if (seenIds.has(matchId)) return;
                seenIds.add(matchId);

                // Walk up to find the row context (up to 5 levels)
                let rowText = '';
                let el = a;
                for (let i = 0; i < 6; i++) {
                    el = el.parentElement;
                    if (!el) break;
                    const t = el.textContent || '';
                    if (t.length > 10 && t.length < 500) {
                        rowText = t;
                    }
                }

                // Also get the direct link text
                const linkText = a.textContent || '';

                results.push({
                    matchId: matchId,
                    href: href,
                    linkText: linkText.trim().substring(0, 200),
                    rowText: rowText.trim().substring(0, 300),
                });
            });
            return results;
        """)
    except Exception as e:
        _log(progress_callback, f"  JS row extraction failed: {e}")
        row_data = []

    _log(progress_callback, f"  Got row context for {len(row_data)} match links")

    # Log first 5 samples for debugging
    for sample in row_data[:5]:
        _log(progress_callback,
             f"  Sample: id={sample['matchId'][:12]}... "
             f"linkText='{sample['linkText'][:60]}' "
             f"rowText='{sample['rowText'][:80]}...'")

    # Now classify each match
    for item in row_data:
        mid = item["matchId"]
        if mid in seen_ids:
            continue
        seen_ids.add(mid)

        link_text = item.get("linkText", "")
        row_text = item.get("rowText", "")
        combined = f"{link_text} {row_text}"

        # Check for completed match indicators
        is_completed = _is_match_completed(combined)

        if is_completed:
            # Try to extract week number from row context
            week = _extract_week_from_text(row_text)

            # Build clean label
            label = link_text if len(link_text) > 3 else row_text[:80]
            label = re.sub(r'\s+', ' ', label).strip()

            href = item.get("href", id_to_href.get(mid, ""))
            stats_url = _to_stats_url(href, base_url, mid)

            matches.append({
                "match_id": mid,
                "match_url": f"{base_url}/match/view/{mid}",
                "stats_url": stats_url,
                "week": week,
                "label": label[:100] if label else mid,
                "completed": True,
            })
        else:
            rejected_count += 1

    if rejected_count > 0:
        _log(progress_callback, f"  Rejected {rejected_count} non-completed matches")

    return matches


def _is_match_completed(text):
    """
    Determine if a match is completed based on row text.
    Looks for:
    - "FT" (full time) indicator
    - Score patterns: "2 v 3", "2 - 3", "2–3", "2:3"
    - Absence of future indicators like time patterns "20:00", "TBD"
    """
    if not text:
        return False

    text_upper = text.upper()

    # Strong positive indicator: "FT" (full time)
    if re.search(r'\bFT\b', text_upper):
        return True

    # Score with "v" separator (Scoresway format): "2 v 3"
    if re.search(r'\d+\s*v\s*\d+', text, re.IGNORECASE):
        return True

    # Score with dash/colon separator: "2-3", "2–3", "2:3"
    if re.search(r'\d+\s*[-–]\s*\d+', text):
        # But exclude time patterns like "20:00", "15:30"
        if not re.search(r'\b\d{1,2}:\d{2}\b', text):
            return True

    # Other completion indicators
    if any(ind in text_upper for ind in ["AET", "PEN", "AWARDED", "ABANDONED"]):
        return True

    return False


def _extract_match_id(href):
    """Extract the match ID hash from a /match/view/ URL."""
    m = re.search(r'/match/view/([a-zA-Z0-9]+)', href)
    return m.group(1) if m else None


def _to_stats_url(href, base_url, match_id):
    """Convert any match URL to the /player-stats URL."""
    return f"{base_url}/match/view/{match_id}/player-stats"


def _extract_week_from_text(text):
    """Extract matchweek number from text like 'Matchday 27'."""
    m = re.search(r'(?:Matchday|Round|Week|Journée|MD)\s*(\d+)', text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _save_debug_page_source(driver, progress_callback=None):
    """Save page source for debugging when discovery finds nothing."""
    try:
        source = driver.page_source
        debug_path = os.path.join("data", "raw", "debug_page_source.html")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(source)
        _log(progress_callback, f"  Debug page source saved to {debug_path} ({len(source)} chars)")
    except Exception as e:
        _log(progress_callback, f"  Failed to save debug source: {e}")


def _scroll_to_bottom(driver, max_scrolls=20, pause=1.0):
    """Scroll the page to load lazy/infinite-scroll content."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


# ═══════════════════════════════════════════════════════════════════════════════
# CAPTURE — extract PerformFeeds JSON from a match's player-stats page
# ═══════════════════════════════════════════════════════════════════════════════

def capture_match_events(driver, match, output_json_path, progress_callback=None):
    """
    Navigate to a match's /player-stats page and extract the PerformFeeds
    matchevent JSON data.

    Tries in order:
    1. Chrome CDP Network.getResponseBody for matchevent API responses
    2. Direct fetch of captured API URL using browser cookies
    3. Page source extraction for embedded JSON/JSONP
    """
    stats_url = match.get("stats_url") or match.get("match_url", "") + "/player-stats"
    match_id = match.get("match_id", "")

    _log(progress_callback, f"  Opening {stats_url}")

    # Clear previous performance logs
    try:
        driver.get_log("performance")
    except Exception:
        pass

    # Navigate
    try:
        driver.get(stats_url)
    except Exception as e:
        _log(progress_callback, f"  Page load issue (continuing): {e}")

    # Wait for API calls to fire
    time.sleep(7)

    # ─── Attempt 1: CDP network interception ─────────────────────────
    captured_requests = _collect_matchevent_requests(driver, progress_callback)

    for req_info in captured_requests:
        body = _get_response_body(driver, req_info, progress_callback)
        if body:
            parsed = _parse_api_payload(body, progress_callback)
            if parsed and _validate_match_data(parsed):
                _save_to_file(parsed, output_json_path)
                events = parsed.get("liveData", {}).get("event", [])
                _log(progress_callback, f"  ✅ Captured via CDP ({len(events)} events)")
                return True

    # ─── Attempt 2: Direct fetch with browser cookies ────────────────
    for req_info in captured_requests:
        url = req_info.get("url", "")
        if not url:
            continue
        _log(progress_callback, f"  Trying direct fetch: {url[:100]}...")
        body = _fetch_with_browser_cookies(driver, url, progress_callback)
        if body:
            parsed = _parse_api_payload(body, progress_callback)
            if parsed and _validate_match_data(parsed):
                _save_to_file(parsed, output_json_path)
                events = parsed.get("liveData", {}).get("event", [])
                _log(progress_callback, f"  ✅ Captured via direct fetch ({len(events)} events)")
                return True

    # ─── Attempt 3: Page source extraction ───────────────────────────
    _log(progress_callback, f"  Trying page source extraction...")
    try:
        page_source = driver.page_source
        extracted = _extract_match_json_from_source(page_source)
        if extracted:
            _save_to_file(extracted, output_json_path)
            events = extracted.get("liveData", {}).get("event", [])
            _log(progress_callback, f"  ✅ Extracted from page source ({len(events)} events)")
            return True
    except Exception as e:
        _log(progress_callback, f"  Page source extraction failed: {e}")

    # ─── All attempts failed — save debug artifacts ──────────────────
    _save_capture_debug(driver, match_id, captured_requests, output_json_path, progress_callback)
    _log(progress_callback, f"  ❌ Failed to capture data for {match_id}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# CDP NETWORK INTERCEPTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_matchevent_requests(driver, progress_callback=None):
    """
    Scan Chrome performance logs for /soccerdata/matchevent/ API requests.
    Returns list of dicts with url, requestId, status, contentType.
    """
    results = []
    all_urls_seen = []

    try:
        logs = driver.get_log("performance")
        _log(progress_callback, f"  Performance log entries: {len(logs)}")

        for entry in logs:
            try:
                log_data = json.loads(entry["message"])
                msg = log_data.get("message", {})
                method = msg.get("method", "")
                params = msg.get("params", {})

                if method == "Network.responseReceived":
                    resp = params.get("response", {})
                    url = resp.get("url", "")
                    status = resp.get("status", 0)
                    content_type = resp.get("headers", {}).get("content-type", resp.get("headers", {}).get("Content-Type", ""))
                    request_id = params.get("requestId", "")

                    # Track ALL performfeeds URLs for debug
                    if "performfeeds.com" in url:
                        all_urls_seen.append(url[:120])

                    # Filter: must be matchevent endpoint with 200 status
                    if "/soccerdata/matchevent/" in url and status == 200:
                        results.append({
                            "url": url,
                            "requestId": request_id,
                            "status": status,
                            "contentType": content_type,
                        })

                    # Also accept match endpoint (some pages use /match/ not /matchevent/)
                    elif "/soccerdata/match/" in url and "matchevent" not in url and status == 200:
                        results.append({
                            "url": url,
                            "requestId": request_id,
                            "status": status,
                            "contentType": content_type,
                            "fallback": True,
                        })

            except Exception:
                continue

    except Exception as e:
        _log(progress_callback, f"  Failed to read perf logs: {e}")

    _log(progress_callback, f"  PerformFeeds URLs seen: {len(all_urls_seen)}")
    for u in all_urls_seen[:5]:
        _log(progress_callback, f"    {u}")
    _log(progress_callback, f"  Matchevent responses (200): {len(results)}")

    # Sort: prefer matchevent over match, prefer non-fallback
    results.sort(key=lambda r: (r.get("fallback", False), r["url"]))

    return results


def _get_response_body(driver, req_info, progress_callback=None):
    """
    Get the actual response body from Chrome CDP for a captured request.
    Returns the raw text body, or None on failure.
    """
    request_id = req_info.get("requestId", "")
    url = req_info.get("url", "")

    if not request_id:
        return None

    try:
        result = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
        body = result.get("body", "")
        base64_encoded = result.get("base64Encoded", False)

        if base64_encoded:
            import base64
            body = base64.b64decode(body).decode("utf-8", errors="replace")

        _log(progress_callback,
             f"  CDP body for {url[:80]}...\n"
             f"    Length: {len(body)}, Preview: {body[:200]}")

        if len(body) < 50:
            _log(progress_callback, f"    Body too short, skipping")
            return None

        return body

    except Exception as e:
        _log(progress_callback, f"  CDP getResponseBody failed for {request_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# JSONP / RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_api_payload(text, progress_callback=None):
    """
    Parse a PerformFeeds API response that may be:
    - plain JSON: {"matchInfo": ...}
    - JSONP: callbackName({"matchInfo": ...});
    - JS assignment: var data = {"matchInfo": ...};
    - HTML error page
    Returns parsed dict or None.
    """
    if not text or len(text) < 10:
        return None

    text = text.strip()

    # Detect HTML error pages
    if text.startswith("<!") or text.startswith("<html") or text.startswith("<HTML"):
        _log(progress_callback, f"    Response is HTML, not JSON/JSONP")
        return None

    # Case 1: Plain JSON
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            _log(progress_callback, f"    Parsed as plain JSON")
            return parsed
        except json.JSONDecodeError:
            pass

    # Case 2: JSONP — callback({"matchInfo":...});
    # Pattern: anyCallbackName( ... );  or  anyCallbackName( ... )
    jsonp_match = re.match(r'^[a-zA-Z_$][\w$]*\s*\(\s*(.*)\s*\)\s*;?\s*$', text, re.DOTALL)
    if jsonp_match:
        inner = jsonp_match.group(1)
        try:
            parsed = json.loads(inner)
            _log(progress_callback, f"    Parsed as JSONP (stripped callback wrapper)")
            return parsed
        except json.JSONDecodeError as e:
            _log(progress_callback, f"    JSONP inner parse failed: {e}")

    # Case 3: JS assignment — var x = {...};
    assign_match = re.match(r'^(?:var|let|const)\s+\w+\s*=\s*(.*?)\s*;?\s*$', text, re.DOTALL)
    if assign_match:
        inner = assign_match.group(1)
        try:
            parsed = json.loads(inner)
            _log(progress_callback, f"    Parsed as JS assignment")
            return parsed
        except json.JSONDecodeError:
            pass

    # Case 4: Try to find a JSON object starting with { in the text
    brace_idx = text.find("{")
    if brace_idx >= 0 and brace_idx < 200:
        candidate = text[brace_idx:]
        # Strip trailing non-JSON
        candidate = re.sub(r'\)\s*;?\s*$', '', candidate)
        try:
            parsed = json.loads(candidate)
            _log(progress_callback, f"    Parsed by extracting JSON object at offset {brace_idx}")
            return parsed
        except json.JSONDecodeError:
            pass

    _log(progress_callback, f"    Failed to parse response (len={len(text)}, start={text[:80]})")
    return None


def _validate_match_data(data):
    """Check that parsed data contains matchInfo with events."""
    if not isinstance(data, dict):
        return False
    if "matchInfo" not in data:
        return False
    events = data.get("liveData", {}).get("event", [])
    return len(events) > 5


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECT FETCH WITH BROWSER COOKIES
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_with_browser_cookies(driver, url, progress_callback=None):
    """
    Fetch a URL using the browser's cookies and a realistic session.
    Returns response text or None.
    """
    try:
        import urllib.request
        import http.cookiejar

        # Build cookie header from Selenium
        cookies = driver.get_cookies()
        cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

        # Get the current page URL for Referer
        referer = driver.current_url

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": referer,
            "Cookie": cookie_str,
            "X-Requested-With": "XMLHttpRequest",
        })

        with urllib.request.urlopen(req, timeout=20) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read().decode("utf-8", errors="replace")

            _log(progress_callback,
                 f"    Direct fetch status={status}, type={content_type}, len={len(body)}")
            _log(progress_callback,
                 f"    Body preview: {body[:200]}")

            if len(body) < 50:
                return None

            return body

    except Exception as e:
        _log(progress_callback, f"    Direct fetch failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FILE SAVE & DEBUG
# ═══════════════════════════════════════════════════════════════════════════════

def _save_to_file(parsed_dict, output_path):
    """Save a validated parsed dict to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_dict, f, indent=2, ensure_ascii=False)


def _save_capture_debug(driver, match_id, captured_requests, output_json_path, progress_callback=None):
    """Save debug artifacts when capture fails."""
    debug_dir = os.path.join(os.path.dirname(output_json_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Save captured request info
    try:
        debug_info = {
            "match_id": match_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "current_url": driver.current_url,
            "captured_requests": captured_requests,
        }
        debug_path = os.path.join(debug_dir, f"{match_id}_debug.json")
        with open(debug_path, "w") as f:
            json.dump(debug_info, f, indent=2)
        _log(progress_callback, f"  Debug info saved to {debug_path}")
    except Exception:
        pass

    # Save page source snippet
    try:
        source = driver.page_source
        source_path = os.path.join(debug_dir, f"{match_id}_page.html")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source[:500_000])  # first 500KB
        _log(progress_callback, f"  Page source saved ({len(source)} chars)")
    except Exception:
        pass


def _extract_match_json_from_source(text):
    """
    Extract matchInfo+liveData JSON from page source text.
    Handles plain JSON, JSONP callbacks in script tags, etc.
    """
    # First try: find matchInfo block directly
    idx = text.find('"matchInfo"')
    if idx < 0:
        return None

    # Walk backwards to find opening brace
    start = idx
    while start > 0 and text[start] != '{':
        start -= 1
    if start < 0:
        return None

    # Walk forward to find matching close brace
    depth = 0
    for i in range(start, min(start + 10_000_000, len(text))):
        c = text[i]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        if depth == 0:
            candidate = text[start:i + 1]
            try:
                obj = json.loads(candidate)
                if "matchInfo" in obj and "liveData" in obj:
                    events = obj.get("liveData", {}).get("event", [])
                    if len(events) > 5:
                        return obj
            except json.JSONDecodeError:
                pass
            break

    return None


def _log(callback, msg):
    logger.info(msg)
    if callback:
        callback(msg)
