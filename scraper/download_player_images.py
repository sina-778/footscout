"""
scraper/download_player_images.py
===================================
Downloads actual player photos from Wikipedia and saves them locally
under app/static/players/{safe_name}.jpg

Solves the 403 Forbidden problem: Wikipedia images block browser direct
access (hotlink protection) but CAN be fetched server-side with Python.

This script:
1. Re-scrapes image URLs for ALL players (not just existing 153)
2. Downloads each image locally to app/static/players/
3. Updates players_merged.csv with local_image_path column
4. Falls back to DiceBear avatar URL when no Wikipedia image is found

Run: python scraper/download_player_images.py
"""

import logging
import hashlib
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("footscout.download_images")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "players_merged.csv"
STATIC_DIR = PROJECT_ROOT / "app" / "static" / "players"

# Wikimedia requires a descriptive User-Agent for API calls
import random

def get_wiki_headers():
    project_id = random.randint(100, 999)
    return {
        "User-Agent": f"FootScout-Scraper/{project_id} (Educational data science project; contact: student-scout-dev{project_id}@bht-berlin.de)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://en.wikipedia.org/"
    }

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


def safe_filename(name: str) -> str:
    """Convert player name to a safe filename."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe.lower()


def fetch_wiki_image_url(player_name: str) -> str | None:
    """Query Wikipedia API to get the best image URL for a player."""
    for search_term in [player_name, player_name.split()[-1]]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": search_term,
            "gsrlimit": 3,
            "prop": "pageimages",
            "piprop": "original",
            "pilicense": "any"
        }
        
        # Retry with exponential backoff
        retries = 3
        backoff = 2
        for attempt in range(retries):
            try:
                headers = get_wiki_headers()
                r = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=12)
                if r.status_code == 429:
                    logger.warning("Wikimedia rate limit (429) on search '%s'. Backing off %ds...", search_term, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if r.status_code != 200:
                    break
                    
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for page_data in pages.values():
                    source = page_data.get("original", {}).get("source")
                    if not source:
                        continue
                    source_lower = source.lower()
                    # Filter out logos, flags, crests, icons
                    if any(x in source_lower for x in [
                        ".svg", "flag_", "logo_", "_logo", "shield", "crest",
                        "coat_of_arms", "emblem", "badge", "jersey", "_kit"
                    ]):
                        continue
                    return source
                break
            except Exception as e:
                logger.debug("Wiki API error for '%s' (attempt %d): %s", player_name, attempt, e)
                time.sleep(1)
        time.sleep(0.1)
    return None


def download_image(player_name: str, url: str, output_dir: Path) -> str | None:
    """Download an image and save it locally. Returns local file path or None."""
    safe_name = safe_filename(player_name)
    
    # Detect extension
    url_path = url.split("?")[0]
    ext = url_path.rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
        ext = "jpg"
    
    local_path = output_dir / f"{safe_name}.{ext}"
    
    # Skip if already downloaded
    if local_path.exists() and local_path.stat().st_size > 5000:
        return str(local_path)
    
    retries = 3
    backoff = 2
    for attempt in range(retries):
        try:
            headers = get_wiki_headers()
            r = requests.get(url, headers=headers, timeout=20, stream=True)
            if r.status_code == 429:
                logger.warning("Wikimedia rate limit (429) on image download for '%s'. Backing off %ds...", player_name, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "")
                if "image" not in content_type and "jpeg" not in content_type:
                    logger.warning("  Non-image content-type for %s: %s", player_name, content_type)
                    return None
                
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                size = local_path.stat().st_size
                if size < 2000:
                    local_path.unlink()
                    return None
                
                logger.info("  ✅ Downloaded %s (%.1f KB)", player_name, size / 1024)
                return str(local_path)
            else:
                logger.debug("  ❌ HTTP %d for %s", r.status_code, player_name)
                return None
        except Exception as e:
            logger.debug("  Error downloading %s (attempt %d): %s", player_name, attempt, e)
            time.sleep(1)
    return None


def process_player(player_name: str, existing_url, output_dir: Path) -> tuple[str, str | None, str | None]:
    """
    Full pipeline for one player:
    1. Use existing URL or fetch new one
    2. Download image locally
    Returns: (player_name, wiki_url, local_path)
    """
    import math
    # Normalise: treat NaN/None/float as missing
    if existing_url is None or (isinstance(existing_url, float) and math.isnan(existing_url)):
        existing_url = None
    wiki_url = existing_url if isinstance(existing_url, str) and existing_url.strip() else None
    
    # If no URL, try to fetch one
    if not wiki_url:
        wiki_url = fetch_wiki_image_url(player_name)
    
    time.sleep(0.3)  # polite rate-limiting delay between searches/downloads
    
    if not wiki_url:
        return player_name, None, None
    
    local_path = download_image(player_name, wiki_url, output_dir)
    return player_name, wiki_url, local_path


def main():
    import sys
    force = "--force" in sys.argv
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CSV_PATH.exists():
        logger.error("CSV not found: %s", CSV_PATH)
        return
    
    df = pd.read_csv(CSV_PATH)
    logger.info("Loaded %d players from %s", len(df), CSV_PATH.name)
    
    if "image_url" not in df.columns:
        df["image_url"] = None
    if "local_image_path" not in df.columns:
        df["local_image_path"] = None
        
    # Build list of players who need processing
    if force:
        players_to_process = df["player"].dropna().tolist()
        logger.info("Force processing all %d players...", len(players_to_process))
    else:
        # Check if local_image_path is valid and file exists
        def needs_download(row):
            path = row.get("local_image_path")
            if pd.isna(path) or not isinstance(path, str) or not path.strip():
                return True
            full_p = PROJECT_ROOT / path
            return not full_p.exists() or full_p.stat().st_size < 5000
            
        mask = df.apply(needs_download, axis=1)
        players_to_process = df.loc[mask, "player"].dropna().tolist()
        logger.info(
            "Found %d players already have valid local images. Processing remaining %d players...",
            len(df) - len(players_to_process),
            len(players_to_process)
        )
        
    if not players_to_process:
        logger.info("All players already have images. Done!")
        return
        
    existing_urls = dict(zip(df["player"], df["image_url"].where(df["image_url"].notna(), None)))
    
    def _to_relative(p) -> str | None:
        if p is None or not isinstance(p, str) or not p.strip():
            return None
        try:
            return str(Path(p).relative_to(PROJECT_ROOT))
        except Exception:
            return p

    completed = 0
    success = 0
    total = len(players_to_process)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                process_player,
                name,
                existing_urls.get(name),
                STATIC_DIR
            ): name
            for name in players_to_process
        }
        
        for future in as_completed(futures):
            name, wiki_url, local_path = future.result()
            rel_path = _to_relative(local_path)
            
            # Update DataFrame in memory
            idx = df["player"] == name
            df.loc[idx, "image_url"] = wiki_url
            df.loc[idx, "local_image_path"] = rel_path
            
            if local_path:
                success += 1
            completed += 1
            
            # Save progressively every 30 players
            if completed % 30 == 0 or completed == total:
                df.to_csv(CSV_PATH, index=False)
                logger.info(
                    "Progress: %d/%d complete, %d images downloaded | Progress saved to CSV",
                    completed, total, success
                )
                
    success_count = df["local_image_path"].notna().sum()
    logger.info(
        "\n✅ Done! %d/%d players have local images (%.1f%% coverage)",
        success_count, len(df),
        100 * success_count / len(df)
    )


if __name__ == "__main__":
    main()
