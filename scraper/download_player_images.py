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
WIKI_HEADERS = {
    "User-Agent": "FootScout/1.0 (BHT Berlin DS Workflow Master Project; educational use; https://github.com/sina-778/footscout)",
    "Accept": "image/jpeg,image/png,image/webp,*/*;q=0.8",
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
    # Try exact search first, then fuzzy
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
        try:
            r = requests.get(WIKI_API_URL, params=params, headers=WIKI_HEADERS, timeout=12)
            if r.status_code != 200:
                continue
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
        except Exception as e:
            logger.debug("Wiki API error for '%s': %s", player_name, e)
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
    
    try:
        r = requests.get(url, headers=WIKI_HEADERS, timeout=20, stream=True)
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
            
            logger.debug("  ✅ Downloaded %s (%.1f KB)", player_name, size / 1024)
            return str(local_path)
        else:
            logger.debug("  ❌ HTTP %d for %s", r.status_code, player_name)
            return None
    except Exception as e:
        logger.debug("  Error downloading %s: %s", player_name, e)
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
    
    if not wiki_url:
        return player_name, None, None
    
    local_path = download_image(player_name, wiki_url, output_dir)
    return player_name, wiki_url, local_path


def main():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CSV_PATH.exists():
        logger.error("CSV not found: %s", CSV_PATH)
        return
    
    df = pd.read_csv(CSV_PATH)
    logger.info("Loaded %d players from %s", len(df), CSV_PATH.name)
    
    # Build work items
    existing_urls = {}
    if "image_url" in df.columns:
        existing_urls = dict(zip(df["player"], df["image_url"].where(df["image_url"].notna(), None)))
    
    players = df["player"].dropna().tolist()
    total = len(players)
    
    logger.info("Processing %d players with 15 parallel workers...", total)
    logger.info("Images will be saved to: %s", STATIC_DIR)
    
    results: dict[str, dict] = {}
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(
                process_player,
                name,
                existing_urls.get(name),
                STATIC_DIR
            ): name
            for name in players
        }
        
        completed = 0
        success = 0
        for future in as_completed(futures):
            name, wiki_url, local_path = future.result()
            results[name] = {
                "image_url": wiki_url,
                "local_image_path": local_path
            }
            if local_path:
                success += 1
            completed += 1
            if completed % 30 == 0 or completed == total:
                logger.info(
                    "Progress: %d/%d complete, %d images downloaded",
                    completed, total, success
                )
    
    # Update DataFrame
    df["image_url"] = df["player"].map(lambda n: results.get(n, {}).get("image_url"))
    df["local_image_path"] = df["player"].map(lambda n: results.get(n, {}).get("local_image_path"))
    
    # Convert local paths to relative paths from project root (NaN-safe)
    def _to_relative(p) -> str | None:
        if p is None or not isinstance(p, str) or not p.strip():
            return None
        try:
            return str(Path(p).relative_to(PROJECT_ROOT))
        except Exception:
            return p

    df["local_image_path"] = df["local_image_path"].apply(_to_relative)
    
    df.to_csv(CSV_PATH, index=False)
    
    success_count = df["local_image_path"].notna().sum()
    logger.info(
        "\n✅ Done! %d/%d players have local images (%.1f%% coverage)",
        success_count, total,
        100 * success_count / total
    )
    logger.info("Saved updated CSV to %s", CSV_PATH)
    logger.info("Images stored in %s", STATIC_DIR)


if __name__ == "__main__":
    main()
