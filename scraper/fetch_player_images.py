"""
scraper/fetch_player_images.py
===============================
Pre-fetches actual player photo URLs from the Wikipedia PageImage API
for all players in the merged dataset.

Saves them to a new 'image_url' column in players_merged.csv.
Uses a ThreadPoolExecutor for fast parallel fetching.
"""

import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("footscout.fetch_images")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "players_merged.csv"

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "FootScout/1.0 (BHT Berlin DS Workflow Master Project; contact@footscout.de)"
}

def fetch_player_image(player_name: str) -> tuple[str, str | None]:
    """Query Wikipedia to find the primary image for a player name."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": player_name,
        "gsrlimit": 3,
        "prop": "pageimages",
        "piprop": "original",
        "pilicense": "any"
    }
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            # Sort pages by relevance index
            sorted_pages = sorted(pages.values(), key=lambda p: p.get("index", 999))
            for page_data in sorted_pages:
                source = page_data.get("original", {}).get("source")
                if source:
                    # Filter out logos, flags, or icons that are not player photos
                    source_lower = source.lower()
                    if any(x in source_lower for x in [".svg", "flag", "logo", "shield", "crest"]):
                        continue
                    return player_name, source
    except Exception as e:
        logger.debug("Failed to fetch image for %s: %s", player_name, e)
    return player_name, None

def main():
    if not CSV_PATH.exists():
        logger.error("merged CSV not found at %s", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)
    logger.info("Loaded %d players from %s", len(df), CSV_PATH.name)

    player_names = df["player"].dropna().unique().tolist()
    image_urls = {}

    logger.info("Fetching actual player images from Wikipedia API...")
    
    # Run with 20 parallel threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_player_image, name): name for name in player_names}
        
        completed = 0
        for future in as_completed(futures):
            name, url = future.result()
            if url:
                image_urls[name] = url
            completed += 1
            if completed % 50 == 0 or completed == len(player_names):
                logger.info("Progress: %d/%d players processed...", completed, len(player_names))

    # Map image URLs to players
    df["image_url"] = df["player"].map(image_urls)
    
    # Save back to CSV
    df.to_csv(CSV_PATH, index=False)
    logger.info("Saved %d image URLs (%.1f%% coverage) back to %s",
                df["image_url"].notna().sum(),
                100 * df["image_url"].notna().sum() / len(df),
                CSV_PATH.name)

if __name__ == "__main__":
    main()
