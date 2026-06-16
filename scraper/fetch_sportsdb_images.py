"""
scraper/fetch_sportsdb_images.py
==================================
Fetches player image URLs from TheSportsDB (free, browser-accessible CDN).

TheSportsDB images work directly in browsers (HTTP 200, no hotlink blocking)
unlike Wikipedia images which return 403 to browsers.

Run: python scraper/fetch_sportsdb_images.py
"""

import logging
import re
import time
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("footscout.sportsdb")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "players_merged.csv"

SPORTSDB_API = "https://www.thesportsdb.com/api/v1/json/3/searchplayers.php"
HEADERS = {
    "User-Agent": "FootScout/1.0 (BHT Berlin; educational use)",
    "Accept": "application/json",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def normalize(name: str) -> str:
    """Normalize name: remove accents, lowercase, strip punctuation."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", ascii_name.lower()).strip()


def name_score(search: str, candidate: str) -> float:
    """Return 0.0–1.0 similarity between two player name strings."""
    s = normalize(search)
    c = normalize(candidate)
    if s == c:
        return 1.0
    s_parts = set(s.split())
    c_parts = set(c.split())
    if not s_parts or not c_parts:
        return 0.0
    # Jaccard on words
    overlap = len(s_parts & c_parts)
    union = len(s_parts | c_parts)
    jaccard = overlap / union if union else 0.0
    # Bonus for substring containment
    if s in c or c in s:
        jaccard = max(jaccard, 0.75)
    # Partial last-name match
    s_last = s.split()[-1] if s.split() else ""
    c_last = c.split()[-1] if c.split() else ""
    if s_last and c_last and s_last == c_last:
        jaccard = max(jaccard, 0.7)
    return jaccard


def build_search_attempts(player_name: str) -> list[str]:
    """Generate multiple search query variants for a player name."""
    name = player_name.strip()
    parts = name.split()
    attempts = [name]  # full name
    if len(parts) >= 2:
        attempts.append(parts[-1])       # last name
        attempts.append(parts[0])        # first name
        attempts.append(f"{parts[0]} {parts[-1]}")  # first + last
        # Try without accents
        norm = normalize(name)
        if norm != name.lower():
            attempts.append(norm)
    # Remove suffixes
    cleaned = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV)$', '', name, flags=re.I).strip()
    if cleaned != name:
        attempts.append(cleaned)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for a in attempts:
        if a.lower() not in seen and a.strip():
            seen.add(a.lower())
            unique.append(a)
    return unique


def fetch_sportsdb_image(player_name: str) -> tuple[str, str | None]:
    """
    Query TheSportsDB API for a player image URL.
    Returns: (player_name, image_url or None)
    """
    attempts = build_search_attempts(player_name)
    
    best_url = None
    best_score = 0.0

    for attempt in attempts:
        try:
            r = SESSION.get(SPORTSDB_API, params={"p": attempt}, timeout=12)
            if r.status_code == 429:
                time.sleep(2)
                r = SESSION.get(SPORTSDB_API, params={"p": attempt}, timeout=12)
            if r.status_code != 200:
                continue
            
            data = r.json()
            candidates = data.get("player") or []
            
            for p in candidates[:6]:
                p_name = p.get("strPlayer") or ""
                score = name_score(player_name, p_name)
                img = (
                    p.get("strThumb") or
                    p.get("strCutout") or
                    p.get("strRender") or ""
                )
                if score > best_score and img and img.startswith("http"):
                    best_score = score
                    best_url = img
            
            if best_score >= 0.65:
                break  # good enough match found
                
        except Exception as e:
            logger.debug("SportsDB error for '%s': %s", attempt, e)
        
        time.sleep(0.08)  # gentle rate limiting
    
    if best_score >= 0.45:  # accept looser matches for common surnames
        return player_name, best_url
    return player_name, None


def main():
    import sys
    force = "--force" in sys.argv
    
    if not CSV_PATH.exists():
        logger.error("CSV not found: %s", CSV_PATH)
        return
    
    df = pd.read_csv(CSV_PATH)
    logger.info("Loaded %d players", len(df))
    
    if "sportsdb_image_url" not in df.columns:
        df["sportsdb_image_url"] = None

    # Filter to players who need fetching
    if force:
        players_to_fetch = df["player"].dropna().tolist()
        logger.info("Force fetching all %d players...", len(players_to_fetch))
    else:
        # Fetch players where sportsdb_image_url is null or not a valid URL
        mask_to_fetch = df["sportsdb_image_url"].isna() | (~df["sportsdb_image_url"].astype(str).str.startswith("http"))
        players_to_fetch = df.loc[mask_to_fetch, "player"].dropna().tolist()
        logger.info(
            "Found %d players already fetched. Fetching remaining %d players...",
            len(df) - len(players_to_fetch),
            len(players_to_fetch)
        )
        
    if not players_to_fetch:
        logger.info("No players to fetch. Done!")
        return
    
    results: dict[str, str | None] = {}
    completed = 0
    found = 0
    total = len(players_to_fetch)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_sportsdb_image, name): name
            for name in players_to_fetch
        }
        
        for future in as_completed(futures):
            name, url = future.result()
            results[name] = url
            if url:
                found += 1
            completed += 1
            
            # Update dataframe progressively
            df.loc[df["player"] == name, "sportsdb_image_url"] = url
            
            # Save progressively every 20 players to prevent data loss on cancel/interrupt
            if completed % 20 == 0 or completed == total:
                df.to_csv(CSV_PATH, index=False)
                logger.info(
                    "Progress: %d/%d  |  Found: %d (%.0f%%)  |  Progress saved to CSV",
                    completed, total, found, 100 * found / total
                )
    
    total_img = df["sportsdb_image_url"].notna().sum()
    logger.info(
        "\n✅ Done! %d/%d players have TheSportsDB images (%.1f%%)",
        total_img, len(df), 100 * total_img / len(df)
    )
    
    # Show a few examples
    sample = df[df["sportsdb_image_url"].notna()][["player", "sportsdb_image_url"]].head(10)
    logger.info("\nSample results:\n%s", sample.to_string(index=False))


if __name__ == "__main__":
    main()
