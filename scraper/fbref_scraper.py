"""
scraper/fbref_scraper.py
========================
FootScout — FBref Scraper
-------------------------
Scrapes per-90 player statistics from FBref for:
  - Premier League
  - Bundesliga
  - La Liga
  - Serie A
  - Ligue 1
  - FIFA World Cup 2026 (qualifying/group stage)

Design Principles
-----------------
1. HTML buffering: raw HTML is saved to data/raw/html/ before parsing.
   If FBref layouts change, you can re-parse locally without re-scraping.
2. Rate throttle: ~1 request per second (configurable via SCRAPER_DELAY_S).
3. Kaggle fallback: set USE_KAGGLE_FALLBACK=true in .env to bypass live scraping.
4. Entry threshold: players with < 450 minutes are excluded from per-90 stats.
"""

from __future__ import annotations

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
HTML_CACHE    = DATA_RAW / "html" / "fbref"
CSV_OUT       = DATA_RAW / "fbref_raw.csv"
KAGGLE_CSV    = DATA_RAW / "kaggle_fallback.csv"

# Create directories
DATA_RAW.mkdir(parents=True, exist_ok=True)
HTML_CACHE.mkdir(parents=True, exist_ok=True)

SCRAPER_DELAY_S    = float(os.getenv("SCRAPER_DELAY_S", "1.2"))
USE_KAGGLE_FALLBACK = os.getenv("USE_KAGGLE_FALLBACK", "false").lower() == "true"
MIN_MINUTES        = int(os.getenv("MIN_MINUTES", "450"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("footscout.fbref")

# ─── FBref League Table URLs ────────────────────────────────────────────────────
# Standard stats tables — includes per-90 columns when selecting "Per 90 Minutes"
LEAGUE_URLS: dict[str, str] = {
    "Premier League": (
        "https://fbref.com/en/comps/9/stats/Premier-League-Stats"
    ),
    "Bundesliga": (
        "https://fbref.com/en/comps/20/stats/Bundesliga-Stats"
    ),
    "La Liga": (
        "https://fbref.com/en/comps/12/stats/La-Liga-Stats"
    ),
    "Serie A": (
        "https://fbref.com/en/comps/11/stats/Serie-A-Stats"
    ),
    "Ligue 1": (
        "https://fbref.com/en/comps/13/stats/Ligue-1-Stats"
    ),
    "World Cup 2026": (
        "https://fbref.com/en/comps/1/stats/World-Cup-Stats"
    ),
}

# Additional stat table suffixes for advanced metrics
STAT_TABLE_SUFFIXES: dict[str, str] = {
    "shooting": "shooting",
    "passing":  "passing",
    "gca":      "gca",         # Goal-Creating Actions
    "defense":  "defense",
    "possession": "possession",
    "misc":     "misc",
}

# ─── HTTP Session Setup ─────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FootScoutBot/1.0; "
        "+https://github.com/footscout; research-project)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _url_to_cache_path(url: str, suffix: str = "") -> Path:
    """Map a URL to a stable, filesystem-safe cache path."""
    key = hashlib.md5(url.encode()).hexdigest()[:12]
    name = f"{key}{suffix}.html"
    return HTML_CACHE / name


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10))
def _fetch_html(url: str, force_refresh: bool = False) -> str:
    """
    Fetch HTML from URL, using local cache if available.

    DEFENSIVE DESIGN: HTML is always persisted to disk first.
    Parsing only happens on the cached copy.
    """
    cache_path = _url_to_cache_path(url)

    if cache_path.exists() and not force_refresh:
        logger.info("Cache HIT  → %s", cache_path.name)
        return cache_path.read_text(encoding="utf-8")

    logger.info("Fetching   → %s", url)
    time.sleep(SCRAPER_DELAY_S)  # Rate throttle

    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    html = resp.text

    # ── Save raw HTML to disk BEFORE parsing ──────────────────────────────────
    cache_path.write_text(html, encoding="utf-8")
    logger.info("Cached     → %s (%d bytes)", cache_path.name, len(html))

    return html


def _parse_standard_table(html: str, league: str) -> Optional[pd.DataFrame]:
    """
    Parse FBref 'Standard Stats' table from HTML.
    Returns a DataFrame with one row per player, or None on failure.
    """
    soup = BeautifulSoup(html, "lxml")

    # FBref table IDs follow a predictable pattern
    table = soup.find("table", {"id": lambda x: x and "stats_standard" in x})
    if table is None:
        logger.warning("Standard stats table not found for league: %s", league)
        return None

    try:
        df = pd.read_html(str(table), header=[0, 1])[0]
    except Exception as exc:
        logger.error("Failed to parse HTML table: %s", exc)
        return None

    # FBref uses multi-level headers — flatten them
    df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns]
    df = df.copy()

    # Drop aggregate rows (FBref inserts repeated header rows in HTML)
    if "Unnamed: 0_level_0_Rk" in df.columns:
        df = df[df["Unnamed: 0_level_0_Rk"] != "Rk"].copy()
    elif "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].copy()

    df["league"] = league
    logger.info("Parsed %d rows for %s", len(df), league)
    return df


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FBref multi-level column names to snake_case."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        clean = (
            col.lower()
               .replace("unnamed: ", "")
               .replace("_level_0_", "_")
               .replace("per 90 minutes_", "per90_")
               .replace("performance_", "")
               .replace("expected_", "xg_")
               .replace("progression_", "prog_")
               .replace(".", "")
               .replace(" ", "_")
               .strip("_")
        )
        rename_map[col] = clean
    return df.rename(columns=rename_map)


def _apply_per90_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-90 normalization: stat_per90 = stat / (minutes / 90)

    CORE NORMALIZATION RULE: Eliminates playtime bias so that a player
    with 900 minutes is compared fairly with one who played 1800 minutes.

    Also applies the ENTRY THRESHOLD: drop players with < MIN_MINUTES.
    """
    minutes_col = None
    for candidate in ["playing_time_min", "min", "minutes", "playing_time_minutes"]:
        if candidate in df.columns:
            minutes_col = candidate
            break

    if minutes_col is None:
        logger.warning("Minutes column not found — skipping per-90 normalization")
        return df

    df[minutes_col] = pd.to_numeric(df[minutes_col], errors="coerce")

    # ── Apply entry threshold ──────────────────────────────────────────────────
    before = len(df)
    df = df[df[minutes_col] >= MIN_MINUTES].copy()
    logger.info(
        "Entry threshold (≥%d min): %d → %d players", MIN_MINUTES, before, len(df)
    )

    # ── Duplicate resolution: mid-season transfer → keep max minutes row ──────
    id_cols = [c for c in ["player", "name"] if c in df.columns]
    if id_cols:
        df = (
            df.sort_values(minutes_col, ascending=False)
              .drop_duplicates(subset=id_cols, keep="first")
        )
        logger.info("After dedup: %d players remaining", len(df))

    # ── Compute per-90 for counting stats ─────────────────────────────────────
    COUNT_STATS = [
        "gls", "ast", "g+a", "g-pk", "pk", "pkatt",
        "sh", "sot", "touches", "att_3rd", "prog_carries", "prog_passes",
        "tackles_tkl", "tkl_w", "blocks", "int", "clr",
        "xg", "npxg", "xag", "xa",
    ]
    per90_factor = df[minutes_col] / 90.0

    for stat in COUNT_STATS:
        if stat in df.columns:
            df[stat] = pd.to_numeric(df[stat], errors="coerce")
            df[f"{stat}_per90"] = (df[stat] / per90_factor).round(4)

    return df


def _deduplicate_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """
    DATA QUALITY: Resolve mid-season multi-row player transfers.
    Keep only the row with maximum minutes played per player name.
    """
    id_col = next((c for c in ["player", "name"] if c in df.columns), None)
    if id_col is None:
        return df

    minutes_col = next(
        (c for c in ["playing_time_min", "min", "minutes"] if c in df.columns), None
    )
    if minutes_col is None:
        return df

    df[minutes_col] = pd.to_numeric(df[minutes_col], errors="coerce")
    original_len = len(df)
    df = (
        df.sort_values(minutes_col, ascending=False)
          .drop_duplicates(subset=[id_col], keep="first")
          .reset_index(drop=True)
    )
    logger.info(
        "Transfer dedup: %d → %d rows (removed %d duplicates)",
        original_len, len(df), original_len - len(df),
    )
    return df


# ─── Kaggle Fallback ────────────────────────────────────────────────────────────

def load_kaggle_fallback() -> pd.DataFrame:
    """
    KAGGLE FALLBACK SWITCH:
    When live scrapers are blocked, load a pre-downloaded Kaggle FIFA dataset.
    Set USE_KAGGLE_FALLBACK=true in .env and place the CSV at data/raw/kaggle_fallback.csv.

    Expected Kaggle dataset: "FIFA 22 Complete Player Dataset" or similar.
    Download from: https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset
    """
    if not KAGGLE_CSV.exists():
        raise FileNotFoundError(
            f"Kaggle fallback CSV not found at {KAGGLE_CSV}.\n"
            "Download from Kaggle and place at data/raw/kaggle_fallback.csv\n"
            "Or set USE_KAGGLE_FALLBACK=false to use live scrapers."
        )

    logger.info("KAGGLE FALLBACK active — loading %s", KAGGLE_CSV)
    df = pd.read_csv(KAGGLE_CSV, low_memory=False)

    # ── Normalize Kaggle column names to match FBref schema ───────────────────
    kaggle_rename: dict[str, str] = {
        "short_name": "player",
        "club_name": "squad",
        "league_name": "league",
        "nationality_name": "nation",
        "player_positions": "pos",
        "age": "age",
        "overall": "overall_rating",
        "value_eur": "market_value_eur",
        "wage_eur": "wage_eur",
        "goals_scored": "gls",
        "assists": "ast",
        "minutes_played": "playing_time_min",
        "contract_valid_until": "contract_expires",
    }
    df = df.rename(columns={k: v for k, v in kaggle_rename.items() if k in df.columns})

    # Synthesize per-90 stats where possible
    if "playing_time_min" in df.columns and "gls" in df.columns:
        df["playing_time_min"] = pd.to_numeric(df["playing_time_min"], errors="coerce")
        df = df[df["playing_time_min"] >= MIN_MINUTES].copy()
        per90 = df["playing_time_min"] / 90
        for col in ["gls", "ast"]:
            if col in df.columns:
                df[f"{col}_per90"] = (
                    pd.to_numeric(df[col], errors="coerce") / per90
                ).round(4)

    logger.info("Kaggle fallback loaded: %d players", len(df))
    return df


# ─── Main Scraping Orchestrator ─────────────────────────────────────────────────

def scrape_league(league_name: str, url: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Scrape and process one league's standard stats table."""
    try:
        html = _fetch_html(url, force_refresh=force_refresh)
        df   = _parse_standard_table(html, league_name)
        if df is None:
            return None
        df = _clean_column_names(df)
        df = _apply_per90_normalization(df)
        return df
    except requests.HTTPError as exc:
        logger.error("HTTP error scraping %s: %s", league_name, exc)
        return None
    except Exception as exc:
        logger.error("Unexpected error scraping %s: %s", league_name, exc)
        return None


def run(force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point.
    Scrapes all leagues and saves the combined DataFrame to data/raw/fbref_raw.csv.
    Falls back to Kaggle dataset if USE_KAGGLE_FALLBACK is set.
    """
    if USE_KAGGLE_FALLBACK:
        df = load_kaggle_fallback()
        df.to_csv(CSV_OUT, index=False)
        logger.info("Kaggle data saved → %s", CSV_OUT)
        return df

    all_frames: list[pd.DataFrame] = []

    for league_name, url in LEAGUE_URLS.items():
        logger.info("━━ Scraping: %s ━━", league_name)
        df = scrape_league(league_name, url, force_refresh=force_refresh)
        if df is not None and not df.empty:
            all_frames.append(df)
        else:
            logger.warning("No data retrieved for %s", league_name)

    if not all_frames:
        raise RuntimeError(
            "All scrapers failed. Set USE_KAGGLE_FALLBACK=true to use the fallback dataset."
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined = _deduplicate_transfers(combined)

    combined.to_csv(CSV_OUT, index=False)
    logger.info(
        "✅ FBref scraping complete: %d players → %s", len(combined), CSV_OUT
    )
    return combined


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FootScout FBref Scraper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch all URLs even if cached HTML exists",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use Kaggle fallback dataset instead of scraping",
    )
    args = parser.parse_args()

    if args.kaggle:
        os.environ["USE_KAGGLE_FALLBACK"] = "true"

    result = run(force_refresh=args.force_refresh)
    print(f"\n✅ Done — {len(result):,} players saved to {CSV_OUT}")
    print(result[["player", "squad", "league", "pos", "age"]].head(10).to_string(index=False))
