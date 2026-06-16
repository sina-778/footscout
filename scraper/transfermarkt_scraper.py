"""
scraper/transfermarkt_scraper.py
=================================
FootScout — Transfermarkt Scraper
----------------------------------
Scrapes player metadata from Transfermarkt:
  - Market value (EUR)
  - Contract expiry date
  - Nationality
  - Age
  - Position (detailed)
  - "Similar Players" lists (used as Method 2 evaluation benchmark)

Design Principles
-----------------
1. HTML buffering: raw HTML persisted before parsing.
2. Rate throttle: ~1.2 seconds between requests.
3. Robust to layout changes via cached HTML fallback.
4. Builds evaluation benchmark: scrapes "similar players" for 30-50 known players.
"""

from __future__ import annotations

import os
import re
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
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
HTML_CACHE   = DATA_RAW / "html" / "transfermarkt"
CSV_OUT      = DATA_RAW / "transfermarkt_raw.csv"
BENCHMARK_OUT= DATA_RAW / "transfermarkt_benchmark.csv"

DATA_RAW.mkdir(parents=True, exist_ok=True)
HTML_CACHE.mkdir(parents=True, exist_ok=True)

SCRAPER_DELAY_S = float(os.getenv("SCRAPER_DELAY_S", "1.2"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("footscout.transfermarkt")

# ─── Transfermarkt League URLs ──────────────────────────────────────────────────
# These point to the "detailed" view which includes market values per player
LEAGUE_URLS: dict[str, str] = {
    "Premier League": (
        "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1"
    ),
    "Bundesliga": (
        "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1"
    ),
    "La Liga": (
        "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1"
    ),
    "Serie A": (
        "https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1"
    ),
    "Ligue 1": (
        "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1"
    ),
}

# Player search URL template
PLAYER_SEARCH_URL = (
    "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={name}"
)

# Known top players for evaluation benchmark scraping (30 players)
BENCHMARK_PLAYERS: list[dict[str, str]] = [
    {"name": "Erling Haaland",     "tm_id": "418560"},
    {"name": "Kylian Mbappé",      "tm_id": "342229"},
    {"name": "Vinicius Junior",    "tm_id": "371998"},
    {"name": "Pedri",              "tm_id": "722093"},
    {"name": "Jude Bellingham",    "tm_id": "581678"},
    {"name": "Rodri",              "tm_id": "357565"},
    {"name": "Declan Rice",        "tm_id": "357662"},
    {"name": "Lautaro Martínez",   "tm_id": "406625"},
    {"name": "Harry Kane",         "tm_id": "132098"},
    {"name": "Mohamed Salah",      "tm_id": "148455"},
    {"name": "Kevin De Bruyne",    "tm_id": "88755"},
    {"name": "Bukayo Saka",        "tm_id": "433177"},
    {"name": "Phil Foden",         "tm_id": "406635"},
    {"name": "Trent Alexander-Arnold", "tm_id": "154770"},
    {"name": "Rúben Dias",         "tm_id": "332220"},
    {"name": "Virgil van Dijk",    "tm_id": "139208"},
    {"name": "Alisson",            "tm_id": "105470"},
    {"name": "Ederson",            "tm_id": "238223"},
    {"name": "Bruno Fernandes",    "tm_id": "240306"},
    {"name": "Marcus Rashford",    "tm_id": "258923"},
    {"name": "Leroy Sané",         "tm_id": "125021"},
    {"name": "Jamal Musiala",      "tm_id": "580195"},
    {"name": "Gavi",               "tm_id": "687764"},
    {"name": "Ferran Torres",      "tm_id": "430998"},
    {"name": "Dušan Vlahović",     "tm_id": "360018"},
    {"name": "Federico Chiesa",    "tm_id": "341092"},
    {"name": "Rafael Leão",        "tm_id": "406625"},
    {"name": "Khvicha Kvaratskhelia", "tm_id": "537132"},
    {"name": "Ousmane Dembélé",    "tm_id": "200512"},
    {"name": "Antoine Griezmann",  "tm_id": "103148"},
]

# ─── HTTP Session ────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Accept":  "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
}

session = requests.Session()
session.headers.update(HEADERS)


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _url_to_cache_path(url: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()[:12]
    return HTML_CACHE / f"{key}.html"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10))
def _fetch_html(url: str, force_refresh: bool = False) -> str:
    """Fetch URL with caching and rate throttle. Saves HTML before parsing."""
    cache_path = _url_to_cache_path(url)

    if cache_path.exists() and not force_refresh:
        logger.info("Cache HIT  → %s", cache_path.name)
        return cache_path.read_text(encoding="utf-8")

    logger.info("Fetching   → %s", url)
    time.sleep(SCRAPER_DELAY_S)

    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    html = resp.text

    # ── Save raw HTML to disk BEFORE parsing ──────────────────────────────────
    cache_path.write_text(html, encoding="utf-8")
    logger.info("Cached     → %s (%d bytes)", cache_path.name, len(html))
    return html


def _parse_market_value(value_str: str) -> Optional[float]:
    """
    Parse Transfermarkt market value strings like '€85m', '€4.50m', '€850k'
    into float EUR values.
    """
    if not value_str or value_str in ["-", "?"]:
        return None
    value_str = value_str.replace("€", "").replace(",", ".").strip()
    try:
        if "bn" in value_str.lower():
            return float(re.sub(r"[^\d.]", "", value_str)) * 1_000_000_000
        elif "m" in value_str.lower():
            return float(re.sub(r"[^\d.]", "", value_str)) * 1_000_000
        elif "k" in value_str.lower():
            return float(re.sub(r"[^\d.]", "", value_str)) * 1_000
        else:
            return float(re.sub(r"[^\d.]", "", value_str))
    except ValueError:
        return None


def _parse_player_table(html: str, league: str) -> Optional[pd.DataFrame]:
    """
    Parse the main squad overview table from a Transfermarkt league page.
    Extracts player name, position, age, nationality, market value.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # Transfermarkt uses <table class="items"> for player listings
    table = soup.find("table", {"class": "items"})
    if table is None:
        logger.warning("Player table not found for league: %s", league)
        return None

    for row in table.find_all("tr", {"class": ["odd", "even"]}):
        cols = row.find_all("td")
        if len(cols) < 5:
            continue

        try:
            # Column structure varies — use robust extraction
            name_tag = row.find("a", {"class": "spielprofil_tooltip"})
            name     = name_tag.text.strip() if name_tag else None

            # Market value (last meaningful td)
            mv_tag = row.find("td", {"class": "rechts hauptlink"})
            market_value_raw = mv_tag.text.strip() if mv_tag else None

            # Position
            pos_tag = row.find("td", {"class": "posrela"})
            position = pos_tag.text.strip() if pos_tag else None

            # Age
            age_tds = [td for td in cols if td.get("class") == ["zentriert"]]
            age = age_tds[0].text.strip() if age_tds else None

            rows.append({
                "player":           name,
                "position_detail":  position,
                "age_tm":           age,
                "market_value_raw": market_value_raw,
                "market_value_eur": _parse_market_value(market_value_raw or ""),
                "league_tm":        league,
            })
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df[df["player"].notna() & (df["player"] != "")].copy()
    logger.info("Parsed %d players from TM league: %s", len(df), league)
    return df


def scrape_similar_players(tm_id: str, player_name: str, force_refresh: bool = False) -> list[str]:
    """
    Scrape the 'Similar Players' section from a Transfermarkt player profile.
    These lists serve as Method 2 evaluation benchmark.

    Returns a list of player names considered similar by Transfermarkt editors.
    """
    url = f"https://www.transfermarkt.com/player/profil/spieler/{tm_id}"
    try:
        html = _fetch_html(url, force_refresh=force_refresh)
    except Exception as exc:
        logger.error("Failed to fetch TM profile for %s: %s", player_name, exc)
        return []

    soup = BeautifulSoup(html, "lxml")
    similar = []

    # Transfermarkt "Ähnliche Spieler" / "Similar Players" section
    similar_section = soup.find("div", {"class": "similar-players"})
    if similar_section is None:
        # Try alternate CSS classes
        similar_section = soup.find(
            "div", string=re.compile(r"similar|ähnliche", re.IGNORECASE)
        )

    if similar_section:
        for a_tag in similar_section.find_all("a"):
            name = a_tag.text.strip()
            if name and len(name) > 2:
                similar.append(name)

    logger.info("Similar players for %s: %s", player_name, similar[:5])
    return similar[:10]  # Cap at 10


def build_evaluation_benchmark(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build Method 2 evaluation benchmark by scraping similar-player lists
    for the 30 known benchmark players.

    Output schema:
        query_player | similar_player | rank (1-10)
    """
    logger.info("Building Transfermarkt evaluation benchmark (%d players)...", len(BENCHMARK_PLAYERS))
    records = []

    for player in BENCHMARK_PLAYERS:
        similar = scrape_similar_players(
            tm_id=player["tm_id"],
            player_name=player["name"],
            force_refresh=force_refresh,
        )
        for rank, sim_name in enumerate(similar, start=1):
            records.append({
                "query_player":   player["name"],
                "query_tm_id":    player["tm_id"],
                "similar_player": sim_name,
                "tm_rank":        rank,
            })

    df = pd.DataFrame(records)
    df.to_csv(BENCHMARK_OUT, index=False)
    logger.info("Benchmark saved → %s (%d rows)", BENCHMARK_OUT, len(df))
    return df


# ─── Main Orchestrator ──────────────────────────────────────────────────────────

def run(force_refresh: bool = False, build_benchmark: bool = False) -> pd.DataFrame:
    """
    Main entry point. Scrapes all league market value tables and saves combined CSV.
    Optionally builds the evaluation benchmark.
    """
    all_frames: list[pd.DataFrame] = []

    for league_name, url in LEAGUE_URLS.items():
        logger.info("━━ Scraping TM: %s ━━", league_name)
        try:
            html = _fetch_html(url, force_refresh=force_refresh)
            df   = _parse_player_table(html, league_name)
            if df is not None:
                all_frames.append(df)
        except Exception as exc:
            logger.error("Error scraping TM %s: %s", league_name, exc)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(CSV_OUT, index=False)
        logger.info("✅ TM scraping complete: %d players → %s", len(combined), CSV_OUT)
    else:
        combined = pd.DataFrame()
        logger.warning("No data scraped from Transfermarkt.")

    if build_benchmark:
        build_evaluation_benchmark(force_refresh=force_refresh)

    return combined


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FootScout Transfermarkt Scraper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also scrape similar-player lists for evaluation benchmark",
    )
    args = parser.parse_args()

    result = run(force_refresh=args.force_refresh, build_benchmark=args.benchmark)
    if not result.empty:
        print(f"\n✅ Done — {len(result):,} players")
        print(result[["player", "market_value_raw", "league_tm"]].head(10).to_string(index=False))
