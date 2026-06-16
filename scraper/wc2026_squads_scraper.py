"""
scraper/wc2026_squads_scraper.py
====================================
Scrapes all 48 World Cup 2026 squad lists from Wikipedia and merges with
the existing players_merged.csv.

Run: python scraper/wc2026_squads_scraper.py
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("footscout.wc2026_squads")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "players_merged.csv"

HEADERS = {
    "User-Agent": "FootScout/1.0 (BHT Berlin DS Project; educational; https://github.com/sina-778/footscout)",
    "Accept-Language": "en-US,en;q=0.9",
}

WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

# Map full country names (as they appear on Wikipedia) → ISO-3 codes used in the dataset
COUNTRY_TO_ISO = {
    "Czech Republic": "CZE", "Mexico": "MEX", "South Africa": "RSA",
    "South Korea": "KOR", "Bosnia and Herzegovina": "BIH", "Canada": "CAN",
    "Qatar": "QAT", "Switzerland": "SUI", "Brazil": "BRA", "Haiti": "HAI",
    "Morocco": "MAR", "Scotland": "SCO", "Australia": "AUS", "Paraguay": "PAR",
    "Turkey": "TUR", "United States": "USA", "Curaçao": "CUW", "Ecuador": "ECU",
    "Germany": "GER", "Ivory Coast": "CIV", "Japan": "JPN", "Netherlands": "NED",
    "Sweden": "SWE", "Tunisia": "TUN", "Belgium": "BEL", "Egypt": "EGY",
    "Iran": "IRN", "New Zealand": "NZL", "Cape Verde": "CPV", "Saudi Arabia": "SAU",
    "Spain": "ESP", "Uruguay": "URU", "France": "FRA", "Iraq": "IRQ",
    "Norway": "NOR", "Senegal": "SEN", "Algeria": "ALG", "Argentina": "ARG",
    "Austria": "AUT", "Jordan": "JOR", "Colombia": "COL", "DR Congo": "COD",
    "Portugal": "POR", "Uzbekistan": "UZB", "Croatia": "CRO", "England": "ENG",
    "Ghana": "GHA", "Panama": "PAN",
    # Sometimes listed differently
    "United States of America": "USA", "South Korea (Republic of Korea)": "KOR",
    "Côte d'Ivoire": "CIV", "Bosnia-Herzegovina": "BIH",
}

# Full country name for display
ISO_TO_FULL_NAME = {v: k for k, v in COUNTRY_TO_ISO.items()}

# Position code mapping (Wikipedia uses "1GK", "2DF", "3MF", "4FW")
POS_MAP = {
    "1GK": "GK", "GK": "GK",
    "2DF": "DF", "DF": "DF",
    "3MF": "MF", "MF": "MF",
    "4FW": "FW", "FW": "FW",
}


def scrape_all_squads() -> list[dict]:
    """Scrape all squads from the single Wikipedia page."""
    logger.info("Fetching %s ...", WIKI_URL)
    r = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    
    soup = BeautifulSoup(r.text, "html.parser")
    all_players = []
    
    skip_headings = {
        "Statistics", "Age", "Player representation by club",
        "Player representation by league system",
        "Player representation by club confederation",
        "Average age of squads", "Coach representation by country",
        "Contents", "References", "Notes", "External links",
        "See also",
    }
    
    for heading in soup.find_all(["h2", "h3"]):
        country_name = heading.get_text(strip=True)
        
        if country_name in skip_headings or country_name.startswith("Group "):
            continue
        
        iso = COUNTRY_TO_ISO.get(country_name)
        if not iso:
            logger.debug("Unknown country: %s (skipping)", country_name)
            continue
        
        # Find the wikitable immediately following this heading
        table = heading.find_next("table", class_="wikitable")
        if not table:
            logger.debug("No table for %s", country_name)
            continue
        
        rows = table.find_all("tr")
        country_players = []
        
        for row in rows[1:]:  # skip header row
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            
            cell_texts = [c.get_text(strip=True) for c in cells]
            
            pos = None
            name = None
            club = None
            
            for i, txt in enumerate(cell_texts):
                mapped = POS_MAP.get(txt)
                if mapped:
                    pos = mapped
                    # Player name is the next cell
                    if i + 1 < len(cell_texts):
                        raw = cell_texts[i + 1]
                        # Strip captain marker "(c)" or footnote markers
                        name = re.sub(r'\s*\([^)]*\)\s*', '', raw).strip()
                        name = re.sub(r'\s*\[[^\]]*\]\s*', '', name).strip()
                    # Club is typically 2 cells after position
                    if i + 3 < len(cell_texts):
                        club = cell_texts[i + 3]
                    break
            
            if name and pos and len(name) > 1:
                country_players.append({
                    "player": name,
                    "pos": pos,
                    "nation": iso,
                    "squad": club or "",
                    "nationality_tm": country_name,
                    "is_world_cup": 1,
                })
        
        logger.debug("%-25s → %d players (ISO: %s)", country_name, len(country_players), iso)
        all_players.extend(country_players)
    
    logger.info("Scraped %d players from %d nations", len(all_players), len(set(p["nation"] for p in all_players)))
    return all_players


def get_position_averages(df: pd.DataFrame) -> dict:
    """Calculate average stats per position group from existing data."""
    stat_cols = [
        "gls_per90", "ast_per90", "xg_per90", "npxg_per90", "xag_per90",
        "sh_per90", "sot_per90", "touches_per90", "prog_carries_per90",
        "prog_passes_per90", "tackles_tkl_per90", "int_per90", "clr_per90",
        "pass_completion_pct", "playing_time_min", "market_value_eur",
    ]
    avgs = {}
    for pos in ["GK", "DF", "MF", "FW"]:
        pos_df = df[df["pos"].fillna("").str.upper().str.startswith(pos)]
        avgs[pos] = {
            col: float(pos_df[col].mean()) if col in pos_df.columns else 0.0
            for col in stat_cols
        }
    return avgs


def main():
    if not CSV_PATH.exists():
        logger.error("CSV not found: %s", CSV_PATH)
        return
    
    df_existing = pd.read_csv(CSV_PATH)
    logger.info("Existing dataset: %d players", len(df_existing))
    
    all_players = scrape_all_squads()
    
    existing_names_lower = set(df_existing["player"].str.lower().str.strip())
    pos_avgs = get_position_averages(df_existing)
    
    # Find truly new players
    new_players = []
    for p in all_players:
        if p["player"].lower().strip() not in existing_names_lower:
            new_players.append(p)
            existing_names_lower.add(p["player"].lower().strip())
    
    # Also ensure all WC players have is_world_cup=1
    wc_names_lower = {p["player"].lower().strip() for p in all_players}
    df_existing.loc[
        df_existing["player"].str.lower().str.strip().isin(wc_names_lower),
        "is_world_cup"
    ] = 1
    
    logger.info("New players to add: %d", len(new_players))
    
    if not new_players:
        df_existing.to_csv(CSV_PATH, index=False)
        logger.info("No new players found. Updated is_world_cup flags and saved.")
        return
    
    # Build rows for new players with estimated stats
    new_rows = []
    for p in new_players:
        pos_key = p.get("pos", "MF")[:2].upper()
        avgs = pos_avgs.get(pos_key, pos_avgs.get("MF", {}))
        
        row = {col: None for col in df_existing.columns}
        row.update({
            "player": p["player"],
            "squad": p.get("squad", ""),
            "league": "",
            "pos": p["pos"],
            "nation": p["nation"],
            "nationality_tm": p.get("nationality_tm", ""),
            "is_world_cup": 1,
            "_match_strategy": "wc2026_wiki",
        })
        # Fill stats with position averages (conservative 70% scale)
        for col, val in avgs.items():
            row[col] = val * 0.7 if col != "pass_completion_pct" else val
        
        new_rows.append(row)
    
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["player"], keep="first")
    
    df_combined.to_csv(CSV_PATH, index=False)
    logger.info(
        "\n✅ Dataset expanded: %d → %d players (+%d new WC 2026 players)",
        len(df_existing), len(df_combined), len(new_rows)
    )
    logger.info("Nations covered: %s", sorted(set(p["nation"] for p in all_players)))


if __name__ == "__main__":
    main()
