"""
scraper/merge.py
================
FootScout — Fuzzy Join Orchestrator
-------------------------------------
Joins FBref player statistics with Transfermarkt market metadata.

Primary join key: [Player Name + Club Name]
Fuzzy matching: rapidfuzz.fuzz.token_sort_ratio (threshold: 85%)

This resolves spelling variations like:
  - "Vinicius Junior" (FBref) ↔ "Vinícius Júnior" (Transfermarkt)
  - "Alexis Mac Allister" ↔ "Alexis Mac Allister"
  - "Heung-min Son" ↔ "Son Heung-Min"

Output:
  data/processed/players_merged.csv — Master dataset for embedding pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

# ─── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

FBREF_CSV    = DATA_RAW / "fbref_raw.csv"
TM_CSV       = DATA_RAW / "transfermarkt_raw.csv"
MERGED_CSV   = DATA_PROCESSED / "players_merged.csv"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

FUZZY_THRESHOLD = int(85)  # Minimum token_sort_ratio score to accept a match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("footscout.merge")


# ─── Name Normalization ─────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Normalize player names for robust matching:
    - Remove diacritics-to-ASCII where safe (é→e, ü→u, etc.)
    - Lowercase
    - Strip extra whitespace
    - Handle common abbreviations
    """
    import unicodedata
    if not isinstance(name, str):
        return ""
    # NFD decomposition + ASCII stripping (best-effort diacritic removal)
    nfd = unicodedata.normalize("NFD", name)
    ascii_name = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return ascii_name.lower().strip()


def _build_match_key(player: str, club: str) -> str:
    """Composite key for fuzzy matching: 'player_name | club_name'"""
    return f"{normalize_name(player)} | {normalize_name(club)}"


# ─── Fuzzy Join Engine ──────────────────────────────────────────────────────────

def fuzzy_join(
    fbref_df: pd.DataFrame,
    tm_df: pd.DataFrame,
    threshold: int = FUZZY_THRESHOLD,
) -> pd.DataFrame:
    """
    Join FBref and Transfermarkt DataFrames using rapidfuzz token_sort_ratio.

    Strategy:
    1. Build composite key "player | club" for both datasets.
    2. For each FBref player, find the best-matching Transfermarkt key.
    3. If score >= threshold, merge the Transfermarkt columns.
    4. Unmatched FBref players are kept (market_value = NaN).

    Returns:
        Merged DataFrame with FBref stats + TM metadata columns.
    """
    logger.info(
        "Starting fuzzy join: %d FBref × %d TM players",
        len(fbref_df), len(tm_df)
    )

    # ── Identify column names (flexible schema) ────────────────────────────────
    fbref_name_col = _detect_col(fbref_df, ["player", "name"])
    fbref_club_col = _detect_col(fbref_df, ["squad", "club", "team"])
    tm_name_col    = _detect_col(tm_df, ["player", "name"])
    tm_club_col    = _detect_col(tm_df, ["squad", "club", "team", "league_tm"])

    # Build TM keys: "player | club" → index in tm_df
    tm_keys: dict[str, int] = {}
    for idx, row in tm_df.iterrows():
        player = str(row.get(tm_name_col, ""))
        club   = str(row.get(tm_club_col, "")) if tm_club_col else ""
        key = _build_match_key(player, club)
        tm_keys[key] = int(idx)  # type: ignore[arg-type]

    tm_key_list = list(tm_keys.keys())

    # ── Iterative fuzzy matching ───────────────────────────────────────────────
    matched_indices: list[Optional[int]] = []
    match_scores:    list[Optional[float]] = []
    match_strategies: list[str] = []

    for _, fb_row in fbref_df.iterrows():
        fb_player = str(fb_row.get(fbref_name_col, ""))
        fb_club   = str(fb_row.get(fbref_club_col, "")) if fbref_club_col else ""
        fb_key    = _build_match_key(fb_player, fb_club)

        # Strategy 1: Full "player | club" key match
        result = process.extractOne(
            fb_key,
            tm_key_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )

        if result is not None:
            best_key, score, _ = result
            matched_indices.append(tm_keys[best_key])
            match_scores.append(score)
            match_strategies.append("full_key")
            continue

        # Strategy 2: Player name only (fallback for club name mismatches)
        fb_name_only = normalize_name(fb_player)
        tm_names_only = {
            normalize_name(str(tm_df.loc[idx, tm_name_col])): idx
            for idx in tm_df.index
        }
        result2 = process.extractOne(
            fb_name_only,
            list(tm_names_only.keys()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=max(threshold + 5, 90),  # Stricter for name-only match
        )

        if result2 is not None:
            best_name, score2, _ = result2
            matched_indices.append(tm_names_only[best_name])
            match_scores.append(score2)
            match_strategies.append("name_only")
        else:
            matched_indices.append(None)
            match_scores.append(None)
            match_strategies.append("no_match")

    # ── Assemble results ───────────────────────────────────────────────────────
    fbref_copy = fbref_df.copy().reset_index(drop=True)
    fbref_copy["_tm_idx"]      = matched_indices
    fbref_copy["_match_score"] = match_scores
    fbref_copy["_match_strategy"] = match_strategies

    matched    = fbref_copy[fbref_copy["_tm_idx"].notna()].copy()
    unmatched  = fbref_copy[fbref_copy["_tm_idx"].isna()].copy()

    logger.info(
        "Match results: %d matched (%.1f%%), %d unmatched",
        len(matched), 100 * len(matched) / max(len(fbref_copy), 1), len(unmatched),
    )

    # TM columns to bring across (avoid column collision)
    tm_meta_cols = [
        c for c in [
            "market_value_eur", "market_value_raw", "position_detail",
            "age_tm", "contract_expires", "nationality_tm",
        ]
        if c in tm_df.columns
    ]

    # Left-join matched rows
    tm_subset = tm_df[tm_meta_cols].copy().reset_index()
    tm_subset.rename(columns={"index": "_tm_idx"}, inplace=True)

    matched = matched.merge(tm_subset, on="_tm_idx", how="left")

    # Combine matched + unmatched
    merged = pd.concat([matched, unmatched], ignore_index=True)

    # Drop internal helper columns
    merged.drop(columns=["_tm_idx"], inplace=True, errors="ignore")

    logger.info("Final merged dataset: %d rows", len(merged))
    return merged


# ─── Column Detection Helpers ────────────────────────────────────────────────────

def _detect_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column that exists in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ─── Data Quality Post-Merge ────────────────────────────────────────────────────

def run_post_merge_quality_check(df: pd.DataFrame) -> None:
    """
    Log a summary of data quality after the merge.
    Covers completeness (missing rates) and match coverage.
    """
    logger.info("\n─── Post-Merge Quality Check ───────────────────────────")

    # Missing rate by column
    missing_rates = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing  = missing_rates[missing_rates > 20]
    if not high_missing.empty:
        logger.warning("Columns with >20%% missing values:")
        for col, rate in high_missing.items():
            logger.warning("  %-35s %.1f%%", col, rate)

    # Match coverage
    if "_match_strategy" in df.columns:
        strategy_counts = df["_match_strategy"].value_counts()
        logger.info("Match strategy breakdown:\n%s", strategy_counts.to_string())

    # Market value coverage
    if "market_value_eur" in df.columns:
        mv_coverage = df["market_value_eur"].notna().mean() * 100
        logger.info("Market value coverage: %.1f%%", mv_coverage)

    logger.info("────────────────────────────────────────────────────────")


# ─── Main Orchestrator ──────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    """
    Main entry point.
    Loads FBref and Transfermarkt CSVs, runs fuzzy join, saves merged dataset.
    """
    # ── Load inputs ────────────────────────────────────────────────────────────
    if not FBREF_CSV.exists():
        raise FileNotFoundError(
            f"FBref data not found at {FBREF_CSV}. Run fbref_scraper.py first."
        )

    fbref_df = pd.read_csv(FBREF_CSV, low_memory=False)
    logger.info("Loaded FBref: %d players", len(fbref_df))

    if TM_CSV.exists():
        tm_df = pd.read_csv(TM_CSV, low_memory=False)
        logger.info("Loaded Transfermarkt: %d players", len(tm_df))
    else:
        logger.warning(
            "Transfermarkt CSV not found at %s — proceeding without market values.", TM_CSV
        )
        tm_df = pd.DataFrame()

    # ── Fuzzy join ─────────────────────────────────────────────────────────────
    if not tm_df.empty:
        merged = fuzzy_join(fbref_df, tm_df)
    else:
        merged = fbref_df.copy()
        merged["market_value_eur"] = None

    # ── Quality check ──────────────────────────────────────────────────────────
    run_post_merge_quality_check(merged)

    # ── Save ───────────────────────────────────────────────────────────────────
    merged.to_csv(MERGED_CSV, index=False)
    logger.info("✅ Merged dataset saved → %s (%d players)", MERGED_CSV, len(merged))

    return merged


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run()
    print(f"\n✅ Merge complete: {len(result):,} players")

    key_cols = [
        c for c in [
            "player", "squad", "league", "pos", "age",
            "market_value_eur", "_match_score", "_match_strategy"
        ]
        if c in result.columns
    ]
    print(result[key_cols].head(10).to_string(index=False))
