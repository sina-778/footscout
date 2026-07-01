"""
src/recommender.py
==================
FootScout — Cosine Similarity Recommender Engine
-------------------------------------------------
Implements three recommendation modes over the hybrid embedding matrix:

  Mode 1 — Standard Similarity:
    Top-k players most similar to a query player (globally).

  Mode 2 — Budget-Aware Replacement:
    Top-k similar players where market_value_eur ≤ budget.
    Use case: Find affordable alternatives to a star player.

  Mode 3 — Hidden Gem Finder:
    Players with high style match (cosine similarity) but low market value.
    Use case: Undervalued players matching a target profile.

All modes return a ranked DataFrame with similarity scores and key metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ─── Setup ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("footscout.recommender")


# ─── Recommender Core ────────────────────────────────────────────────────────────

class FootScoutRecommender:
    """
    Content-based football player recommender using cosine similarity.

    Usage
    -----
    >>> rec = FootScoutRecommender(df=players_df, embeddings=hybrid_emb)
    >>> results = rec.find_similar("Erling Haaland", k=5)
    >>> results = rec.find_budget_replacement("Erling Haaland", budget=30_000_000, k=5)
    >>> results = rec.find_hidden_gems(position="FW", max_value=10_000_000, k=5)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        player_col: str = "player",
        position_col: str = "pos",
        market_value_col: str = "market_value_eur",
        squad_col: str = "squad",
        league_col: str = "league",
    ):
        """
        Args:
            df:          Player metadata DataFrame (from merge.py output)
            embeddings:  Hybrid embedding matrix (n_players, n_dims)
            player_col:  Column name for player names
            position_col: Column name for player positions
            market_value_col: Column name for market values (EUR)
        """
        assert len(df) == embeddings.shape[0], (
            f"DataFrame ({len(df)} rows) and embeddings ({embeddings.shape[0]}) "
            "must have the same number of rows."
        )

        self.df               = df.reset_index(drop=True).copy()
        self.embeddings       = embeddings
        self.player_col       = player_col
        self.position_col     = position_col
        self.market_value_col = market_value_col
        self.squad_col        = squad_col
        self.league_col       = league_col

        # Build name → index lookup (case-insensitive)
        self._name_to_idx: dict[str, int] = {
            str(name).lower(): idx
            for idx, name in enumerate(self.df[player_col])
        }

        logger.info(
            "Recommender loaded: %d players, embedding shape %s",
            len(df), embeddings.shape,
        )

    # ─── Name Resolution ────────────────────────────────────────────────────────

    def _resolve_player(self, query: str) -> int:
        """
        Return the DataFrame index for a player name.
        Supports exact and partial matches (case-insensitive).

        Raises ValueError if no match found.
        """
        query_lower = query.lower().strip()

        # 1. Exact match
        if query_lower in self._name_to_idx:
            return self._name_to_idx[query_lower]

        # 2. Partial match (substring)
        partial = [k for k in self._name_to_idx if query_lower in k]
        if len(partial) == 1:
            return self._name_to_idx[partial[0]]
        if len(partial) > 1:
            logger.warning("Ambiguous player query '%s'. Candidates: %s", query, partial[:5])
            return self._name_to_idx[partial[0]]

        # 3. Fuzzy fallback (rapidfuzz)
        try:
            from rapidfuzz import process as rfuzz_process, fuzz as rfuzz
            
            # Try last-name-only matching first (better for short queries like "mesi")
            last_names = {}
            for name in self.df[self.player_col].dropna():
                name_str = str(name)
                if ' ' in name_str:
                    last = name_str.split()[-1].lower()
                    last_names[last] = name_str.lower()
            result = rfuzz_process.extractOne(
                query_lower,
                list(last_names.keys()),
                scorer=rfuzz.WRatio,
                score_cutoff=55,
            )
            if result:
                best_last, score, _ = result
                if score >= 60:
                    best_name = last_names[best_last]
                    logger.info("Fuzzy match (last name): '%s' → '%s' (score=%d)", query, best_name, score)
                    return self._name_to_idx[best_name]
            
            # Fall back to full name matching
            result = rfuzz_process.extractOne(
                query_lower,
                list(self._name_to_idx.keys()),
                scorer=rfuzz.WRatio,
                score_cutoff=55,
            )
            if result:
                best_name, score, _ = result
                logger.info("Fuzzy match (full name): '%s' → '%s' (score=%d)", query, best_name, score)
                return self._name_to_idx[best_name]
        except ImportError:
            pass

        raise ValueError(
            f"Player '{query}' not found in the dataset. "
            "Try a partial name or check spelling."
        )

    def _compute_similarity_scores(self, query_idx: int) -> np.ndarray:
        """
        Compute cosine similarity between the query player and all others.
        Returns a 1D array of shape (n_players,).
        """
        query_emb = self.embeddings[query_idx].reshape(1, -1)
        scores    = cosine_similarity(query_emb, self.embeddings)[0]
        return scores

    def _format_results(
        self,
        scores: np.ndarray,
        top_indices: np.ndarray,
        query_idx: int,
    ) -> pd.DataFrame:
        """
        Build a clean result DataFrame from top-k indices.
        Excludes the query player from results.
        """
        # Remove query player from candidates
        top_indices = [i for i in top_indices if i != query_idx]

        if not top_indices:
            return pd.DataFrame()

        results = self.df.iloc[top_indices].copy()
        
        # Add rank and similarity columns
        results.insert(0, "rank", range(1, len(results) + 1))
        results["similarity"] = [round(float(scores[idx]), 4) for idx in top_indices]
        
        if "position" not in results.columns and "pos" in results.columns:
            results["position"] = results["pos"]

        return results.reset_index(drop=True)

    # ─── Mode 1: Standard Similarity ────────────────────────────────────────────

    def find_similar(
        self,
        player: str,
        k: int = 10,
        position_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        MODE 1: Find the k most similar players globally.

        Args:
            player:           Query player name (fuzzy-matched)
            k:                Number of results to return (3–20 recommended)
            position_filter:  Optional position code to restrict results (e.g., "FW")

        Returns:
            DataFrame with columns: rank, player, similarity, position, squad, ...
        """
        logger.info("[Mode 1] Similar to: %s (k=%d)", player, k)
        query_idx = self._resolve_player(player)
        scores    = self._compute_similarity_scores(query_idx)

        # Apply optional position filter
        if position_filter:
            mask = self.df[self.position_col].str.contains(
                position_filter, case=False, na=False
            )
            candidate_indices = np.where(mask)[0]
        else:
            candidate_indices = np.arange(len(self.df))

        # Sort by score descending
        sorted_by_score = sorted(
            candidate_indices, key=lambda i: scores[i], reverse=True
        )
        top_k = sorted_by_score[:k + 1]  # +1 to account for self-exclusion

        return self._format_results(scores, top_k, query_idx).head(k)

    # ─── Mode 2: Budget-Aware Replacement ───────────────────────────────────────

    def find_budget_replacement(
        self,
        player: str,
        budget: float,
        k: int = 5,
        same_position: bool = True,
    ) -> pd.DataFrame:
        """
        MODE 2: Find similar players within a given market value budget.

        Useful for scouts looking to replace a player within budget constraints.

        Args:
            player:         Query player name
            budget:         Maximum market value in EUR (e.g., 30_000_000)
            k:              Number of results
            same_position:  If True, restrict to same broad position group

        Returns:
            DataFrame sorted by similarity score, filtered to budget ≤ budget.
        """
        logger.info(
            "[Mode 2] Budget replacement for: %s | Budget: €%.1fM | k=%d",
            player, budget / 1e6, k,
        )
        query_idx = self._resolve_player(player)
        scores    = self._compute_similarity_scores(query_idx)

        # Build budget mask
        if self.market_value_col in self.df.columns:
            mv_series = pd.to_numeric(self.df[self.market_value_col], errors="coerce")
            budget_mask = (mv_series <= budget) | mv_series.isna()
        else:
            logger.warning("No market value column found — budget filter not applied")
            budget_mask = pd.Series([True] * len(self.df))

        # Optional: same position group
        if same_position and self.position_col in self.df.columns:
            query_pos = str(self.df.loc[query_idx, self.position_col])
            broad_pos = _broad_position(query_pos)
            pos_mask  = self.df[self.position_col].apply(
                lambda p: _broad_position(str(p)) == broad_pos
            )
        else:
            pos_mask = pd.Series([True] * len(self.df))

        combined_mask = budget_mask & pos_mask
        candidate_indices = np.where(combined_mask.values)[0]

        sorted_candidates = sorted(
            candidate_indices, key=lambda i: scores[i], reverse=True
        )
        top_k = sorted_candidates[:k + 1]

        result = self._format_results(scores, top_k, query_idx).head(k)
        logger.info("Budget results: %d candidates under €%.1fM", len(result), budget / 1e6)
        return result

    # ─── Mode 3: Hidden Gem Finder ───────────────────────────────────────────────

    def find_hidden_gems(
        self,
        position: str,
        max_value: float,
        style_description: Optional[str] = None,
        reference_player: Optional[str] = None,
        min_similarity: float = 0.50,
        k: int = 5,
    ) -> pd.DataFrame:
        """
        MODE 3: Find undervalued players (hidden gems).

        A hidden gem is a player with:
        - Low market value (≤ max_value)
        - High cosine similarity to the target profile (≥ min_similarity)
        - Correct position group

        The query can be defined either by a reference player or by computing
        a centroid of the target position group in embedding space.

        Args:
            position:           Position filter ("FW", "MF", "DF", etc.)
            max_value:          Maximum market value in EUR
            style_description:  Free-text description (encoded via text embedder)
            reference_player:   Query by similarity to a reference player
            min_similarity:     Minimum cosine similarity threshold
            k:                  Number of gems to return

        Returns:
            DataFrame of hidden gems sorted by similarity (descending).
        """
        logger.info(
            "[Mode 3] Hidden gems | Position: %s | Max value: €%.1fM | k=%d",
            position, max_value / 1e6, k,
        )

        # ── Determine query embedding ──────────────────────────────────────────
        if reference_player:
            query_idx  = self._resolve_player(reference_player)
            query_emb  = self.embeddings[query_idx].reshape(1, -1)
            logger.info("Query embedding: reference player '%s'", reference_player)
        else:
            # Centroid of target position group as fallback query
            pos_mask = self.df[self.position_col].str.contains(
                position, case=False, na=False
            )
            if not pos_mask.any():
                raise ValueError(f"No players found for position '{position}'")
            query_emb = self.embeddings[pos_mask.values].mean(axis=0, keepdims=True)
            logger.info(
                "Query embedding: centroid of %d '%s' players",
                pos_mask.sum(), position
            )

        scores = cosine_similarity(query_emb, self.embeddings)[0]

        # ── Apply position + value + similarity filters ─────────────────────────
        pos_mask = self.df[self.position_col].str.contains(
            position, case=False, na=False
        )

        if self.market_value_col in self.df.columns:
            mv_series   = pd.to_numeric(self.df[self.market_value_col], errors="coerce")
            value_mask  = (mv_series <= max_value)
        else:
            value_mask = pd.Series([True] * len(self.df))

        sim_mask = pd.Series(scores >= min_similarity)

        combined_mask = pos_mask & value_mask & sim_mask
        candidate_indices = np.where(combined_mask.values)[0]

        if len(candidate_indices) == 0:
            logger.warning(
                "No hidden gems found with current filters. "
                "Try lowering min_similarity (%.2f) or raising max_value.", min_similarity
            )
            return pd.DataFrame()

        sorted_candidates = sorted(
            candidate_indices, key=lambda i: scores[i], reverse=True
        )
        top_k = sorted_candidates[:k + 1]

        query_idx_for_fmt = (
            self._resolve_player(reference_player) if reference_player else -1
        )

        result = self._format_results(scores, top_k, query_idx_for_fmt).head(k)
        logger.info("Found %d hidden gems matching criteria", len(result))
        return result


# ─── Utility Functions ───────────────────────────────────────────────────────────

def _broad_position(pos_code: str) -> str:
    """
    Map detailed position codes to broad groups for budget-mode filtering.

    FBref positions: GK, CB, LB, RB, LWB, RWB, DM, CM, AM, LM, RM, LW, RW, CF, SS
    """
    pos_code = pos_code.upper()
    if "GK" in pos_code:
        return "GK"
    elif any(p in pos_code for p in ["CB", "LB", "RB", "WB"]):
        return "DEF"
    elif any(p in pos_code for p in ["DM", "CM", "AM", "LM", "RM"]):
        return "MID"
    elif any(p in pos_code for p in ["LW", "RW", "CF", "ST", "SS", "FW"]):
        return "FWD"
    return "UNK"


def get_radar_data(
    player_name: str,
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    position_avg: bool = True,
) -> dict:
    """
    ★ REUSABLE RADAR CHART UTILITY ★
    ════════════════════════════════
    Extract normalized radar chart data for a player.

    This function is designed to be fully reusable between:
    - Notebook 02_eda.ipynb (exploratory visualizations)
    - app/streamlit_app.py (live interactive comparisons)

    Args:
        player_name:  Player to visualize
        df:           Player DataFrame (merged dataset)
        features:     Stat columns to include (default: key per-90 metrics)
        position_avg: If True, also compute the position-average benchmark line

    Returns:
        dict with keys:
          - "player_name": str
          - "categories":  list[str]  (feature display names)
          - "values":      list[float] (player values, 0–1 normalized)
          - "avg_values":  list[float] (position average, 0–1 normalized)
          - "raw_values":  dict        (un-normalized values for tooltips)
    """
    DEFAULT_RADAR_FEATURES = [
        "gls_per90", "ast_per90", "xg_per90",
        "prog_carries_per90", "prog_passes_per90",
        "tackles_tkl_per90", "int_per90",
        "pass_completion_pct",
    ]
    features = features or [f for f in DEFAULT_RADAR_FEATURES if f in df.columns]

    if not features:
        raise ValueError("No valid radar features found in DataFrame.")

    # ── Find player row ────────────────────────────────────────────────────────
    name_col = next((c for c in ["player", "name"] if c in df.columns), None)
    if name_col is None:
        raise ValueError("No player name column found.")

    mask = df[name_col].str.lower() == player_name.lower()
    if not mask.any():
        # Partial match
        mask = df[name_col].str.lower().str.contains(player_name.lower(), na=False)

    if not mask.any():
        raise ValueError(f"Player '{player_name}' not found.")

    player_row = df[mask].iloc[0]

    # ── Raw values ─────────────────────────────────────────────────────────────
    raw_values = {}
    for f in features:
        raw_values[f] = float(pd.to_numeric(player_row.get(f, 0), errors="coerce") or 0.0)

    # ── Position-group average (benchmark line) ────────────────────────────────
    avg_raw: dict[str, float] = {}
    if position_avg and "pos" in df.columns:
        player_pos = str(player_row.get("pos", ""))
        broad_pos  = _broad_position(player_pos)
        pos_mask   = df["pos"].apply(lambda p: _broad_position(str(p)) == broad_pos)
        pos_group  = df[pos_mask][features].apply(pd.to_numeric, errors="coerce")
        for f in features:
            avg_raw[f] = float(pos_group[f].mean()) if f in pos_group.columns else 0.0

    # ── 0–1 Normalization per feature (relative to dataset max) ───────────────
    def _normalize(val: float, col: str) -> float:
        feat_series = pd.to_numeric(df[col], errors="coerce")
        col_max = feat_series.quantile(0.95)  # Use 95th percentile to avoid outlier distortion
        if col_max == 0 or pd.isna(col_max):
            return 0.0
        return min(float(val / col_max), 1.0)

    normalized_values  = [_normalize(raw_values[f], f)     for f in features]
    normalized_avg     = [_normalize(avg_raw.get(f, 0), f) for f in features]

    # ── Human-readable category labels ────────────────────────────────────────
    LABEL_MAP = {
        "gls_per90":          "Goals/90",
        "ast_per90":          "Assists/90",
        "xg_per90":           "xG/90",
        "npxg_per90":         "npxG/90",
        "xag_per90":          "xAG/90",
        "prog_carries_per90": "Prog. Carries/90",
        "prog_passes_per90":  "Prog. Passes/90",
        "tackles_tkl_per90":  "Tackles/90",
        "int_per90":          "Interceptions/90",
        "clr_per90":          "Clearances/90",
        "pass_completion_pct":"Pass Completion %",
        "sh_per90":           "Shots/90",
        "sot_per90":          "SoT/90",
        "touches_per90":      "Touches/90",
    }
    categories = [LABEL_MAP.get(f, f.replace("_per90", "/90").replace("_", " ").title()) for f in features]

    return {
        "player_name": str(player_row.get(name_col, player_name)),
        "categories":  categories,
        "values":      normalized_values,
        "avg_values":  normalized_avg,
        "raw_values":  raw_values,
        "features":    features,
    }


def make_radar_figure(radar_data: dict, show_average: bool = True):
    """
    ★ REUSABLE RADAR CHART VISUALIZER ★
    ════════════════════════════════════
    Create a Plotly polar/radar chart from get_radar_data() output.

    Designed for direct reuse in both EDA notebooks and Streamlit frontend.

    Args:
        radar_data:   Output dict from get_radar_data()
        show_average: If True, overlay position-average benchmark line

    Returns:
        plotly.graph_objects.Figure ready for fig.show() or st.plotly_chart()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Install plotly: pip install plotly")

    categories = radar_data["categories"]
    values     = radar_data["values"]
    avg_values = radar_data["avg_values"]
    player_name = radar_data["player_name"]
    raw_values  = radar_data["raw_values"]
    features    = radar_data["features"]

    # Close the radar polygon
    cats_closed   = categories + [categories[0]]
    vals_closed   = values + [values[0]]
    avg_closed    = avg_values + [avg_values[0]]

    # Hover text with raw values
    hover_player = [
        f"{cat}<br>Value: {raw_values.get(feat, 0):.3f}"
        for cat, feat in zip(categories, features)
    ] + [f"{categories[0]}<br>Value: {raw_values.get(features[0], 0):.3f}"]

    fig = go.Figure()

    # ── Player trace ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        name=player_name,
        hoverinfo="text",
        hovertext=hover_player,
        line=dict(color="#6C63FF", width=2.5),
        fillcolor="rgba(108, 99, 255, 0.2)",
    ))

    # ── Position average trace ─────────────────────────────────────────────────
    if show_average and avg_values:
        fig.add_trace(go.Scatterpolar(
            r=avg_closed,
            theta=cats_closed,
            fill="toself",
            name="Position Average",
            hoverinfo="skip",
            line=dict(color="#00D2A8", width=1.5, dash="dot"),
            fillcolor="rgba(0, 210, 168, 0.08)",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode="array",
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100%"],
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.3)",
                tickfont=dict(size=9, color="rgba(255,255,255,0.6)"),
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="white"),
                gridcolor="rgba(255,255,255,0.1)",
            ),
            bgcolor="rgba(15, 17, 26, 0.0)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title=dict(
            text=f"<b>{player_name}</b> — Stat Profile",
            font=dict(size=16, color="white"),
            x=0.5,
        ),
        margin=dict(t=60, b=60, l=60, r=60),
        height=480,
    )

    return fig


# ─── Convenience Loader ──────────────────────────────────────────────────────────

def load_recommender(
    method: str = "umap",
    alpha: float = 0.6,
    df: Optional[pd.DataFrame] = None,
) -> FootScoutRecommender:
    """
    Convenience function: load embeddings from disk and initialize recommender.

    Args:
        method: Embedding method ("pca" or "umap")
        alpha:  Blending weight used when building embeddings
        df:     Player DataFrame (loaded from data/processed/ if not provided)

    Returns:
        Ready-to-use FootScoutRecommender instance
    """
    from src.embeddings import load_embeddings

    if df is None:
        merged_path = DATA_PROCESSED / "players_merged.csv"
        if not merged_path.exists():
            raise FileNotFoundError(
                f"Player data not found at {merged_path}. Run the scraper pipeline first."
            )
        df = pd.read_csv(merged_path, low_memory=False)

    results = load_embeddings(method=method, alpha=alpha)
    return FootScoutRecommender(df=df, embeddings=results["hybrid"])


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FootScout Recommender CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--player",   required=True, help="Query player name")
    parser.add_argument("--mode",     default="similar",
                        choices=["similar", "budget", "gem"])
    parser.add_argument("--k",        type=int, default=5)
    parser.add_argument("--budget",   type=float, default=30_000_000,
                        help="Budget in EUR (for budget mode)")
    parser.add_argument("--position", default="FW",
                        help="Position filter (for gem mode)")
    parser.add_argument("--max-value",type=float, default=15_000_000,
                        help="Max market value EUR (for gem mode)")
    parser.add_argument("--method",   default="umap", choices=["pca", "umap"])
    parser.add_argument("--alpha",    type=float, default=0.6)
    args = parser.parse_args()

    rec = load_recommender(method=args.method, alpha=args.alpha)

    if args.mode == "similar":
        results = rec.find_similar(args.player, k=args.k)
    elif args.mode == "budget":
        results = rec.find_budget_replacement(args.player, budget=args.budget, k=args.k)
    else:  # gem
        results = rec.find_hidden_gems(
            position=args.position, max_value=args.max_value, k=args.k
        )

    print(f"\n🔍 Results for: {args.player} (Mode: {args.mode})\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(results.to_string(index=False))
