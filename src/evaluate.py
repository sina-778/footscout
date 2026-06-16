"""
src/evaluate.py
================
FootScout — Offline Evaluation Engine
---------------------------------------
Computes retrieval evaluation metrics:
  - Precision@k
  - Recall@k
  - NDCG@k (Normalized Discounted Cumulative Gain)

Two evaluation methods:
  Method 1 — Position-based ground truth:
    Assumption: a good recommender for a CM should return other CMs.
    Automatically derives ground truth from position groups.

  Method 2 — Transfermarkt benchmark:
    Human-curated "similar players" lists from Transfermarkt editors.
    Scraped by transfermarkt_scraper.py and stored in data/raw/transfermarkt_benchmark.csv.

W&B logging: all metric tables and comparison runs are logged as artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── Setup ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

BENCHMARK_CSV  = DATA_RAW / "transfermarkt_benchmark.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("footscout.evaluate")


# ─── Core Metric Functions ──────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Precision@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved: Ordered list of retrieved player names (top-k results)
        relevant:  Set of ground-truth relevant player names
        k:         Cut-off rank

    Returns:
        Precision@k in [0, 1]
    """
    if k == 0:
        return 0.0
    top_k_retrieved = retrieved[:k]
    hits = len(set(top_k_retrieved) & relevant)
    return hits / k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved: Ordered list of retrieved player names
        relevant:  Set of ground-truth relevant player names
        k:         Cut-off rank

    Returns:
        Recall@k in [0, 1]
    """
    if not relevant:
        return 0.0
    top_k_retrieved = retrieved[:k]
    hits = len(set(top_k_retrieved) & relevant)
    return hits / len(relevant)


def dcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Discounted Cumulative Gain @ k.
    Binary relevance: 1 if in relevant set, 0 otherwise.
    """
    score = 0.0
    for rank, player in enumerate(retrieved[:k], start=1):
        rel = 1 if player in relevant else 0
        score += rel / np.log2(rank + 1)
    return score


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    NDCG@k = DCG@k / IDCG@k

    IDCG@k = ideal DCG (assumes all k relevant items are ranked perfectly).

    Args:
        retrieved: Ordered list of retrieved player names
        relevant:  Set of ground-truth relevant player names
        k:         Cut-off rank

    Returns:
        NDCG@k in [0, 1]
    """
    dcg  = dcg_at_k(retrieved, relevant, k)
    # Ideal: min(|relevant|, k) hits in perfect order
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def f1_at_k(retrieved: list, relevant: set, k: int) -> float:
    """F1@k = harmonic mean of Precision@k and Recall@k."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ─── Method 1: Position-Based Ground Truth ──────────────────────────────────────

class PositionBasedEvaluator:
    """
    METHOD 1: Evaluate recommender using position groups as ground truth.

    Ground truth definition:
    For a query player in position group X, all other players in X are "relevant".

    Position groups:
    - GK:     Goalkeepers
    - CB:     Centre-backs
    - FB:     Full-backs (LB, RB, LWB, RWB)
    - CDM:    Defensive midfielders (DM)
    - CM:     Central midfielders
    - CAM:    Attacking midfielders / 10s
    - Winger: Wide attackers (LW, RW)
    - Striker: Centre-forwards (CF, ST, SS)
    """

    POSITION_GROUPS: dict[str, list[str]] = {
        "GK":     ["GK"],
        "CB":     ["CB"],
        "FB":     ["LB", "RB", "LWB", "RWB"],
        "CDM":    ["DM"],
        "CM":     ["CM"],
        "CAM":    ["AM"],
        "Winger": ["LW", "RW"],
        "Striker":["CF", "ST", "SS", "FW"],
    }

    def __init__(self, df: pd.DataFrame, position_col: str = "pos"):
        self.df           = df.reset_index(drop=True)
        self.position_col = position_col
        self._build_group_index()

    def _build_group_index(self) -> None:
        """Map each player index to their position group."""
        self._player_to_group: dict[int, str] = {}
        for group, codes in self.POSITION_GROUPS.items():
            for code in codes:
                mask = self.df[self.position_col].str.contains(
                    code, case=False, na=False
                )
                for idx in mask[mask].index:
                    self._player_to_group[int(idx)] = group

    def get_relevant_set(self, query_idx: int) -> set[str]:
        """
        Return the set of player names relevant for the query player
        (all players in the same position group, excluding the query).
        """
        name_col = next(
            (c for c in ["player", "name"] if c in self.df.columns), None
        )
        group = self._player_to_group.get(query_idx)
        if group is None:
            return set()

        relevant_indices = [
            i for i, g in self._player_to_group.items()
            if g == group and i != query_idx
        ]
        if name_col:
            return set(self.df.loc[relevant_indices, name_col].tolist())
        return set(str(i) for i in relevant_indices)

    def evaluate(
        self,
        recommender,
        k_values: list[int] = [3, 5, 10],
        n_queries: Optional[int] = None,
        embedding_type: str = "hybrid",
    ) -> pd.DataFrame:
        """
        Run evaluation over a sample of query players.

        Args:
            recommender: FootScoutRecommender instance
            k_values:    List of k values to evaluate at [3, 5, 10]
            n_queries:   Number of random query players (None = all)
            embedding_type: Label for W&B logging ("stat", "text", "hybrid")

        Returns:
            DataFrame with columns: position_group, k, precision, recall, ndcg, f1
        """
        name_col = next(
            (c for c in ["player", "name"] if c in self.df.columns), None
        )
        all_indices = list(self._player_to_group.keys())

        if n_queries and n_queries < len(all_indices):
            np.random.seed(42)
            query_indices = list(np.random.choice(all_indices, n_queries, replace=False))
        else:
            query_indices = all_indices

        logger.info(
            "[Method 1] Evaluating %d query players × k=%s", len(query_indices), k_values
        )

        records = []
        for query_idx in query_indices:
            if name_col is None:
                continue
            player_name = self.df.loc[query_idx, name_col]
            relevant    = self.get_relevant_set(query_idx)
            group       = self._player_to_group.get(query_idx, "unknown")

            if not relevant:
                continue

            try:
                results = recommender.find_similar(
                    player_name, k=max(k_values)
                )
                retrieved = results["player"].tolist()
            except Exception as exc:
                logger.debug("Skipping %s: %s", player_name, exc)
                continue

            for k in k_values:
                records.append({
                    "embedding_type":  embedding_type,
                    "eval_method":     "position_based",
                    "query_player":    player_name,
                    "position_group":  group,
                    "k":               k,
                    "precision_at_k":  precision_at_k(retrieved, relevant, k),
                    "recall_at_k":     recall_at_k(retrieved, relevant, k),
                    "ndcg_at_k":       ndcg_at_k(retrieved, relevant, k),
                    "f1_at_k":         f1_at_k(retrieved, relevant, k),
                    "n_relevant":      len(relevant),
                })

        df_results = pd.DataFrame(records)
        logger.info("[Method 1] Completed: %d evaluation records", len(df_results))
        return df_results

    def summarize(self, results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results by position group and k."""
        return (
            results
            .groupby(["position_group", "k"])[
                ["precision_at_k", "recall_at_k", "ndcg_at_k", "f1_at_k"]
            ]
            .mean()
            .round(4)
            .reset_index()
        )


# ─── Method 2: Transfermarkt Benchmark ──────────────────────────────────────────

class TransfermarktBenchmarkEvaluator:
    """
    METHOD 2: Evaluate against human-curated Transfermarkt "similar players" lists.

    Ground truth: If Transfermarkt says Player X is similar to Player Y,
    a good recommender should rank Y in the top-k for X.

    Benchmark file: data/raw/transfermarkt_benchmark.csv
    Schema:  query_player | similar_player | tm_rank
    """

    def __init__(self, benchmark_df: Optional[pd.DataFrame] = None):
        if benchmark_df is not None:
            self.benchmark = benchmark_df
        elif BENCHMARK_CSV.exists():
            self.benchmark = pd.read_csv(BENCHMARK_CSV)
            logger.info(
                "Loaded TM benchmark: %d records, %d query players",
                len(self.benchmark),
                self.benchmark["query_player"].nunique(),
            )
        else:
            logger.warning(
                "Transfermarkt benchmark CSV not found at %s. "
                "Run: python scraper/transfermarkt_scraper.py --benchmark",
                BENCHMARK_CSV,
            )
            self.benchmark = pd.DataFrame(
                columns=["query_player", "similar_player", "tm_rank"]
            )

    def get_relevant_set(self, query_player: str) -> set[str]:
        """Return the TM-curated similar players for the query."""
        mask = (
            self.benchmark["query_player"].str.lower()
            == query_player.lower()
        )
        return set(self.benchmark[mask]["similar_player"].tolist())

    def evaluate(
        self,
        recommender,
        k_values: list[int] = [3, 5, 10],
        embedding_type: str = "hybrid",
    ) -> pd.DataFrame:
        """
        Evaluate recommender against TM benchmark.

        Args:
            recommender: FootScoutRecommender instance
            k_values:    Evaluation cut-offs
            embedding_type: Label for logging

        Returns:
            DataFrame with evaluation records
        """
        if self.benchmark.empty:
            logger.error("Benchmark is empty — cannot evaluate.")
            return pd.DataFrame()

        query_players = self.benchmark["query_player"].unique().tolist()
        logger.info(
            "[Method 2] Evaluating %d TM benchmark players × k=%s",
            len(query_players), k_values,
        )

        records = []
        for player_name in query_players:
            relevant = self.get_relevant_set(player_name)
            if not relevant:
                continue

            try:
                results   = recommender.find_similar(player_name, k=max(k_values))
                retrieved = results["player"].tolist()
            except Exception as exc:
                logger.debug("Skipping %s: %s", player_name, exc)
                continue

            for k in k_values:
                records.append({
                    "embedding_type":  embedding_type,
                    "eval_method":     "transfermarkt_benchmark",
                    "query_player":    player_name,
                    "k":               k,
                    "precision_at_k":  precision_at_k(retrieved, relevant, k),
                    "recall_at_k":     recall_at_k(retrieved, relevant, k),
                    "ndcg_at_k":       ndcg_at_k(retrieved, relevant, k),
                    "f1_at_k":         f1_at_k(retrieved, relevant, k),
                    "n_relevant":      len(relevant),
                })

        df_results = pd.DataFrame(records)
        logger.info("[Method 2] Completed: %d evaluation records", len(df_results))
        return df_results


# ─── Comparison Experiment Runner ────────────────────────────────────────────────

def run_comparison_experiment(
    df: pd.DataFrame,
    embedding_variants: dict[str, np.ndarray],
    k_values: list[int] = [3, 5, 10],
    n_queries: int = 100,
    wandb_run=None,
) -> pd.DataFrame:
    """
    Run both evaluation methods across multiple embedding variants.

    This produces the final comparison table:
    | Embedding Type | Method         | k | Precision@k | Recall@k | NDCG@k |

    Args:
        df:                 Player DataFrame
        embedding_variants: Dict of {"stat": emb, "text": emb, "hybrid": emb}
        k_values:           Evaluation cut-offs
        n_queries:          Number of random queries for Method 1
        wandb_run:          Active W&B run for logging (optional)

    Returns:
        Combined comparison DataFrame
    """
    from src.recommender import FootScoutRecommender

    all_results: list[pd.DataFrame] = []
    pos_evaluator = PositionBasedEvaluator(df)
    tm_evaluator  = TransfermarktBenchmarkEvaluator()

    for emb_type, embeddings in embedding_variants.items():
        logger.info("═══ Evaluating: %s ═══════════════════", emb_type)
        rec = FootScoutRecommender(df=df, embeddings=embeddings)

        # Method 1: Position-based
        m1 = pos_evaluator.evaluate(
            rec, k_values=k_values, n_queries=n_queries, embedding_type=emb_type
        )
        all_results.append(m1)

        # Method 2: Transfermarkt benchmark
        if not tm_evaluator.benchmark.empty:
            m2 = tm_evaluator.evaluate(
                rec, k_values=k_values, embedding_type=emb_type
            )
            all_results.append(m2)

    combined = pd.concat(all_results, ignore_index=True)

    # ── W&B Logging ─────────────────────────────────────────────────────────────
    if wandb_run is not None:
        try:
            import wandb

            # Log summary table
            summary = (
                combined
                .groupby(["embedding_type", "eval_method", "k"])[
                    ["precision_at_k", "recall_at_k", "ndcg_at_k"]
                ]
                .mean()
                .round(4)
                .reset_index()
            )
            wandb_run.log({"evaluation_summary": wandb.Table(dataframe=summary)})

            # Per-k metrics
            for k in k_values:
                k_data = combined[combined["k"] == k]
                for emb_type in embedding_variants:
                    emb_data = k_data[k_data["embedding_type"] == emb_type]
                    if not emb_data.empty:
                        wandb_run.log({
                            f"{emb_type}/precision_at_{k}": emb_data["precision_at_k"].mean(),
                            f"{emb_type}/recall_at_{k}":    emb_data["recall_at_k"].mean(),
                            f"{emb_type}/ndcg_at_{k}":      emb_data["ndcg_at_k"].mean(),
                        })

            logger.info("W&B: Evaluation metrics logged")
        except Exception as exc:
            logger.warning("W&B logging failed: %s", exc)

    return combined


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FootScout Evaluation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method",     default="umap", choices=["pca", "umap"])
    parser.add_argument("--alpha",      type=float, default=0.6)
    parser.add_argument("--k",          nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--n-queries",  type=int, default=100)
    parser.add_argument("--wandb",      action="store_true")
    args = parser.parse_args()

    from src.embeddings  import load_embeddings
    from src.recommender import FootScoutRecommender

    df   = pd.read_csv(DATA_PROCESSED / "players_merged.csv", low_memory=False)
    embs = load_embeddings(method=args.method, alpha=args.alpha)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="footscout", job_type="evaluation")
        except Exception:
            pass

    results = run_comparison_experiment(
        df=df,
        embedding_variants={
            "stat":   embs["stat"],
            "text":   embs["text"],
            "hybrid": embs["hybrid"],
        },
        k_values=args.k,
        n_queries=args.n_queries,
        wandb_run=wandb_run,
    )

    summary = (
        results
        .groupby(["embedding_type", "eval_method", "k"])[
            ["precision_at_k", "recall_at_k", "ndcg_at_k"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )

    print("\n📊 Evaluation Summary:\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    if wandb_run:
        wandb_run.finish()
