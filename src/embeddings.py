"""
src/embeddings.py
==================
FootScout — Embedding Pipeline
--------------------------------
Builds hybrid embeddings combining:
  1. Statistical vector: 20–25 per-90 metrics → PCA / UMAP → compressed vector
  2. Text embedding: player profile template → Sentence-Transformers (all-MiniLM-L6-v2)
  3. Hybrid concatenation: [alpha * stat_emb, (1-alpha) * text_emb]

W&B logging is integrated throughout — hyperparameters, embedding matrices,
and evaluation metrics are all tracked as artifacts.

Design notes
------------
- PCA is always computed first (fast baseline, interpretable)
- UMAP is the improved variant (preserves local structure better)
- Optional: shallow autoencoder stub provided for extension
- alpha parameter is tunable via Optuna (see notebooks/03_embeddings.ipynb)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Optional heavy imports — loaded lazily to avoid slow startup
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import joblib

# ─── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("footscout.embeddings")

# ─── Feature Groups (20–25 per-90 stats, grouped by football role) ──────────────

#: Attacking metrics
ATTACK_FEATURES = [
    "gls_per90", "ast_per90", "xg_per90", "npxg_per90",
    "xag_per90", "sot_per90", "sh_per90",
]

#: Passing & creativity
PASSING_FEATURES = [
    "prog_passes_per90", "xa_per90",
    "kp",           # Key passes (already per-match in FBref)
    "pass_completion_pct",
]

#: Ball progression
PROGRESSION_FEATURES = [
    "prog_carries_per90",
    "att_3rd_per90",
    "touches_per90",
]

#: Defensive metrics
DEFENSIVE_FEATURES = [
    "tackles_tkl_per90", "tkl_w_per90",
    "blocks_per90", "int_per90", "clr_per90",
]

#: All 24 features (used as default feature set)
ALL_STAT_FEATURES: list[str] = (
    ATTACK_FEATURES
    + PASSING_FEATURES
    + PROGRESSION_FEATURES
    + DEFENSIVE_FEATURES
)

# ─── Player Profile Template ─────────────────────────────────────────────────────
PROFILE_TEMPLATE = (
    "{name} is a {age}-year-old {position} playing for {club} in the {league}. "
    "He scores {goals:.2f} goals per 90 minutes, records {assists:.2f} assists per 90, "
    "completes {pass_pct:.0f}% of passes, generates {xg:.2f} xG per 90, "
    "and makes {tackles:.2f} tackles per 90 minutes."
)


# ─── Statistical Embedding ──────────────────────────────────────────────────────

class StatisticalEmbedder:
    """
    Builds a compressed statistical embedding from per-90 features.

    Supports three reduction methods:
    - "pca"  : Fast, interpretable baseline (recommended starting point)
    - "umap" : Preserves local manifold structure (requires umap-learn)
    - "none" : Raw standardized features (for debugging)
    """

    def __init__(
        self,
        method: Literal["pca", "umap", "none"] = "pca",
        n_components: int = 32,
        features: Optional[list[str]] = None,
        # UMAP-specific hyperparameters (tunable via Optuna)
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
    ):
        self.method          = method
        self.n_components    = n_components
        self.features        = features or ALL_STAT_FEATURES
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist   = umap_min_dist
        self.random_state    = random_state

        self._scaler:   Optional[StandardScaler] = None
        self._reducer   = None           # PCA or UMAP object
        self._is_fitted = False

    def _build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and impute the feature matrix from the DataFrame."""
        available = [f for f in self.features if f in df.columns]
        missing   = set(self.features) - set(available)
        if missing:
            logger.warning(
                "Missing features (will be zero-filled): %s",
                ", ".join(sorted(missing)),
            )

        X = df.reindex(columns=available).copy()
        # Add zero-columns for missing features to maintain shape
        for f in self.features:
            if f not in X.columns:
                X[f] = 0.0

        X = X[self.features]  # Ensure consistent column order
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X.values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the scaler and reducer on df, then return normalized embeddings.

        DATA QUALITY: NaN values are filled with 0 (neutral per-90 value).
        """
        X_raw = self._build_feature_matrix(df)

        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X_raw)

        if self.method == "pca":
            self._reducer = PCA(n_components=self.n_components, random_state=self.random_state)
            embeddings = self._reducer.fit_transform(X_scaled).astype(np.float32)
            explained  = self._reducer.explained_variance_ratio_.sum()
            logger.info("PCA: %d components explain %.2f%% variance", self.n_components, explained * 100)

        elif self.method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError("umap-learn is not installed. Run: pip install umap-learn")
            self._reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                random_state=self.random_state,
                metric="euclidean",
            )
            embeddings = self._reducer.fit_transform(X_scaled).astype(np.float32)
            logger.info("UMAP: %d components, %d neighbors", self.n_components, self.umap_n_neighbors)

        else:  # "none" — return raw scaled features
            embeddings = X_scaled.astype(np.float32)

        self._is_fitted = True
        return embeddings

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted scaler and reducer."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform first.")
        X_raw    = self._build_feature_matrix(df)
        X_scaled = self._scaler.transform(X_raw)
        if self.method in ("pca", "umap"):
            return self._reducer.transform(X_scaled).astype(np.float32)
        return X_scaled.astype(np.float32)

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted embedder (scaler + reducer) to disk."""
        path = path or (EMBEDDINGS_DIR / f"stat_embedder_{self.method}.pkl")
        joblib.dump({"scaler": self._scaler, "reducer": self._reducer, "meta": self.__dict__}, path)
        logger.info("StatEmbedder saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "StatisticalEmbedder":
        """Load a previously saved embedder."""
        data = joblib.load(path)
        obj  = cls.__new__(cls)
        obj.__dict__.update(data["meta"])
        obj._scaler   = data["scaler"]
        obj._reducer  = data["reducer"]
        obj._is_fitted = True
        return obj


# ─── Text Embedding ─────────────────────────────────────────────────────────────

class TextEmbedder:
    """
    Generates semantic embeddings from natural-language player profiles.

    Uses Sentence-Transformers (all-MiniLM-L6-v2) — a lightweight 384-dim
    model well-suited for semantic similarity tasks.

    The player profile template is defined in PROFILE_TEMPLATE above.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        self.model_name = model_name
        
        # Check for Apple Silicon GPU (MPS) or standard CUDA GPU
        import torch
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
            
        logger.info("Loading SentenceTransformer: %s on device: %s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def build_profile(row: pd.Series) -> str:
        """
        Generate a natural-language description for one player row.
        Gracefully handles missing values with neutral defaults.
        """
        def safe_float(val: object, default: float = 0.0) -> float:
            import math
            try:
                f = float(val)
                if math.isnan(f):
                    return default
                return f
            except (TypeError, ValueError):
                return default

        return PROFILE_TEMPLATE.format(
            name=row.get("player", "Unknown Player"),
            age=int(safe_float(row.get("age", None), 25)),
            position=row.get("pos", "Unknown Position"),
            club=row.get("squad", "Unknown Club"),
            league=row.get("league", "Unknown League"),
            goals=safe_float(row.get("gls_per90", row.get("gls", 0))),
            assists=safe_float(row.get("ast_per90", row.get("ast", 0))),
            pass_pct=safe_float(row.get("pass_completion_pct", row.get("cmp_pct", 0))),
            xg=safe_float(row.get("xg_per90", row.get("xg", 0))),
            tackles=safe_float(row.get("tackles_tkl_per90", row.get("tackles_tkl", 0))),
        )

    def build_all_profiles(self, df: pd.DataFrame) -> list[str]:
        """Generate profile strings for every player in the DataFrame."""
        profiles = [self.build_profile(row) for _, row in df.iterrows()]
        logger.info("Generated %d player profiles", len(profiles))
        return profiles

    def encode(self, df: pd.DataFrame, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Encode all player profiles to 384-dim sentence embeddings.

        Args:
            df: DataFrame with player rows
            batch_size: Sentence-Transformers batch size
            show_progress: Show tqdm progress bar

        Returns:
            np.ndarray of shape (n_players, 384)
        """
        profiles = self.build_all_profiles(df)
        logger.info("Encoding %d profiles with %s...", len(profiles), self.model_name)

        embeddings = self._model.encode(
            profiles,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
        )
        logger.info("Text embeddings shape: %s", embeddings.shape)
        return embeddings.astype(np.float32)

    def encode_single(self, profile_text: str) -> np.ndarray:
        """Encode a single profile string (for real-time Streamlit queries)."""
        return self._model.encode(
            [profile_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)


# ─── Hybrid Embedding ────────────────────────────────────────────────────────────

def build_hybrid_embedding(
    stat_emb: np.ndarray,
    text_emb: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Construct the hybrid embedding vector.

    Formula: hybrid = [alpha * stat_emb_normalized, (1-alpha) * text_emb_normalized]

    Args:
        stat_emb: Statistical embedding matrix (n_players, n_stat_dims)
        text_emb: Text embedding matrix (n_players, n_text_dims)
        alpha:    Blending weight for statistical component (default 0.6)
                  0.0 → pure text, 1.0 → pure stats, 0.6 → recommended default

    Returns:
        np.ndarray of shape (n_players, n_stat_dims + n_text_dims)
    """
    assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
    assert stat_emb.shape[0] == text_emb.shape[0], "Row count mismatch"

    # L2-normalize each component independently (if not already done)
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # Avoid division by zero
        return X / norms

    stat_norm = _l2_normalize(stat_emb)
    text_norm = _l2_normalize(text_emb)

    hybrid = np.concatenate(
        [alpha * stat_norm, (1.0 - alpha) * text_norm],
        axis=1,
    ).astype(np.float32)

    logger.info(
        "Hybrid embedding: α=%.2f, shape=%s (stat=%d + text=%d dims)",
        alpha, hybrid.shape, stat_emb.shape[1], text_emb.shape[1],
    )
    return hybrid


# ─── Full Pipeline ──────────────────────────────────────────────────────────────

def build_embeddings(
    df: pd.DataFrame,
    method: Literal["pca", "umap"] = "umap",
    n_components: int = 32,
    alpha: float = 0.6,
    wandb_run=None,
    save: bool = True,
) -> dict[str, np.ndarray]:
    """
    Full embedding pipeline: statistical + text + hybrid.

    Args:
        df:          Merged player DataFrame (output of scraper/merge.py)
        method:      Dimensionality reduction method ("pca" or "umap")
        n_components: Target embedding dimensions for stat component
        alpha:        Hybrid blending weight (stat vs text)
        wandb_run:    Active W&B run for logging (optional)
        save:         If True, save all matrices to embeddings/

    Returns:
        dict with keys: "stat", "text", "hybrid", "player_index"
    """
    logger.info("═══ Building Embeddings ═══════════════════════════════")
    logger.info("Method: %s | n_components: %d | alpha: %.2f", method, n_components, alpha)

    # ── 1. Statistical Embedding ───────────────────────────────────────────────
    stat_embedder = StatisticalEmbedder(method=method, n_components=n_components)
    stat_emb      = stat_embedder.fit_transform(df)

    # ── 2. Text Embedding ──────────────────────────────────────────────────────
    text_embedder = TextEmbedder()
    text_emb      = text_embedder.encode(df)

    # ── 3. Hybrid ─────────────────────────────────────────────────────────────
    hybrid_emb = build_hybrid_embedding(stat_emb, text_emb, alpha=alpha)

    # ── 4. Player Index (for lookup) ───────────────────────────────────────────
    name_col   = next((c for c in ["player", "name"] if c in df.columns), None)
    player_idx = df[name_col].tolist() if name_col else list(range(len(df)))

    results = {
        "stat":         stat_emb,
        "text":         text_emb,
        "hybrid":       hybrid_emb,
        "player_index": player_idx,
    }

    # ── 5. W&B Logging ────────────────────────────────────────────────────────
    if wandb_run is not None and WANDB_AVAILABLE:
        _log_to_wandb(
            wandb_run, results, method=method,
            n_components=n_components, alpha=alpha,
        )

    # ── 6. Save to Disk ───────────────────────────────────────────────────────
    if save:
        _save_embeddings(results, method=method, alpha=alpha)
        stat_embedder.save()

    return results


def _save_embeddings(results: dict, method: str, alpha: float) -> None:
    """Persist embedding matrices and player index to embeddings/"""
    tag = f"{method}_a{int(alpha*100)}"

    for key in ["stat", "text", "hybrid"]:
        path = EMBEDDINGS_DIR / f"{key}_{tag}.npy"
        np.save(path, results[key])
        logger.info("Saved %s embeddings → %s", key, path)

    idx_path = EMBEDDINGS_DIR / f"player_index_{tag}.pkl"
    joblib.dump(results["player_index"], idx_path)
    logger.info("Saved player index → %s", idx_path)


def _log_to_wandb(wandb_run, results: dict, **config) -> None:
    """Log embedding metadata and matrices as W&B artifacts."""
    wandb_run.config.update(config)
    wandb_run.log({
        "stat_emb_dims":  results["stat"].shape[1],
        "text_emb_dims":  results["text"].shape[1],
        "hybrid_emb_dims": results["hybrid"].shape[1],
        "n_players":      len(results["player_index"]),
    })

    # Log matrices as W&B artifacts
    artifact = wandb.Artifact(
        name=f"embeddings_{config.get('method', 'unknown')}",
        type="embeddings",
        description=f"Hybrid embedding matrix (alpha={config.get('alpha')})",
    )
    for key in ["stat", "text", "hybrid"]:
        tmp_path = EMBEDDINGS_DIR / f"_wandb_tmp_{key}.npy"
        np.save(tmp_path, results[key])
        artifact.add_file(str(tmp_path))

    wandb_run.log_artifact(artifact)
    logger.info("W&B: Logged embedding artifact for run %s", wandb_run.id)


# ─── Load Saved Embeddings ──────────────────────────────────────────────────────

def load_embeddings(
    method: str = "umap",
    alpha: float = 0.6,
) -> dict[str, np.ndarray]:
    """
    Load pre-computed embedding matrices from disk. If the hybrid matrix for the
    requested alpha doesn't exist, compute it dynamically from stat and text matrices.

    Returns:
        dict with keys: "stat", "text", "hybrid", "player_index"
    """
    tag = f"{method}_a{int(alpha*100)}"
    results = {}

    exact_hybrid_path = EMBEDDINGS_DIR / f"hybrid_{tag}.npy"
    exact_index_path = EMBEDDINGS_DIR / f"player_index_{tag}.pkl"

    if exact_hybrid_path.exists() and exact_index_path.exists():
        # Load precomputed exact alpha matrices
        for key in ["stat", "text", "hybrid"]:
            path = EMBEDDINGS_DIR / f"{key}_{tag}.npy"
            results[key] = np.load(path)
            logger.info("Loaded precomputed %s embeddings: shape=%s", key, results[key].shape)
        results["player_index"] = joblib.load(exact_index_path)
        return results

    # Fallback: find any existing tag matching method (e.g. tag with another alpha like a60)
    # to load the base stat, text, and player index, and compute hybrid dynamically.
    import glob
    stat_pattern = str(EMBEDDINGS_DIR / f"stat_{method}_a*.npy")
    matches = glob.glob(stat_pattern)

    if not matches:
        raise FileNotFoundError(
            f"No base embeddings found for method '{method}' at {EMBEDDINGS_DIR}.\n"
            f"Run the embedding pipeline first: python src/embeddings.py --method {method}"
        )

    # Use first match to extract the tag structure
    existing_tag = Path(matches[0]).stem.replace("stat_", "")
    stat_path = EMBEDDINGS_DIR / f"stat_{existing_tag}.npy"
    text_path = EMBEDDINGS_DIR / f"text_{existing_tag}.npy"
    idx_path = EMBEDDINGS_DIR / f"player_index_{existing_tag}.pkl"

    if not (stat_path.exists() and text_path.exists() and idx_path.exists()):
        raise FileNotFoundError(
            f"Base embedding files incomplete for tag '{existing_tag}'."
        )

    logger.info("Computing hybrid embedding dynamically for alpha=%.2f using base '%s'", alpha, existing_tag)
    stat_emb = np.load(stat_path)
    text_emb = np.load(text_path)
    player_index = joblib.load(idx_path)

    # Compute hybrid on the fly using build_hybrid_embedding
    hybrid_emb = build_hybrid_embedding(stat_emb, text_emb, alpha=alpha)

    return {
        "stat": stat_emb,
        "text": text_emb,
        "hybrid": hybrid_emb,
        "player_index": player_index,
    }


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build FootScout embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method",       default="umap", choices=["pca", "umap", "none"])
    parser.add_argument("--n-components", type=int,   default=32)
    parser.add_argument("--alpha",        type=float, default=0.6)
    parser.add_argument("--data",         type=str,   default=str(DATA_PROCESSED / "players_merged.csv"))
    parser.add_argument("--wandb",        action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    logger.info("Loaded %d players from %s", len(df), args.data)

    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(project="footscout", job_type="embeddings")

    results = build_embeddings(
        df,
        method=args.method,
        n_components=args.n_components,
        alpha=args.alpha,
        wandb_run=wandb_run,
    )

    print(f"\n✅ Embeddings built:")
    print(f"   stat  : {results['stat'].shape}")
    print(f"   text  : {results['text'].shape}")
    print(f"   hybrid: {results['hybrid'].shape}")

    if wandb_run:
        wandb_run.finish()
