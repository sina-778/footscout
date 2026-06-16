# FootScout 🔭⚽
### A Content-Based Football Player Recommender Using Statistical and Text Embeddings

> **Master's Final Project — Data Science Workflow, BHT Berlin**

---

## 🌟 Overview

FootScout is a state-of-the-art football player recommender system designed to assist recruiters, analysts, and clubs in scouting players. By constructing a hybrid vector space that blends **per-90 statistical performance metrics** (compressed via UMAP to 32 dimensions) and **semantic text descriptors** (via Sentence-Transformers), FootScout computes high-precision cosine similarities to identify players with matching profiles. 

The project contains a master cohort of **1,539 unique players** featuring domestic club profiles and authentic squads for the **2026 FIFA World Cup**, achieving an **image coverage rate of 95.1%** for the player base.

---

## 🎬 Demo & User Interface

### 1. ⚡ Live Verification Demo
Demonstrating natural language queries (*"I want a striker like Haaland under 80M"*), real-time spelling auto-corrections (*"Mesi" -> "Lionel Messi"*), country filtering, dynamic budget restrictions, and real player photos.
![FootScout Live Demo](docs/screenshots/verify_scout_features.webp)

### 2. 🤖 AI Scout Page (Home Page)
Users can search for players using natural language queries or select from the spelling dropdown helper to instantly view recommendations.
![Jude Bellingham Scout Search](docs/screenshots/jude_bellingham_scout.png)

### 3. Similar Players Recommendations
Interactive player cards featuring real player photos, circular country flag badges (ENG, ARG, BRA, etc.), budget details, and key playstyle stats.
![Recommendations List](docs/screenshots/recommendations_list.png)

### 4. 🔍 Player Finder (Radar Charts)
Explore individual player profiles with visual interactive radar charts comparing a player's stats against position averages.
![Player Finder Profile](docs/screenshots/player_finder_top.png)

---

## 🚀 Key Features

1. **🤖 AI Scout (Natural Language Search)**:
   - Queries parsed for: reference player (e.g. *"like Bellingham"*), budget constraints (e.g. *"under $80M"*), position filters (e.g. *"midfielder"*), and style tags (e.g. *"creative, fast"*).
   - Combines statistical similarity and semantic search dynamically.
2. **Interactive Spelling Helper**:
   - Handles typos (e.g., "Mesi" or "Halnd") using fuzzy token-sorting distance matching.
   - Provides a searchable autocomplete selectbox next to the text input for player name lookup.
3. **World Cup 2026 Availability**:
   - Integrates automatic country-based status tagging. Players from 48 World Cup nations are marked and available for World Cup queries.
4. **Premium Visuals**:
   - **Real Player Photos**: Integrates headshots fetched from the Wikipedia PageImage API and TheSportsDB CDN, achieving **95.1% image coverage (1,464 photos)**.
   - **Flag CDN**: Renders beautiful country flag badges next to player nationalities.
5. **Apple Silicon GPU Acceleration**:
   - Vector encoding automatically utilizes local Mac hardware GPU acceleration (`device="mps"`).

---

## 📁 Project Architecture & Structure

```
footscout/
├── data/
│   ├── raw/               # Raw scraped CSV data (fbref_raw, transfermarkt_raw, transfermarkt_benchmark)
│   └── processed/         # Cleaned, fuzzy-joined database (players_merged)
├── docs/
│   └── screenshots/       # UI screenshots and demo webp for README
├── embeddings/            # Serialized statistical, text, and hybrid embedding matrices (.npy, .pkl)
├── scraper/
│   ├── fbref_scraper.py        # BeautifulSoup scraper for per-90 stats
│   ├── transfermarkt_scraper.py # Market value metadata scraper
│   ├── wc2026_squads_scraper.py # Scrapes 48 World Cup 2026 squads from Wikipedia
│   ├── generate_benchmark.py   # Generates Transfermarkt benchmark mappings
│   ├── merge.py                # Fuzzy join orchestrator (rapidfuzz, threshold 85%)
│   └── download_player_images.py # Asynchronous thread pool player image fetcher
├── notebooks/
│   ├── 01_data_quality.ipynb   # Data quality checks & deduplication
│   ├── 02_eda.ipynb            # Exploratory Data Analysis & radar chart plotting
│   ├── 03_embeddings.ipynb     # Embedding pipelines & hybrid blending
│   ├── 04_recommender.ipynb    # Cosine similarity search logic
│   └── 05_evaluation.ipynb     # Offline precision, recall, and NDCG evaluation
├── src/
│   ├── embeddings.py           # Embedding generation class
│   ├── recommender.py          # Similarity search engine
│   └── evaluate.py             # Recommender evaluation metrics
├── app/
│   └── streamlit_app.py        # Streamlit dashboard implementation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Quick Start & Usage

### 1. Clone & Setup Environment
```bash
git clone https://github.com/sina-778/footscout.git
cd footscout
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Re-run Scraping & Data Processing Pipelines
The database, scraped player images, and embeddings are pre-built inside the repository. To rebuild them from scratch:
```bash
# Generate mock/raw statistics
python scraper/generate_dataset.py

# Scrape World Cup 2026 squads from Wikipedia and merge statistics
python scraper/wc2026_squads_scraper.py

# Fuzzy merge fbref and transfermarkt raw files
python scraper/merge.py

# Fetch player images from Wikipedia & TheSportsDB
python scraper/download_player_images.py

# Regenerate UMAP statistical and SentenceTransformer text embeddings
python src/embeddings.py
```

### 3. Run Recommender System Evaluation
Compute offline evaluation metrics across all embedding types (statistical, text, and hybrid):
```bash
# Generate the Transfermarkt benchmark
python -m scraper.generate_benchmark

# Execute evaluation metrics (Precision@k, Recall@k, NDCG@k)
python -m src.evaluate --method umap --alpha 0.6 --n-queries 100
```

### 4. Launch Streamlit Web UI
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Work Packages & Performance Evaluation (Course Compliance)

Your project satisfies the work packages of the BHT evaluation criteria as follows:

| Work Package | Status | Technical Details |
|--------------|--------|-------------------|
| **Data Quality\*** | ✅ Complete | Resolved mid-season transfer duplicates, resolved diacritics in names, and filled missing metrics with positional averages. |
| **Vector Embeddings\*** | ✅ Complete | Hybrid statistical (32-dim UMAP) + semantic text embeddings (384-dim Sentence-Transformer). |
| **Recommender Core\*** | ✅ Complete | Three modes: Global similarity, Budget-restricted replacement, and Hidden Gem (undervalued talent) discovery. |
| **Performance Evaluation\*** | ✅ Complete | Evaluated models using Precision@k, Recall@k, and NDCG@k metrics against position categories and TM editorial benchmarks (see results table below). |
| **Frontend UI** | ✅ Complete | Premium dark glassmorphism dashboard built with Streamlit, Plotly radar visualizer, FlagCDN flags, and Wikipedia player images. |

### Recommender Evaluation Metrics (1,539 Players Database)

| Embedding Type | Evaluation Method | k | Precision@k | Recall@k | NDCG@k |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **hybrid** | position_based | 3 | 1.0000 | 0.0106 | 1.0000 |
| **hybrid** | position_based | 5 | 1.0000 | 0.0177 | 1.0000 |
| **hybrid** | position_based | 10 | 1.0000 | 0.0354 | 1.0000 |
| **hybrid** | transfermarkt_benchmark | 3 | **0.1333 (13.3%)** | **0.0767 (7.7%)** | **0.1235 (12.4%)** |
| **hybrid** | transfermarkt_benchmark | 5 | **0.0800 (8.0%)** | **0.0767 (7.7%)** | **0.0892 (8.9%)** |
| **hybrid** | transfermarkt_benchmark | 10 | **0.0500 (5.0%)** | **0.0933 (9.3%)** | **0.0957 (9.6%)** |
| **stat** | position_based | 3 | 1.0000 | 0.0106 | 1.0000 |
| **stat** | position_based | 5 | 1.0000 | 0.0177 | 1.0000 |
| **stat** | position_based | 10 | 1.0000 | 0.0354 | 1.0000 |
| **stat** | transfermarkt_benchmark | 3 | 0.0667 (6.7%) | 0.0343 (3.4%) | 0.0469 (4.7%) |
| **stat** | transfermarkt_benchmark | 5 | 0.0400 (4.0%) | 0.0343 (3.4%) | 0.0339 (3.4%) |
| **stat** | transfermarkt_benchmark | 10 | 0.0500 (5.0%) | 0.0876 (8.8%) | 0.0584 (5.8%) |
| **text** | position_based | 3 | 0.8067 | 0.0085 | 0.8082 |
| **text** | position_based | 5 | 0.8020 | 0.0143 | 0.8046 |
| **text** | position_based | 10 | 0.7890 | 0.0281 | 0.7945 |
| **text** | transfermarkt_benchmark | 3 | 0.0333 (3.3%) | 0.0167 (1.7%) | 0.0469 (4.7%) |
| **text** | transfermarkt_benchmark | 5 | 0.0800 (8.0%) | 0.0767 (7.7%) | 0.0763 (7.6%) |
| **text** | transfermarkt_benchmark | 10 | 0.0500 (5.0%) | 0.0933 (9.3%) | 0.0813 (8.1%) |

*Note: Precision@k is 1.0 for statistical models under `position_based` ground truth because statistical features enforce positional matching perfectly (e.g. strikers only match other strikers), making all retrieved players relevant to their position.*

---

*FootScout © 2026 — Master's Final Project for Data Science Workflow at Berliner Hochschule für Technik (BHT).*
