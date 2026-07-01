"""
app/streamlit_app.py
=====================
FootScout — Multi-Page Interactive Streamlit Dashboard
------------------------------------------------------
Three pages:
  Page 1 — Player Finder:
    Radar chart + top-k recommendations for any player.
    Uses the reusable make_radar_figure() from src/recommender.py.

  Page 2 — Budget-Aware Scout:
    Find the best affordable replacement within a budget.

  Page 3 — Hidden Gem Explorer:
    Discover undervalued players matching a target profile.

Design: Dark glassmorphism theme, Plotly charts, custom CSS.
"""

import sys
import os
import warnings
import logging
from pathlib import Path

# ── Suppress harmless torchvision/transformers file-watcher warnings ─────────
# Streamlit's local_sources_watcher inspects all loaded modules; optional
# heavy-ML extras like torchvision trigger benign ModuleNotFoundErrors.
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*No module named.*")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Path Setup ─────────────────────────────────────────────────────────────────
# Ensure imports from src/ work when running from app/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FootScout ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/footscout",
        "About": "FootScout — AI-Powered Football Player Recommender | BHT Berlin Master's Project",
    },
)

# ─── Custom CSS (Premium Dark Theme — v2) ─────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Space Grotesk', sans-serif;
        color: #E8EBF0;
    }

    /* ── Animated App Background ── */
    .stApp {
        background:
            radial-gradient(ellipse at 20% 10%, rgba(108,99,255,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(0,210,168,0.04) 0%, transparent 50%),
            linear-gradient(160deg, #0A0C14 0%, #0F1117 35%, #0D1520 70%, #090D14 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,12,20,0.98) 0%, rgba(13,17,26,0.98) 100%);
        border-right: 1px solid rgba(108,99,255,0.18);
        backdrop-filter: blur(24px);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #7B74FF;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg,
            rgba(108,99,255,0.12) 0%,
            rgba(70,60,220,0.08) 50%,
            rgba(0,210,168,0.06) 100%);
        border: 1px solid rgba(108,99,255,0.22);
        border-radius: 24px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        box-shadow:
            0 8px 40px rgba(108,99,255,0.12),
            inset 0 1px 0 rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(108,99,255,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8880FF 0%, #6C63FF 40%, #00D2A8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: rgba(200, 205, 220, 0.65);
        font-size: 1.05rem;
        margin-top: 0.6rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(108,99,255,0.18);
        border-color: rgba(108,99,255,0.25);
    }

    /* ── Premium Player Card ── */
    .player-card {
        background: linear-gradient(135deg,
            rgba(108,99,255,0.09) 0%,
            rgba(70,60,220,0.06) 50%,
            rgba(0,210,168,0.05) 100%);
        border: 1px solid rgba(108,99,255,0.28);
        border-radius: 22px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    .player-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108,99,255,0.5), transparent);
    }
    .player-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8880FF 0%, #6C63FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .player-meta {
        color: rgba(200,205,220,0.6);
        font-size: 0.88rem;
        margin-top: 0.35rem;
        letter-spacing: 0.01em;
    }

    /* ── Metric Badge ── */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(108,99,255,0.12);
        border: 1px solid rgba(108,99,255,0.28);
        border-radius: 8px;
        padding: 0.3rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #A89CFF;
        margin: 0.15rem;
        letter-spacing: 0.02em;
        transition: background 0.2s, transform 0.2s;
    }
    .metric-badge:hover {
        background: rgba(108,99,255,0.2);
        transform: translateY(-1px);
    }
    .metric-badge.green {
        background: rgba(0,210,168,0.1);
        border-color: rgba(0,210,168,0.28);
        color: #00D2A8;
    }
    .metric-badge.gold {
        background: rgba(255,193,7,0.1);
        border-color: rgba(255,193,7,0.28);
        color: #FFC107;
    }
    .metric-badge.rose {
        background: rgba(248,87,166,0.1);
        border-color: rgba(248,87,166,0.28);
        color: #F857A6;
    }

    /* ── Result Table ── */
    .rec-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 0.4rem;
    }
    .rec-table th {
        color: rgba(200,205,220,0.45);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .rec-table td {
        background: rgba(255,255,255,0.025);
        padding: 0.65rem 0.8rem;
        font-size: 0.9rem;
    }
    .rec-table tr:hover td {
        background: rgba(108,99,255,0.07);
    }

    /* ── Similarity Bar ── */
    .sim-bar-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sim-bar {
        height: 5px;
        border-radius: 3px;
        background: linear-gradient(90deg, #6C63FF, #00D2A8);
    }
    .sim-value {
        font-size: 0.78rem;
        color: rgba(200,205,220,0.65);
        white-space: nowrap;
        font-weight: 600;
    }

    /* ── Section Title ── */
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(108,99,255,0.3), transparent);
        margin-left: 0.5rem;
    }

    /* ── Player Avatar Ring ── */
    .avatar-ring {
        border-radius: 14px;
        border: 2px solid rgba(108,99,255,0.45);
        box-shadow:
            0 0 0 1px rgba(108,99,255,0.15),
            0 8px 20px rgba(0,0,0,0.45),
            0 0 20px rgba(108,99,255,0.12);
        object-fit: cover;
        background: rgba(30,32,50,0.8);
        flex-shrink: 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .avatar-ring:hover {
        transform: scale(1.04);
        box-shadow:
            0 0 0 2px rgba(108,99,255,0.5),
            0 12px 28px rgba(0,0,0,0.5),
            0 0 30px rgba(108,99,255,0.2);
    }

    /* ── Rank Badge ── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 28px;
        height: 28px;
        border-radius: 8px;
        background: rgba(108,99,255,0.18);
        border: 1px solid rgba(108,99,255,0.35);
        font-size: 0.78rem;
        font-weight: 800;
        color: #9B94FF;
        padding: 0 6px;
        letter-spacing: 0.02em;
        flex-shrink: 0;
    }
    .rank-badge.gold {
        background: rgba(255,193,7,0.15);
        border-color: rgba(255,193,7,0.4);
        color: #FFC107;
    }
    .rank-badge.silver {
        background: rgba(192,192,192,0.12);
        border-color: rgba(192,192,192,0.3);
        color: #C0C0C0;
    }
    .rank-badge.bronze {
        background: rgba(205,127,50,0.12);
        border-color: rgba(205,127,50,0.3);
        color: #CD7F32;
    }

    /* ── Streamlit overrides ── */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSlider"] > div,
    div[data-testid="stNumberInput"] > div {
        background: rgba(255,255,255,0.035);
        border-radius: 12px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #5048E5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.65rem 1.6rem;
        transition: all 0.25s;
        width: 100%;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 16px rgba(108,99,255,0.25);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7B73FF 0%, #6058F0 100%);
        box-shadow: 0 6px 28px rgba(108,99,255,0.5);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(108,99,255,0.3);
    }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.9rem;
        transition: border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(108,99,255,0.2);
    }
    div[data-testid="metric-container"] label {
        color: rgba(200,205,220,0.55) !important;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #A89CFF !important;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
    }

    /* ── Nav radio styling ── */
    div[data-testid="stRadio"] > div {
        gap: 0.3rem;
    }
    div[data-testid="stRadio"] label {
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        transition: background 0.2s;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(108,99,255,0.08);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(108,99,255,0.5); }

    /* ── Toast / Info / Warning ── */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border-left-width: 3px;
    }

    /* ── Input fields ── */
    div[data-testid="stTextInput"] input {
        border-radius: 12px;
        border-color: rgba(108,99,255,0.3);
        background: rgba(255,255,255,0.04);
        color: #E8EBF0;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(108,99,255,0.6);
        box-shadow: 0 0 0 2px rgba(108,99,255,0.15);
    }

    /* ── Multiselect tags ── */
    span[data-baseweb="tag"] {
        background: rgba(108,99,255,0.2) !important;
        border: 1px solid rgba(108,99,255,0.4) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data & Model Loading ────────────────────────────────────────────────────────

def get_csv_mtime() -> float:
    """Get the modification time of the processed players dataset."""
    path = PROJECT_ROOT / "data" / "processed" / "players_merged.csv"
    if path.exists():
        return path.stat().st_mtime
    return 0.0


@st.cache_data(show_spinner=False)
def load_player_data(mtime: float) -> pd.DataFrame:
    """Load the merged player dataset. Depends on mtime to automatically invalidate cache."""
    path = PROJECT_ROOT / "data" / "processed" / "players_merged.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        return df
    # Demo data for development (no real data yet)
    return _generate_demo_data()


@st.cache_resource(show_spinner=False)
def load_recommender_cached(method: str = "umap", alpha: float = 0.6, mtime: float = 0.0):
    """Load (or build) the recommender engine. Cached as a singleton, invalidated by mtime."""
    try:
        from src.recommender import load_recommender
        df = load_player_data(mtime)
        return load_recommender(method=method, alpha=alpha, df=df), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def _generate_demo_data() -> pd.DataFrame:
    """
    Generate realistic demo player data for UI development.
    Used when real scraped data is not yet available.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    positions = ["GK", "CB", "LB", "RB", "CM", "DM", "AM", "LW", "RW", "CF", "ST"]
    leagues   = ["Premier League", "Bundesliga", "La Liga", "Serie A", "Ligue 1"]
    clubs = {
        "Premier League": ["Manchester City", "Arsenal", "Liverpool", "Chelsea", "Tottenham"],
        "Bundesliga":     ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen"],
        "La Liga":        ["Real Madrid", "Barcelona", "Atlético Madrid", "Sevilla"],
        "Serie A":        ["Juventus", "Inter Milan", "AC Milan", "Napoli"],
        "Ligue 1":        ["Paris Saint-Germain", "Marseille", "Lyon", "Monaco"],
    }

    players = []
    for i in range(300):
        league  = random.choice(leagues)
        club    = random.choice(clubs[league])
        pos     = random.choice(positions)
        is_fw   = pos in ["LW", "RW", "CF", "ST", "AM"]
        is_def  = pos in ["CB", "LB", "RB", "DM"]

        players.append({
            "player":               f"Player {i+1:03d}",
            "squad":                club,
            "league":               league,
            "pos":                  pos,
            "age":                  random.randint(18, 35),
            "playing_time_min":     random.randint(450, 3200),
            "gls_per90":            round(np.random.beta(2, 8) * (0.8 if is_fw else 0.15), 3),
            "ast_per90":            round(np.random.beta(2, 8) * (0.5 if is_fw else 0.2), 3),
            "xg_per90":             round(np.random.beta(2, 8) * (0.7 if is_fw else 0.1), 3),
            "npxg_per90":           round(np.random.beta(2, 8) * 0.5, 3),
            "xag_per90":            round(np.random.beta(2, 8) * 0.3, 3),
            "prog_carries_per90":   round(np.random.beta(3, 5) * 6.0, 3),
            "prog_passes_per90":    round(np.random.beta(3, 5) * 8.0, 3),
            "tackles_tkl_per90":    round(np.random.beta(3, 5) * (3.0 if is_def else 1.5), 3),
            "int_per90":            round(np.random.beta(3, 5) * (2.5 if is_def else 1.0), 3),
            "clr_per90":            round(np.random.beta(3, 5) * (3.5 if pos == "CB" else 0.8), 3),
            "pass_completion_pct":  round(random.gauss(82, 8), 1),
            "sh_per90":             round(np.random.beta(2, 6) * (3.5 if is_fw else 1.0), 3),
            "sot_per90":            round(np.random.beta(2, 7) * (1.5 if is_fw else 0.4), 3),
            "touches_per90":        round(random.gauss(55, 15), 1),
            "market_value_eur":     random.choice([
                None, 500_000, 1_000_000, 3_000_000, 5_000_000,
                10_000_000, 20_000_000, 40_000_000, 80_000_000, 150_000_000,
            ]),
        })

    return pd.DataFrame(players)


# ─── Helper Functions ────────────────────────────────────────────────────────────

# Full ISO 3166 alpha-3 / FBref code → FlagCDN alpha-2 mapping
FLAG_MAP = {
    # European nations
    "ENG": "gb-eng", "SCO": "gb-sct", "WAL": "gb-wls", "NIR": "gb-nir",
    "ESP": "es", "GER": "de", "FRA": "fr", "ITA": "it", "IT": "it",
    "POR": "pt", "NED": "nl", "BEL": "be", "SUI": "ch", "CHE": "ch",
    "AUT": "at", "POL": "pl", "CZE": "cz", "SVK": "sk", "SVN": "si",
    "CRO": "hr", "HRV": "hr", "SRB": "rs", "DEN": "dk", "DNK": "dk",
    "NOR": "no", "SWE": "se", "FIN": "fi", "HUN": "hu", "ROU": "ro",
    "RUS": "ru", "UKR": "ua", "GRE": "gr", "GRC": "gr", "TUR": "tr",
    "ARM": "am", "GEO": "ge", "KOS": "xk", "ISR": "il", "AZE": "az",
    "ALB": "al", "BIH": "ba", "MKD": "mk", "MNE": "me",
    # Americas
    "BRA": "br", "ARG": "ar", "COL": "co", "URU": "uy", "CHI": "cl",
    "CHL": "cl", "PER": "pe", "ECU": "ec", "PAR": "py", "BOL": "bo",
    "VEN": "ve", "MEX": "mx", "USA": "us", "CAN": "ca", "CRC": "cr",
    "HON": "hn", "PAN": "pa", "JAM": "jm", "TRI": "tt", "HAI": "ht",
    # African nations
    "MAR": "ma", "NGA": "ng", "SEN": "sn", "GHA": "gh", "CIV": "ci",
    "CMR": "cm", "EGY": "eg", "ALG": "dz", "TUN": "tn", "MLI": "ml",
    "GUI": "gn", "GAB": "ga", "COD": "cd", "BFA": "bf", "CPV": "cv",
    "ZIM": "zw", "ANG": "ao", "MOZ": "mz", "TAN": "tz", "UGA": "ug",
    "KEN": "ke", "ETH": "et", "RSA": "za", "ZAF": "za",
    # Asian nations
    "JPN": "jp", "KOR": "kr", "CHN": "cn", "IRN": "ir", "SAU": "sa",
    "UAE": "ae", "QAT": "qa", "AUS": "au", "NZL": "nz", "IND": "in",
    "IRQ": "iq", "JOR": "jo", "LBN": "lb", "SYR": "sy",
    # Others
    "IRL": "ie", "KAZ": "kz", "CUW": "cw", "UZB": "uz",
}




# Full country name labels for UI display
NATION_LABELS = {
    "ENG": "England 🏴󠁧󠁢󠁥󠁮󠁧󠁿", "SCO": "Scotland 🏴󠁧󠁢󠁳󠁣󠁴󠁿", "WAL": "Wales 🏴󠁧󠁢󠁷󠁬󠁳󠁿", "IRL": "Ireland 🇮🇪",
    "ESP": "Spain 🇪🇸", "GER": "Germany 🇩🇪", "FRA": "France 🇫🇷", "ITA": "Italy 🇮🇹",
    "POR": "Portugal 🇵🇹", "NED": "Netherlands 🇳🇱", "BEL": "Belgium 🇧🇪",
    "SUI": "Switzerland 🇨🇭", "AUT": "Austria 🇦🇹", "POL": "Poland 🇵🇱",
    "CZE": "Czech Republic 🇨🇿", "SVK": "Slovakia 🇸🇰", "SVN": "Slovenia 🇸🇮",
    "CRO": "Croatia 🇭🇷", "SRB": "Serbia 🇷🇸", "DEN": "Denmark 🇩🇰",
    "NOR": "Norway 🇳🇴", "SWE": "Sweden 🇸🇪", "FIN": "Finland 🇫🇮",
    "HUN": "Hungary 🇭🇺", "UKR": "Ukraine 🇺🇦", "TUR": "Turkey 🇹🇷",
    "ARM": "Armenia 🇦🇲", "GEO": "Georgia 🇬🇪", "KOS": "Kosovo 🇽🇰",
    "ISR": "Israel 🇮🇱", "ALB": "Albania 🇦🇱",
    "BIH": "Bosnia & Herzegovina 🇧🇦", "UZB": "Uzbekistan 🇺🇿",
    "QAT": "Qatar 🇶🇦", "CPV": "Cape Verde 🇨🇻",
    "RSA": "South Africa 🇿🇦", "TUN": "Tunisia 🇹🇳",
    "HAI": "Haiti 🇭🇹", "CUW": "Curaçao 🇨🇼",
    "PAR": "Paraguay 🇵🇾", "SAU": "Saudi Arabia 🇸🇦",
    "JOR": "Jordan 🇯🇴", "IRQ": "Iraq 🇮🇶",
    "NZL": "New Zealand 🇳🇿", "CHN": "China 🇨🇳",
    "BRA": "Brazil 🇧🇷", "ARG": "Argentina 🇦🇷", "COL": "Colombia 🇨🇴",
    "URU": "Uruguay 🇺🇾", "ECU": "Ecuador 🇪🇨", "PAR": "Paraguay 🇵🇾",
    "CHI": "Chile 🇨🇱", "BOL": "Bolivia 🇧🇴", "VEN": "Venezuela 🇻🇪",
    "MEX": "Mexico 🇲🇽", "USA": "United States 🇺🇸", "CAN": "Canada 🇨🇦",
    "CRC": "Costa Rica 🇨🇷", "HON": "Honduras 🇭🇳", "PAN": "Panama 🇵🇦",
    "JAM": "Jamaica 🇯🇲", "TRI": "Trinidad & Tobago 🇹🇹",
    "MAR": "Morocco 🇲🇦", "NGA": "Nigeria 🇳🇬", "SEN": "Senegal 🇸🇳",
    "GHA": "Ghana 🇬🇭", "CIV": "Ivory Coast 🇨🇮", "CMR": "Cameroon 🇨🇲",
    "EGY": "Egypt 🇪🇬", "ALG": "Algeria 🇩🇿", "MLI": "Mali 🇲🇱",
    "GUI": "Guinea 🇬🇳", "GAB": "Gabon 🇬🇦", "COD": "DR Congo 🇨🇩",
    "BFA": "Burkina Faso 🇧🇫", "TUN": "Tunisia 🇹🇳", "RSA": "South Africa 🇿🇦",
    "JPN": "Japan 🇯🇵", "KOR": "South Korea 🇰🇷", "IRN": "Iran 🇮🇷",
    "SAU": "Saudi Arabia 🇸🇦", "AUS": "Australia 🇦🇺", "NZL": "New Zealand 🇳🇿",
    "CHN": "China 🇨🇳", "IRQ": "Iraq 🇮🇶", "JOR": "Jordan 🇯🇴",
}
# Reverse map: full label or name → ISO code
NATION_CODE_FROM_LABEL = {v: k for k, v in NATION_LABELS.items()}
NATION_CODE_FROM_LABEL.update({k: k for k in NATION_LABELS})  # also accept raw codes


def apply_country_filter(results: pd.DataFrame, selected_nations: list[str]) -> pd.DataFrame:
    """Filter recommendation results to selected nations only."""
    if not selected_nations:
        return results
    # Normalize selection to ISO codes
    codes = set()
    for sel in selected_nations:
        code = NATION_CODE_FROM_LABEL.get(sel, sel)
        codes.add(code.upper())
    
    nation_col = next((c for c in ["nation", "nationality_tm"] if c in results.columns), None)
    if not nation_col:
        return results
    
    return results[results[nation_col].str.upper().isin(codes)]


def get_flag_html(nation) -> str:
    """Generate HTML image tag for country flag using FlagCDN. Handles NaN/None gracefully."""
    if nation is None:
        return ""
    # Handle pandas NA / numpy float NaN
    try:
        if not isinstance(nation, str):
            nation = str(nation)
    except Exception:
        return ""
    nation = nation.strip()
    if not nation or nation in ("N/A", "nan", "None", "-"):
        return ""
    flag_code = FLAG_MAP.get(nation.upper(), "un")
    return (
        f"<img src='https://flagcdn.com/w20/{flag_code}.png' "
        f"style='vertical-align:middle;margin-right:4px;border-radius:2px;"
        f"border:1px solid rgba(255,255,255,0.15);height:14px;' "
        f"title='{nation}'/>"
    )


def get_nation_str(row: pd.Series) -> str:
    """Extract the nation string from a row, trying multiple column names."""
    for col in ("nation", "nationality_tm", "nationality"):
        val = row.get(col)
        if val and isinstance(val, str) and val.strip() and val.strip() not in ("N/A", "nan"):
            return val.strip()
    return ""


def get_flag_and_nation_html(row: pd.Series) -> str:
    """Generate HTML snippet for country flag and clean full country name."""
    import html as _html
    import re
    nation_code = get_nation_str(row)
    flag_html = get_flag_html(nation_code)
    nation_full = NATION_LABELS.get(nation_code.upper(), nation_code)
    # Strip flag emoji from label
    nation_clean = re.sub(r'[\u2600-\u27BF]|[\u2000-\u3300]|[\uD83C-\uD83E][\uDC00-\uDFFF]', '', nation_full).strip()
    nation_display = _html.escape(nation_clean) if nation_clean else ""
    return f"{flag_html}<span>{nation_display}</span>"


# Known bad image keywords — team photos, venue shots, wrong people
_BAD_IMAGE_KEYWORDS = [
    "Manchester_City", "Arsenal_", "Chelsea_", "Liverpool_", "Paris_Saint",
    "Bayern_", "Real_Madrid", "Barcelona_", "Dortmund_", "Juventus_",
    "Atletico_", "Inter_", "Milan_", "Leipzig_", "Crystal_Palace",
    "MLS_All_Star", "Wembley", "Yokohama", "_vs_", "Cliff_Blue",
    "Kyle_Daukaus", "Castleton", "FA_Cup_winner", "warm_up_versus",
    "25-26_vs", "2023-07-19", "1_arsenal_crystal",
]


def _is_valid_player_image(url: str) -> bool:
    """Return True if the URL looks like an individual player photo, not a team/venue shot."""
    if not isinstance(url, str) or not url.strip().startswith("http"):
        return False
    url_lower = url.lower()
    for kw in _BAD_IMAGE_KEYWORDS:
        if kw.lower() in url_lower:
            return False
    return True


def get_player_image_url(row: pd.Series, player_name: str) -> str:
    """
    Resolve the best image URL for a player.
    Priority:
    1. TheSportsDB CDN URL (browser-accessible, no hotlink blocking)
    2. Local file served via Streamlit static serving (app/static/players/)
       — only if it maps to an individual player photo
    3. Styled SVG initials avatar fallback (always works, looks premium)
    """
    # 1. TheSportsDB URL - freely browser-accessible CDN (works on Cloud too)
    sdb_url = row.get("sportsdb_image_url")
    if sdb_url and isinstance(sdb_url, str) and sdb_url.strip().startswith("http"):
        return sdb_url.strip()

    # 2. Check local downloaded image (Streamlit static file serving)
    #    Only works when running locally or if images are committed to repo
    local_path = row.get("local_image_path")
    if local_path and isinstance(local_path, str):
        local_path = local_path.strip()
        if local_path:
            # Verify the source image URL isn't a team/venue photo
            source_url = row.get("image_url", "")
            if _is_valid_player_image(source_url):
                # Streamlit serves app/static/ at /app/static/ when enableStaticServing=True
                if local_path.startswith("app/static/"):
                    full_path = PROJECT_ROOT / local_path
                    if full_path.exists():
                        return "/" + local_path
                # Also handle bare 'static/...' paths
                full_path = PROJECT_ROOT / local_path
                if full_path.exists():
                    return "/" + local_path

    # 3. Premium SVG initials avatar fallback (always works, no external deps)
    return get_player_avatar_svg(player_name)


def get_player_avatar_url(name: str) -> str:
    """Generate a premium avatar URL for the player using Dicebear adventurer style."""
    import urllib.parse
    encoded_name = urllib.parse.quote(name)
    return f"https://api.dicebear.com/7.x/adventurer/svg?seed={encoded_name}&backgroundColor=b6e3f4,c0aade,d1d4f9,ffdfbf"


def format_market_value(val) -> str:
    """Format market value in human-readable EUR string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    val = float(val)
    if val >= 1_000_000:
        return f"€{val / 1_000_000:.1f}M"
    elif val >= 1_000:
        return f"€{val / 1_000:.0f}K"
    return f"€{val:,.0f}"


def similarity_bar_html(score: float) -> str:
    """Render an HTML similarity progress bar."""
    pct = int(score * 100)
    color = "#6C63FF" if score > 0.8 else ("#00D2A8" if score > 0.6 else "#FFC107")
    return f"""
    <div class='sim-bar-container'>
        <div class='sim-bar' style='width: {pct}px; background: {color};'></div>
        <span class='sim-value'>{score:.3f}</span>
    </div>"""
def get_player_avatar_svg(name: str, size: int = 88) -> str:
    """Generate a premium base64-encoded SVG square rounded-corner avatar with initials."""
    import hashlib
    import base64

    h = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16)

    gradients = [
        ("#6C63FF", "#3F37C9", "#2D2880"),  # Deep Indigo
        ("#00C49A", "#0077B6", "#023E8A"),  # Teal-Navy
        ("#F4A261", "#E63946", "#9B1B30"),  # Amber-Crimson
        ("#F72585", "#7209B7", "#480CA8"),  # Magenta-Violet
        ("#4CC9F0", "#4361EE", "#3A0CA3"),  # Sky-Royal Blue
        ("#FF9F1C", "#E76F51", "#C1440E"),  # Orange-Rust
        ("#55A630", "#2DC653", "#007F5F"),  # Green
        ("#7B2FBE", "#9D4EDD", "#5A189A"),  # Purple
    ]
    c1, c2, c3 = gradients[h % len(gradients)]

    parts = [p for p in name.split() if p]
    if len(parts) >= 2:
        initials = (parts[0][0] + parts[-1][0]).upper()
    elif len(parts) == 1:
        initials = parts[0][:2].upper()
    else:
        initials = "FS"

    r = 14  # corner radius for square
    fs = size // 3  # font size
    gid = f"g{h % 99999}"

    svg = f"""<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' fill='none' xmlns='http://www.w3.org/2000/svg'>
  <defs>
    <linearGradient id='{gid}a' x1='0' y1='0' x2='{size}' y2='{size}' gradientUnits='userSpaceOnUse'>
      <stop offset='0%' stop-color='{c1}'/>
      <stop offset='60%' stop-color='{c2}'/>
      <stop offset='100%' stop-color='{c3}'/>
    </linearGradient>
    <linearGradient id='{gid}b' x1='0' y1='0' x2='{size}' y2='0' gradientUnits='userSpaceOnUse'>
      <stop offset='0%' stop-color='rgba(255,255,255,0.12)'/>
      <stop offset='100%' stop-color='rgba(255,255,255,0)'/>
    </linearGradient>
  </defs>
  <!-- Background -->
  <rect width='{size}' height='{size}' rx='{r}' fill='url(#{gid}a)'/>
  <!-- Subtle top sheen -->
  <rect width='{size}' height='{size//2}' rx='{r}' fill='url(#{gid}b)' opacity='0.5'/>
  <!-- Initials -->
  <text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle'
        fill='rgba(255,255,255,0.95)' font-size='{fs}'
        font-family='Outfit, Inter, Arial, sans-serif'
        font-weight='800' letter-spacing='1'>{initials}</text>
</svg>"""
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"


def render_player_card(row: pd.Series, rank: int = 0, show_gem_badge: bool = False) -> None:
    """Render a premium player recommendation card with fully inlined styles."""
    import html as _html
    mv      = format_market_value(row.get("market_value_eur"))
    sim     = float(row.get("similarity", 0) or 0)
    sim_pct = f"{sim*100:.1f}%"
    age     = row.get("age", row.get("age_tm", ""))
    age_str = f" · Age {int(float(age))}" if age and str(age) not in ("nan", "N/A", "") else ""

    player_name = _html.escape(str(row.get("player", "N/A")))
    position    = _html.escape(str(row.get("position", row.get("pos", "N/A"))))
    squad       = _html.escape(str(row.get("squad", "N/A")))
    league      = _html.escape(str(row.get("league", "N/A")))

    def _stat(key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    # Rank badge style
    rank_cls = "gold" if rank == 1 else ("silver" if rank == 2 else ("bronze" if rank == 3 else ""))
    rank_html = f"<span class='rank-badge {rank_cls}'>#{rank}</span> " if rank > 0 else ""

    # Similarity color
    if sim >= 0.82:
        sim_color = "#00D2A8"
        sim_bg    = "rgba(0,210,168,0.12)"
        sim_bord  = "rgba(0,210,168,0.35)"
    elif sim >= 0.65:
        sim_color = "#8880FF"
        sim_bg    = "rgba(108,99,255,0.12)"
        sim_bord  = "rgba(108,99,255,0.35)"
    else:
        sim_color = "#FFC107"
        sim_bg    = "rgba(255,193,7,0.1)"
        sim_bord  = "rgba(255,193,7,0.3)"

    # Similarity bar fill
    bar_width = max(4, int(sim * 80))

    # Badge styles
    b_purple = "display:inline-block;background:rgba(108,99,255,0.12);border:1px solid rgba(108,99,255,0.28);border-radius:8px;padding:3px 10px;font-size:11.5px;font-weight:600;color:#A89CFF;margin:2px;"
    b_teal   = "display:inline-block;background:rgba(0,210,168,0.1);border:1px solid rgba(0,210,168,0.28);border-radius:8px;padding:3px 10px;font-size:11.5px;font-weight:600;color:#00D2A8;margin:2px;"
    b_gold   = "display:inline-block;background:rgba(255,193,7,0.1);border:1px solid rgba(255,193,7,0.28);border-radius:8px;padding:3px 10px;font-size:11.5px;font-weight:600;color:#FFC107;margin:2px;"
    gem_html = f"<span style='{b_gold}'>💎 Hidden Gem</span>" if show_gem_badge else ""

    # World Cup badge
    wc_html = ""
    try:
        if int(float(row.get("is_world_cup", 0))) == 1:
            wc_html = f"<span style='{b_teal}'>🏆 WC 2026</span>"
    except Exception:
        pass

    # Market value color
    mv_color = "#00D2A8"

    # Image
    img_src = get_player_image_url(row, player_name)
    flag_nation_html = get_flag_and_nation_html(row)

    # Stat bars (small progress indicators)
    gls  = _stat("gls_per90")
    ast  = _stat("ast_per90")
    xg   = _stat("xg_per90")
    tkl  = _stat("tackles_tkl_per90")
    prog = _stat("prog_carries_per90")

    # Detect if it's an SVG data URL (initials avatar)
    is_svg_avatar = img_src.startswith("data:image/svg")
    img_style = (
        "width:88px;height:88px;border-radius:14px;object-fit:cover;flex-shrink:0;"
        "border:2px solid rgba(108,99,255,0.4);box-shadow:0 6px 20px rgba(0,0,0,0.4),0 0 20px rgba(108,99,255,0.1);"
        "background:rgba(20,22,40,0.8);"
    )

    # Build card HTML via safe string concatenation (avoids markdown parser issues)
    fallback_svg = get_player_avatar_svg(player_name)

    parts = [
        "<div style='background:linear-gradient(135deg,rgba(20,22,40,0.92) 0%,rgba(16,18,35,0.92) 100%);",
        "border:1px solid rgba(108,99,255,0.2);border-radius:20px;padding:18px 22px;",
        "margin-bottom:12px;box-shadow:0 4px 24px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.04);",
        "position:relative;overflow:hidden;'>",
        "<div style='position:absolute;top:0;left:0;right:0;height:1px;",
        "background:linear-gradient(90deg,transparent,rgba(108,99,255,0.35),transparent);'></div>",
        "<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:14px;align-items:flex-start;'>",
        # left
        "<div style='display:flex;align-items:center;gap:18px;'>",
        f"<img src='{img_src}' style='{img_style}' onerror=\"this.onerror=null;this.src='{fallback_svg}';\"/>",
        "<div>",
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>{rank_html}",
        f"<span style='font-family:Outfit,sans-serif;font-size:17px;font-weight:800;color:#E8EBF0;'>{player_name}</span></div>",
        f"<div style='font-size:12.5px;color:rgba(200,205,220,0.55);display:flex;align-items:center;flex-wrap:wrap;gap:4px;'>",
        f"{flag_nation_html}",
        f"<span style='color:rgba(255,255,255,0.2);'>&middot;</span>",
        f"<span style='background:rgba(108,99,255,0.14);border-radius:5px;padding:1px 7px;color:#A89CFF;font-weight:600;'>{position}</span>",
        f"<span style='color:rgba(255,255,255,0.2);'>&middot;</span><span>{squad}</span>",
        f"<span style='color:rgba(255,255,255,0.2);'>&middot;</span>",
        f"<span style='color:rgba(200,205,220,0.4);font-size:11.5px;'>{league}</span>",
        f"<span style='color:rgba(200,205,220,0.35);'>{age_str}</span></div>",
        f"<div style='margin-top:9px;display:flex;flex-wrap:wrap;gap:4px;'>",
        f"<span style='{b_purple}'>\u26bd {gls:.2f} G/90</span>",
        f"<span style='{b_purple}'>\U0001f170 {ast:.2f} A/90</span>",
        f"<span style='{b_purple}'>\U0001f4c8 {xg:.2f} xG/90</span>",
        f"<span style='{b_purple}'>\U0001f6e1 {tkl:.2f} Tkl/90</span>",
        f"<span style='{b_purple}'>\U0001f3c3 {prog:.1f} Prog/90</span>",
        "</div></div></div>",
        # right
        "<div style='display:flex;flex-direction:column;align-items:flex-end;gap:8px;'>",
        f"<div style='background:{sim_bg};border:1px solid {sim_bord};border-radius:12px;padding:6px 14px;text-align:center;'>",
        f"<div style='font-family:Outfit,sans-serif;font-size:20px;font-weight:800;color:{sim_color};line-height:1;'>{sim_pct}</div>",
        "<div style='font-size:10px;color:rgba(200,205,220,0.45);margin-top:2px;text-transform:uppercase;letter-spacing:0.07em;'>match</div>",
        "</div>",
        f"<div style='font-family:Outfit,sans-serif;font-size:15px;font-weight:700;color:#00D2A8;'>{mv}</div>",
        f"<div style='display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end;'>{wc_html}{gem_html}</div>",
        f"<div style='width:80px;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;margin-top:4px;'>",
        f"<div style='width:{bar_width}px;height:4px;background:linear-gradient(90deg,#6C63FF,{sim_color});border-radius:2px;'></div></div>",
        "</div>",
        "</div></div>",
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


# ─── Sidebar Navigation ─────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> tuple[str, str, float]:
    """Render sidebar navigation and return selected page + config."""
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0;'>
            <div style='font-size:2.5rem;'>⚽</div>
            <div style='font-family:Outfit; font-size:1.5rem; font-weight:700;
                        background:linear-gradient(135deg,#6C63FF,#00D2A8);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                FootScout
            </div>
            <div style='color:rgba(232,235,240,0.5); font-size:0.8rem; margin-top:0.2rem;'>
                AI Football Scout
            </div>
        </div>
        <hr style='border-color:rgba(108,99,255,0.2); margin:0.5rem 0;'>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🤖 AI Scout", "🔍 Player Finder", "💰 Budget Scout", "💎 Hidden Gem Explorer"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(108,99,255,0.2);'>", unsafe_allow_html=True)
        st.markdown("**⚙️ Model Settings**")

        embedding_type = st.selectbox(
            "Embedding Type",
            ["hybrid", "stat", "text"],
            index=0,
            help="hybrid = statistical + text (best); stat = per-90 stats only; text = NLP profile only",
        )

        alpha = st.slider(
            "Alpha (stat weight)",
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            help="Controls stat vs text weighting: 1.0 = pure stats, 0.0 = pure text",
        )

        st.markdown("<hr style='border-color:rgba(108,99,255,0.2);'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='color:rgba(232,235,240,0.45); font-size:0.78rem; text-align:center;'>
            {len(df):,} players loaded<br>
            BHT Berlin · DS Workflow 2026
        </div>
        """, unsafe_allow_html=True)

    return page, embedding_type, alpha


# ─── PAGE 1: Player Finder ───────────────────────────────────────────────────────

def page_player_finder(df: pd.DataFrame, rec, embedding_type: str) -> None:
    """Player Finder page with radar chart and top-k recommendations."""
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>🔍 Player Finder</div>
        <div class='hero-subtitle'>
            Search any player and discover who plays like them — worldwide.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])

    name_col = next((c for c in ["player", "name"] if c in df.columns), None)
    player_names = sorted(df[name_col].dropna().tolist()) if name_col else []

    with col1:
        selected_player = st.selectbox(
            "Select Player",
            options=player_names,
            index=0 if player_names else None,
            help="Type to search",
        )
    with col2:
        k = st.slider("Top-k Results", min_value=3, max_value=15, value=5)
    with col3:
        pos_filter = st.selectbox(
            "Position Filter",
            ["All", "GK", "CB", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "CF", "ST"],
        )
        pos_filter = None if pos_filter == "All" else pos_filter

    # Country filter
    available_nations = sorted([
        NATION_LABELS.get(n, n)
        for n in df["nation"].dropna().unique()
        if isinstance(n, str)
    ])
    country_filter = st.multiselect(
        "🌍 Filter by Country (leave empty = all countries)",
        options=available_nations,
        default=[],
        placeholder="Type a country name...",
        key="finder_country_filter",
    )

    if not selected_player:
        st.info("Select a player to get started.")
        return

    # ── Player info ───────────────────────────────────────────────────────────
    if name_col:
        player_row = df[df[name_col].str.lower() == selected_player.lower()]
        if not player_row.empty:
            r = player_row.iloc[0]
            mv = format_market_value(r.get("market_value_eur"))
            nation = str(r.get("nation", r.get("nationality_tm", "N/A")))
            avatar_url = get_player_image_url(r, selected_player)
            flag_and_nation = get_flag_and_nation_html(r)

            st.markdown(f"""
            <div class='player-card' style='background:linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(0,210,168,0.08) 100%); border:1px solid rgba(108,99,255,0.45); border-radius:20px; padding:1.5rem; margin-bottom:1rem;'>
                <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; align-items:center;'>
                    <div style='display:flex; align-items:center; gap:20px;'>
                        <img src='{avatar_url}' width='80' height='80' style='border-radius:16px; object-fit: cover; background:rgba(255,255,255,0.05); border:2px solid rgba(108,99,255,0.45); box-shadow:0 4px 16px rgba(0,0,0,0.4); flex-shrink:0;'/>
                        <div>
                            <div class='player-name' style='font-size:2rem;'>{r.get('player', selected_player)}</div>
                            <div class='player-meta' style='display:flex; align-items:center; gap:6px; flex-wrap:wrap; color:rgba(232, 235, 240, 0.65); font-size:0.95rem; margin-top:0.3rem;'>
                                📍 {r.get('pos','N/A')} &nbsp;&bull;&nbsp;
                                🏟 {r.get('squad','N/A')} &nbsp;&bull;&nbsp;
                                🌍 {r.get('league','N/A')} &nbsp;&bull;&nbsp;
                                🎂 Age {r.get('age','N/A')} &nbsp;&bull;&nbsp;
                                💶 {mv} &nbsp;&bull;&nbsp;
                                {flag_and_nation}
                            </div>
                        </div>
                    </div>
                    <div>
                        <span class='metric-badge' style='font-size:0.95rem; padding:0.4rem 0.8rem;'>⚽ {r.get('gls_per90', 0):.2f} G/90</span>
                        <span class='metric-badge' style='font-size:0.95rem; padding:0.4rem 0.8rem;'>🅰️ {r.get('ast_per90', 0):.2f} A/90</span>
                        <span class='metric-badge green' style='font-size:0.95rem; padding:0.4rem 0.8rem;'>📈 {r.get('xg_per90', 0):.2f} xG/90</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Radar Chart ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📡 Statistical Profile</div>", unsafe_allow_html=True)

    try:
        from src.recommender import get_radar_data, make_radar_figure
        radar_data = get_radar_data(selected_player, df, position_avg=True)
        fig = make_radar_figure(radar_data, show_average=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Radar chart unavailable: {e}")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown(
        f"<div class='section-title'>🎯 Top {k} Similar Players</div>",
        unsafe_allow_html=True
    )

    if rec is None:
        # Demo mode: show random players
        results = _demo_recommendations(df, selected_player, k)
    else:
        try:
            results = rec.find_similar(selected_player, k=k, position_filter=pos_filter)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")
            results = _demo_recommendations(df, selected_player, k)

    if not results.empty:
        results = apply_country_filter(results, country_filter)
        if results.empty:
            st.info("No players found for the selected country filter. Try removing the filter or selecting a different country.")
        for _, row in results.iterrows():
            render_player_card(row, rank=int(row.get("rank", 0)))

        # Summary bar chart
        if "similarity" in results.columns:
            fig_bar = px.bar(
                results.head(10),
                x="player",
                y="similarity",
                color="similarity",
                color_continuous_scale=["#1a1a3e", "#6C63FF", "#00D2A8"],
                labels={"player": "Player", "similarity": "Similarity Score"},
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(t=20, b=40, l=0, r=0),
                height=280,
            )
            fig_bar.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})
    else:
        st.info("No recommendations found.")


# ─── PAGE 2: Budget Scout ────────────────────────────────────────────────────────

def page_budget_scout(df: pd.DataFrame, rec) -> None:
    """Budget-Aware Scout page."""
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>💰 Budget-Aware Scout</div>
        <div class='hero-subtitle'>
            Find the best stylistic replacement for any player — within your transfer budget.
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    name_col     = next((c for c in ["player", "name"] if c in df.columns), None)
    player_names = sorted(df[name_col].dropna().tolist()) if name_col else []

    with col1:
        selected_player = st.selectbox(
            "Target Player (to replace)",
            player_names,
            key="budget_player",
        )
    with col2:
        budget_m = st.slider(
            "Budget (€M)",
            min_value=1, max_value=200, value=30, step=1,
            help="Maximum market value in millions of EUR",
        )
        budget_eur = budget_m * 1_000_000

    col3, col4 = st.columns(2)
    with col3:
        k = st.slider("Number of Results", 3, 10, 5, key="budget_k")
    with col4:
        same_pos = st.checkbox("Same Position Group", value=True)

    if st.button("🔎 Find Budget Replacements", key="budget_btn"):
        # Query player value
        if name_col:
            player_row = df[df[name_col].str.lower() == selected_player.lower()]
            if not player_row.empty:
                r = player_row.iloc[0]
                mv = format_market_value(r.get("market_value_eur"))
                nation = str(r.get("nation", r.get("nationality_tm", "N/A")))
                avatar_url = get_player_image_url(r, selected_player)
                flag_and_nation = get_flag_and_nation_html(r)
                
                st.markdown(f"""
                <div style='background:rgba(108,99,255,0.06); border:1px solid rgba(108,99,255,0.25); border-radius:16px; padding:16px 20px; margin-bottom:1.5rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;'>
                        <div style='display:flex; align-items:center; gap:20px;'>
                            <img src='{avatar_url}' width='80' height='80' style='border-radius:16px; object-fit: cover; flex-shrink: 0; background: rgba(255,255,255,0.05); border: 2px solid rgba(108,99,255,0.45); box-shadow: 0 4px 16px rgba(0,0,0,0.4);'/>
                            <div>
                                <div style='font-size:20px; font-weight:700; color:#E8EBF0;'>Replacing style: {selected_player}</div>
                                <div style='font-size:13px; color:rgba(232,235,240,0.5); margin-top:4px; display:flex; align-items:center; gap:6px;'>
                                    📍 {r.get('pos','N/A')} &nbsp;&bull;&nbsp; 🏟 {r.get('squad','N/A')} &nbsp;&bull;&nbsp; 🌍 {r.get('league','N/A')} &nbsp;&bull;&nbsp; 💶 {mv} &nbsp;&bull;&nbsp; {flag_and_nation}
                                </div>
                            </div>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:12px; color:rgba(232,235,240,0.45);'>Target Budget Limit</div>
                            <div style='font-size:24px; font-weight:800; color:#00D2A8;'>&le; €{budget_m}M</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        if rec is None:
            results = _demo_recommendations(df, selected_player, k)
            results = results[
                results.get("market_value_eur", pd.Series([0]*len(results))).fillna(0) <= budget_eur
            ].head(k) if "market_value_eur" in results.columns else results
        else:
            try:
                results = rec.find_budget_replacement(
                    selected_player, budget=budget_eur, k=k, same_position=same_pos
                )
            except Exception as e:
                st.error(f"Budget search failed: {e}")
                return

        if not results.empty:
            # Country filter
            avail_n = sorted([NATION_LABELS.get(n,n) for n in df["nation"].dropna().unique() if isinstance(n,str)])
            cf = st.multiselect("🌍 Filter results by Country", options=avail_n, default=[],
                                placeholder="All countries...", key="budget_cf")
            results = apply_country_filter(results, cf)
            if results.empty and cf:
                st.info("No budget replacements found for the selected countries.")
            for _, row in results.iterrows():
                render_player_card(row, rank=int(row.get("rank", 0)))

            # Market value comparison chart
            if "market_value_eur" in results.columns:
                plot_df = results[["player", "market_value_eur", "similarity"]].copy()
                plot_df["market_value_m"] = plot_df["market_value_eur"].fillna(0) / 1e6

                fig = px.scatter(
                    plot_df,
                    x="market_value_m",
                    y="similarity",
                    text="player",
                    color="similarity",
                    size="market_value_m",
                    color_continuous_scale=["#6C63FF", "#00D2A8"],
                    labels={"market_value_m": "Market Value (€M)", "similarity": "Similarity"},
                    template="plotly_dark",
                )
                fig.update_traces(textposition="top center", textfont_size=9)
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,17,26,0.8)",
                    coloraxis_showscale=False,
                    title="Value vs Similarity — Replacements",
                    title_font_color="white",
                    height=360,
                    margin=dict(t=50, b=40),
                )
                # Budget threshold line
                fig.add_vline(
                    x=budget_m, line_dash="dash",
                    line_color="rgba(255,99,99,0.6)",
                    annotation_text=f"Budget €{budget_m}M",
                    annotation_font_color="rgba(255,99,99,0.9)",
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.warning("No players found within budget constraints. Try increasing the budget.")


# ─── PAGE 3: Hidden Gem Explorer ─────────────────────────────────────────────────

def page_hidden_gems(df: pd.DataFrame, rec) -> None:
    """Hidden Gem Explorer page."""
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>💎 Hidden Gem Explorer</div>
        <div class='hero-subtitle'>
            Uncover undervalued talent that matches a target profile.
            Find the next Pedri at a fraction of the cost.
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        position = st.selectbox(
            "Target Position",
            ["FW", "MF", "DF", "GK", "CM", "AM", "DM", "LW", "RW", "CF", "CB", "LB", "RB"],
        )
    with col2:
        max_val_m = st.slider(
            "Max Market Value (€M)",
            min_value=1, max_value=50, value=15,
        )
        max_val_eur = max_val_m * 1_000_000
    with col3:
        k = st.slider("Number of Gems", 3, 10, 5, key="gem_k")

    name_col     = next((c for c in ["player", "name"] if c in df.columns), None)
    player_names = ["None — Use Position Centroid"] + (
        sorted(df[name_col].dropna().tolist()) if name_col else []
    )

    col4, col5 = st.columns(2)
    with col4:
        reference = st.selectbox(
            "Reference Player (optional)",
            player_names,
            help="Define the target playing style via a reference player",
        )
        reference = None if reference == "None — Use Position Centroid" else reference
    with col5:
        min_sim = st.slider(
            "Min Similarity Threshold",
            min_value=0.3, max_value=0.9, value=0.5, step=0.05,
        )

    if st.button("💎 Find Hidden Gems", key="gem_btn"):
        if rec is None:
            results = _demo_recommendations(df, reference or "Demo", k)
            results["is_gem"] = True
        else:
            try:
                results = rec.find_hidden_gems(
                    position=position,
                    max_value=max_val_eur,
                    reference_player=reference,
                    min_similarity=min_sim,
                    k=k,
                )
            except Exception as e:
                st.error(f"Gem search failed: {e}")
                return

        if not results.empty:
            if reference and name_col:
                player_row = df[df[name_col].str.lower() == reference.lower()]
                if not player_row.empty:
                    r = player_row.iloc[0]
                    mv = format_market_value(r.get("market_value_eur"))
                    nation = str(r.get("nation", r.get("nationality_tm", "N/A")))
                    avatar_url = get_player_image_url(r, reference)
                    flag_and_nation = get_flag_and_nation_html(r)
                    
                    st.markdown(f"""
                    <div style='background:rgba(108,99,255,0.06); border:1px solid rgba(108,99,255,0.25); border-radius:16px; padding:16px 20px; margin-bottom:1.5rem;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;'>
                            <div style='display:flex; align-items:center; gap:20px;'>
                                <img src='{avatar_url}' width='80' height='80' style='border-radius:16px; object-fit: cover; flex-shrink: 0; background: rgba(255,255,255,0.05); border: 2px solid rgba(108,99,255,0.45); box-shadow: 0 4px 16px rgba(0,0,0,0.4);'/>
                                <div>
                                    <div style='font-size:20px; font-weight:700; color:#E8EBF0;'>Comparing style against: {reference}</div>
                                    <div style='font-size:13px; color:rgba(232,235,240,0.5); margin-top:4px; display:flex; align-items:center; gap:6px;'>
                                        📍 {r.get('pos','N/A')} &nbsp;&bull;&nbsp; 🏟 {r.get('squad','N/A')} &nbsp;&bull;&nbsp; 🌍 {r.get('league','N/A')} &nbsp;&bull;&nbsp; 💶 {mv} &nbsp;&bull;&nbsp; {flag_and_nation}
                                    </div>
                                </div>
                            </div>
                            <div style='text-align:right;'>
                                <div style='font-size:12px; color:rgba(232,235,240,0.45);'>Target Position &amp; Value</div>
                                <div style='font-size:20px; font-weight:800; color:#00D2A8;'>{position} &bull; &le; €{max_val_m}M</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:16px 20px; margin-bottom:1.5rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;'>
                        <div>
                            <div style='font-size:20px; font-weight:700; color:#E8EBF0;'>Searching for Gems (Position Centroid)</div>
                            <div style='font-size:13px; color:rgba(232,235,240,0.5); margin-top:4px;'>
                                Targeting average style profile of all <b>{position}</b> players in the database.
                            </div>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:12px; color:rgba(232,235,240,0.45);'>Max Value limit</div>
                            <div style='font-size:24px; font-weight:800; color:#00D2A8;'>&le; €{max_val_m}M</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Country filter
            avail_ng = sorted([NATION_LABELS.get(n,n) for n in df["nation"].dropna().unique() if isinstance(n,str)])
            gcf = st.multiselect("🌍 Filter gems by Country", options=avail_ng, default=[],
                                  placeholder="All countries...", key="gems_cf")
            results = apply_country_filter(results, gcf)
            if results.empty and gcf:
                st.info("No hidden gems found for the selected countries.")
            for _, row in results.iterrows():
                render_player_card(row, rank=int(row.get("rank", 0)), show_gem_badge=True)

            # Gem scatter: xG/90 vs Tackles/90 colored by market value
            plot_cols = [
                c for c in ["gls_per90", "xg_per90", "market_value_eur", "player", "similarity"]
                if c in results.columns
            ]
            if len(plot_cols) >= 3:
                fig = px.scatter(
                    results,
                    x="gls_per90",
                    y=results["similarity"],
                    text="player",
                    color="market_value_eur",
                    size=results["similarity"] * 10,
                    color_continuous_scale=px.colors.sequential.Plasma_r,
                    labels={
                        "gls_per90": "Goals per 90",
                        "y": "Similarity Score",
                        "market_value_eur": "Market Value (€)",
                    },
                    template="plotly_dark",
                    title="💎 Hidden Gems — Goals vs Similarity (color = market value)",
                )
                fig.update_traces(textposition="top center", textfont_size=9)
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,17,26,0.8)",
                    title_font_color="white",
                    height=380,
                    margin=dict(t=60, b=40),
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.warning(
                "No gems found matching the current criteria. "
                "Try lowering the similarity threshold or increasing the max value."
            )


@st.cache_resource(show_spinner=False)
def load_text_embedder():
    """Load the SentenceTransformer text embedder model. Cached as a singleton."""
    from src.embeddings import TextEmbedder
    return TextEmbedder()


def page_ai_scout(df: pd.DataFrame, rec, embedding_type: str) -> None:
    """AI Natural Language Search page."""
    import re
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz

    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>🤖 AI Scout</div>
        <div class='hero-subtitle'>
            Search players using natural language (e.g., "I want a player like Messi under $50M").
        </div>
    </div>""", unsafe_allow_html=True)

    # Search box and Quick select columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input(
            "Describe your ideal player...",
            placeholder="e.g., I want a creative midfielder like De Bruyne under $40M",
            value="",
            key="ai_scout_query"
        )
        
    name_col = next((c for c in ["player", "name"] if c in df.columns), None)
    player_names = sorted(df[name_col].dropna().unique().tolist()) if name_col else []

    with col2:
        selected_ref_helper = st.selectbox(
            "👤 Quick Reference Player (Optional)",
            options=["-- None (Parse from text) --"] + player_names,
            index=0,
            help="Select a player to compare against. This overrides any player parsed in the text."
        )

    if not query and selected_ref_helper == "-- None (Parse from text) --":
        st.info("💡 **Try typing queries like:**\n"
                "- *'I want a striker like Haaland under €80M'*\n"
                "- *'defensive midfielder like Rodri under $60M'*\n"
                "- *'fast winger under 30 million'*\n"
                "- *'World Cup midfielder like Pedri under $25M'*\n"
                "- *'playmaker like Mesi' (test spelling auto-correction)*\n")
        return

    # ── Parse query ───────────────────────────────────────────────────────────
    q_lower = query.lower()

    # 1. Parse reference player (checking if direct mention or 'like {name}' matches with fuzzy support)
    ref_player = None
    extracted_name_segment = ""
    
    if selected_ref_helper != "-- None (Parse from text) --":
        ref_player = selected_ref_helper
        extracted_name_segment = selected_ref_helper
    else:
        # Step A: Direct word match for single names / last names
        # Sort by length descending to match full names first
        for name in sorted(player_names, key=len, reverse=True):
            name_lower = name.lower()
            words_in_name = [w for w in name_lower.split() if len(w) > 2]
            for word in words_in_name:
                if word in ["young", "under", "squad", "club", "league", "winger", "forward", "striker", "midfielder", "defender", "keeper", "player", "similar", "like"]:
                    continue
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, q_lower):
                    ref_player = name
                    extracted_name_segment = word
                    break
            if ref_player:
                break

        # Step B: Fallback to regex patterns and fuzzy matching
        if not ref_player:
            patterns = [
                r"like\s+([a-zA-Z\s'\-ÆæØøÅåÉéÈèÜüÄäÖöííááóóúúññ]+)",
                r"similar\s+to\s+([a-zA-Z\s'\-ÆæØøÅåÉéÈèÜüÄäÖöííááóóúúññ]+)",
                r"type\s+of\s+([a-zA-Z\s'\-ÆæØøÅåÉéÈèÜüÄäÖöííááóóúúññ]+)",
                r"replacement\s+for\s+([a-zA-Z\s'\-ÆæØøÅåÉéÈèÜüÄäÖöííááóóúúññ]+)",
                r"alternative\s+to\s+([a-zA-Z\s'\-ÆæØøÅåÉéÈèÜüÄäÖöííááóóúúññ]+)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, q_lower)
                for match in matches:
                    match_clean = match.strip()
                    if len(match_clean) < 3:
                        continue
                    # Fuzzy match the extracted segment against database
                    res = rf_process.extractOne(match_clean, player_names, scorer=rf_fuzz.WRatio)
                    if res:
                        name, score, _ = res
                        if score >= 60:  # threshold of 60% for typo tolerance
                            ref_player = name
                            extracted_name_segment = match_clean
                            break
                if ref_player:
                    break

        # Step C: Fallback to scanning all n-grams in query for player name fuzzy match
        if not ref_player:
            words = q_lower.split()
            best_name = None
            best_score = 0
            best_segment = ""
            
            # Build last-name lookup for better typo matching
            last_name_map = {}
            for name in player_names:
                parts = name.split()
                if len(parts) >= 2:
                    last_name_map[parts[-1].lower()] = name
            
            for n in range(1, min(4, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n])
                    if len(ngram) < 3:
                        continue
                    if ngram in ["i want a", "looking for", "under a", "under", "million"]:
                        continue
                    
                    # Try last-name matching first (better for typos like "mesi" -> "Messi")
                    res_last = rf_process.extractOne(ngram, list(last_name_map.keys()), scorer=rf_fuzz.WRatio)
                    if res_last and res_last[1] >= 60:
                        candidate = last_name_map[res_last[0]]
                        score = res_last[1]
                        if score > best_score:
                            best_name = candidate
                            best_score = score
                            best_segment = ngram
                        continue
                    
                    # Fall back to full name matching
                    res = rf_process.extractOne(ngram, player_names, scorer=rf_fuzz.WRatio)
                    if res:
                        name, score, _ = res
                        if score >= 60 and score > best_score:
                            best_name = name
                            best_score = score
                            best_segment = ngram
                            
            if best_name:
                ref_player = best_name
                extracted_name_segment = best_segment

    # 2. Parse budget
    budget = None
    budget_patterns = [
        r'(?:under|below|less\s+than|max|maximum|budget|price|value|cost|limit|cap|of|to)\s*[\$€£]?\s*(\d+(?:\.\d+)?)\s*(?:m|million|M)\b',
        r'[\$€£]\s*(\d+(?:\.\d+)?)\s*(?:m|million|M)\b',
        r'(?:under|below|less\s+than|max|maximum|budget)\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})+|\d{6,10})\b'
    ]
    for pattern in budget_patterns:
        matches = re.findall(pattern, q_lower)
        if matches:
            val_str = matches[0].replace(',', '')
            try:
                val = float(val_str)
                if 'm' in pattern or 'million' in pattern:
                    budget = val * 1e6
                else:
                    budget = val
                break
            except ValueError:
                continue

    if not budget:
        # General backup search for "Xm" or "X million"
        matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:m|million|M)\b', q_lower)
        if matches:
            try:
                budget = float(matches[0]) * 1e6
            except ValueError:
                pass

    # 3. Parse position
    position = None
    pos_map = {
        "winger": "FW",
        "striker": "FW",
        "forward": "FW",
        "attacker": "FW",
        "playmaker": "MF",
        "midfielder": "MF",
        "midfield": "MF",
        "mid": "MF",
        "defender": "DF",
        "defense": "DF",
        "def": "DF",
        "center back": "DF",
        "centre back": "DF",
        "cb": "DF",
        "fullback": "DF",
        "full back": "DF",
        "wingback": "DF",
        "wing back": "DF",
        "goalkeeper": "GK",
        "keeper": "GK",
        "gk": "GK"
    }
    for word, pos_code in pos_map.items():
        if re.search(r'\b' + re.escape(word) + r'\b', q_lower):
            position = pos_code
            break

    # 4. Parse World Cup filter
    is_wc_query = "world cup" in q_lower or "wc" in q_lower or "national team" in q_lower

    # 5. Extract descriptive style keywords (remaining clean query text)
    clean_text = query.lower()
    clean_text = re.sub(r'(?:under|below|less\s+than|max|maximum|budget|price|value|cost|limit|cap|of|to)?\s*[\$€£]?\s*\d+(?:\.\d+)?\s*(?:m|million|M)\b', '', clean_text)
    clean_text = re.sub(r'[\$€£]\s*\d+(?:\.\d+)?\s*(?:m|million|M)\b', '', clean_text)
    clean_text = re.sub(r'(?:under|below|less\s+than|max|maximum|budget)\s*[\$€£]?\s*(?:\d{1,3}(?:,\d{3})+|\d{6,10})\b', '', clean_text)
    
    if ref_player:
        clean_text = clean_text.replace(ref_player.lower(), '')
        for part in ref_player.split():
            if len(part) > 2:
                clean_text = clean_text.replace(part.lower(), '')
    if extracted_name_segment:
        clean_text = clean_text.replace(extracted_name_segment, '')
        
    fillers = [
        "i want a", "i want", "want a", "looking for a", "looking for", "find a", "find",
        "player like", "similar to", "like", "type of", "replacement for", "alternative to",
        "under", "less than", "maximum of", "a player", "players", "scout", "search for",
        "winger", "striker", "forward", "attacker", "playmaker", "midfielder", "midfield", "mid",
        "defender", "defense", "def", "center back", "centre back", "cb", "fullback", "full back",
        "wingback", "wing back", "goalkeeper", "keeper", "gk", "world cup", "wc", "national team"
    ]
    for filler in fillers:
        clean_text = re.sub(r'\b' + re.escape(filler) + r'\b', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # ── Display Parsed Criteria Badges ──────────────────────────────────────────
    st.write("🔍 **Scouting Criteria Detected:**")
    badge_container = "<div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1.5rem;'>"
    
    if ref_player:
        is_corrected = ref_player.lower() != extracted_name_segment.lower() and len(extracted_name_segment) > 0
        if is_corrected:
            badge_container += f"<span style='display:inline-block; background:rgba(108,99,255,0.15); border:1px solid #6C63FF; border-radius:8px; padding:4px 12px; font-size:13px; color:#A89CFF;'>👤 Reference: <b>{ref_player}</b> <span style='color:#FFC107;'>(corrected from '{extracted_name_segment}')</span></span>"
        else:
            badge_container += f"<span style='display:inline-block; background:rgba(108,99,255,0.15); border:1px solid #6C63FF; border-radius:8px; padding:4px 12px; font-size:13px; color:#A89CFF;'>👤 Reference: <b>{ref_player}</b></span>"
    else:
        badge_container += "<span style='display:inline-block; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); border-radius:8px; padding:4px 12px; font-size:13px; color:rgba(232,235,240,0.6);'>👤 Reference: <i>None (pure description search)</i></span>"

    if budget:
        budget_str = format_market_value(budget)
        badge_container += f"<span style='display:inline-block; background:rgba(0,210,168,0.12); border:1px solid #00D2A8; border-radius:8px; padding:4px 12px; font-size:13px; color:#00D2A8;'>💰 Max Budget: <b>{budget_str}</b></span>"
    else:
        badge_container += "<span style='display:inline-block; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); border-radius:8px; padding:4px 12px; font-size:13px; color:rgba(232,235,240,0.6);'>💰 Max Budget: <i>Unlimited</i></span>"

    if position:
        badge_container += f"<span style='display:inline-block; background:rgba(255,193,7,0.12); border:1px solid #FFC107; border-radius:8px; padding:4px 12px; font-size:13px; color:#FFC107;'>📍 Position: <b>{position}</b></span>"
    else:
        badge_container += "<span style='display:inline-block; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); border-radius:8px; padding:4px 12px; font-size:13px; color:rgba(232,235,240,0.6);'>📍 Position: <i>Any</i></span>"

    if is_wc_query:
        badge_container += f"<span style='display:inline-block; background:rgba(0,210,168,0.15); border:1px solid #00D2A8; border-radius:8px; padding:4px 12px; font-size:13px; color:#00D2A8;'>🏆 League: <b>World Cup 2026 Squad</b></span>"

    if clean_text:
        badge_container += f"<span style='display:inline-block; background:rgba(108,99,255,0.1); border:1px solid rgba(108,99,255,0.3); border-radius:8px; padding:4px 12px; font-size:13px; color:#E8EBF0;'>🏷️ Style Tags: <i>\"{clean_text}\"</i></span>"
        
    badge_container += "</div>"
    st.markdown(badge_container, unsafe_allow_html=True)

    # ── Reference Player Card ───────────────────────────────────────────────────
    if ref_player and name_col:
        player_row = df[df[name_col].str.lower() == ref_player.lower()]
        if not player_row.empty:
            r = player_row.iloc[0]
            mv = format_market_value(r.get("market_value_eur"))
            nation = str(r.get("nation", r.get("nationality_tm", "N/A")))
            avatar_url = get_player_image_url(r, ref_player)
            flag_and_nation = get_flag_and_nation_html(r)
            
            wc_badge = ""
            try:
                if int(float(r.get("is_world_cup", 0))) == 1:
                    wc_badge = "<span style='display:inline-block;background:rgba(0,210,168,0.15);border:1px solid rgba(0,210,168,0.4);border-radius:8px;padding:3px 10px;font-size:12px;font-weight:600;color:#00D2A8;margin-left:8px;'>🏆 World Cup</span>"
            except Exception:
                pass

            st.markdown(f"""
            <div style='background:rgba(108,99,255,0.06); border:1px solid rgba(108,99,255,0.25); border-radius:16px; padding:16px 20px; margin-bottom:1.5rem;'>
                <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;'>
                    <div style='display:flex; align-items:center; gap:20px;'>
                        <img src='{avatar_url}' width='80' height='80' style='border-radius:16px; object-fit: cover; flex-shrink: 0; background: rgba(255,255,255,0.05); border: 2px solid rgba(108,99,255,0.45); box-shadow: 0 4px 16px rgba(0,0,0,0.4);'/>
                        <div>
                            <div style='font-size:20px; font-weight:700; color:#E8EBF0;'>Comparing against: {ref_player} {wc_badge}</div>
                            <div style='font-size:13px; color:rgba(232,235,240,0.5); margin-top:4px; display:flex; align-items:center; gap:6px;'>
                                📍 {r.get('pos','N/A')} &nbsp;&bull;&nbsp; 🏟 {r.get('squad','N/A')} &nbsp;&bull;&nbsp; 🌍 {r.get('league','N/A')} &nbsp;&bull;&nbsp; 💶 {mv} &nbsp;&bull;&nbsp; {flag_and_nation}
                            </div>
                        </div>
                    </div>
                    <div>
                        <span style='display:inline-block;background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.3);border-radius:8px;padding:3px 10px;font-size:12px;font-weight:600;color:#A89CFF;margin:2px;'>⚽ {r.get('gls_per90', 0):.2f} G/90</span>
                        <span style='display:inline-block;background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.3);border-radius:8px;padding:3px 10px;font-size:12px;font-weight:600;color:#A89CFF;margin:2px;'>🅰️ {r.get('ast_per90', 0):.2f} A/90</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Run Recommendations ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🎯 Scout Recommendations</div>", unsafe_allow_html=True)

    results = pd.DataFrame()
    if rec is None:
        results = _demo_recommendations(df, ref_player or "Query", k=5)
        if budget:
            results = results[results["market_value_eur"] <= budget]
        if position:
            results = results[results["pos"].str.contains(position, na=False)]
        if is_wc_query and "is_world_cup" in results.columns:
            results = results[results["is_world_cup"] == 1]
    else:
        try:
            # Case 1: Search by Reference Player
            if ref_player:
                if budget:
                    results = rec.find_budget_replacement(
                        ref_player,
                        budget=budget,
                        k=25,
                        same_position=(position is not None),
                    )
                    if position and not results.empty:
                        results = results[results["position"].str.contains(position, case=False, na=False)]
                else:
                    results = rec.find_similar(
                        ref_player,
                        k=25,
                        position_filter=position,
                    )
                
                # Fetch is_world_cup flag for results since recommender formats results
                if not results.empty and "is_world_cup" not in results.columns:
                    is_wc_vals = []
                    for _, res_row in results.iterrows():
                        p_name = res_row["player"]
                        match_row = df[df[name_col].str.lower() == p_name.lower()]
                        is_wc_vals.append(int(float(match_row.iloc[0].get("is_world_cup", 0))) if not match_row.empty else 0)
                    results["is_world_cup"] = is_wc_vals

                # Filter by World Cup query
                if is_wc_query and not results.empty:
                    results = results[results["is_world_cup"] == 1]

                # Apply description text search if style tags are present
                if clean_text and not results.empty:
                    text_embedder = load_text_embedder()
                    q_text_emb = text_embedder.encode_single(clean_text).reshape(1, -1)
                    
                    from src.embeddings import load_embeddings
                    emb_data = load_embeddings(method="umap")
                    text_embeddings = emb_data["text"]
                    
                    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
                    text_sims = sk_cosine(q_text_emb, text_embeddings)[0]
                    
                    # Blend player similarity: 60% styling reference player, 40% description tags similarity
                    for i, r_row in results.iterrows():
                        p_name = r_row["player"]
                        p_idx_list = df[df[name_col].str.lower() == p_name.lower()].index.tolist()
                        if p_idx_list:
                            p_idx = p_idx_list[0]
                            desc_sim = float(text_sims[p_idx])
                            ref_sim = float(r_row["similarity"])
                            results.at[i, "similarity"] = round(0.6 * ref_sim + 0.4 * desc_sim, 4)
                    
                    results = results.sort_values(by="similarity", ascending=False).reset_index(drop=True)
                    results["rank"] = range(1, len(results) + 1)

            # Case 2: Pure Description Search
            elif clean_text or is_wc_query:
                # If pure World Cup search, use centroid of World Cup players or description
                from src.embeddings import load_embeddings
                emb_data = load_embeddings(method="umap")
                
                desc_query = clean_text if clean_text else "World Cup 2026 squad player"
                
                text_embedder = load_text_embedder()
                q_text_emb = text_embedder.encode_single(desc_query).reshape(1, -1)
                text_embeddings = emb_data["text"]
                
                from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
                text_sims = sk_cosine(q_text_emb, text_embeddings)[0]
                
                candidates = df.copy()
                candidates["similarity"] = text_sims
                
                if budget:
                    candidates = candidates[candidates["market_value_eur"] <= budget]
                if position:
                    candidates = candidates[candidates["pos"].str.contains(position, case=False, na=False)]
                if is_wc_query:
                    candidates = candidates[candidates["is_world_cup"] == 1]
                
                candidates = candidates.sort_values(by="similarity", ascending=False).head(10)
                
                results = candidates.copy()
                results.insert(0, "rank", range(1, len(results) + 1))
                if "position" not in results.columns and "pos" in results.columns:
                    results["position"] = results["pos"]
        except Exception as e:
            st.error(f"Search failed: {e}")
            results = pd.DataFrame()

    if not results.empty:
        avail_na = sorted([NATION_LABELS.get(n,n) for n in df["nation"].dropna().unique() if isinstance(n,str)])
        aicf = st.multiselect("🌍 Filter results by Country", options=avail_na, default=[],
                               placeholder="All countries...", key="ai_scout_cf")
        results = apply_country_filter(results, aicf)
        if results.empty and aicf:
            st.info("No players found for the selected countries. Try a different filter.")
        display_results = results.head(5)
        for _, row in display_results.iterrows():
            render_player_card(row, rank=int(row.get("rank", 0)))
            
        if "similarity" in results.columns:
            fig_bar = px.bar(
                display_results,
                x="player",
                y="similarity",
                color="similarity",
                color_continuous_scale=["#1a1a3e", "#6C63FF", "#00D2A8"],
                labels={"player": "Player", "similarity": "Similarity Score"},
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(t=20, b=40, l=0, r=0),
                height=280,
            )
            fig_bar.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})
    else:
        st.info("No players matching the criteria found. Try a broader search description or budget.")


# ─── Demo Recommendations (fallback when model not loaded) ───────────────────────

def _demo_recommendations(df: pd.DataFrame, query_player: str, k: int) -> pd.DataFrame:
    """Generate random demo recommendations when real model is unavailable."""
    name_col = next((c for c in ["player", "name"] if c in df.columns), None)
    if name_col is None:
        return pd.DataFrame()

    pool   = df[df[name_col] != query_player]
    n      = min(k, len(pool))
    sample = pool.sample(n, random_state=42).copy()

    np.random.seed(hash(query_player) % (2**31))
    sample["similarity"] = np.sort(np.random.uniform(0.55, 0.97, n))[::-1].round(4).tolist()
    sample["rank"]       = range(1, n + 1)
    # Map pos → position column so render_player_card finds it
    if "pos" in sample.columns:
        sample["position"] = sample["pos"]
    return sample.reset_index(drop=True)


# ─── Main App ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    mtime = get_csv_mtime()
    
    # Load data
    with st.spinner("Loading player data..."):
        df = load_player_data(mtime)

    # Sidebar
    page, embedding_type, alpha = render_sidebar(df)

    # Load recommender (may fail if embeddings not built yet)
    rec, load_error = load_recommender_cached(method="umap", alpha=alpha, mtime=mtime)

    if load_error:
        st.sidebar.warning(
            f"⚠️ Model not loaded: Run the embedding pipeline first.\n\n"
            f"Demo mode active.",
            icon="⚠️",
        )
        rec = None

    # Render selected page
    if page == "🤖 AI Scout":
        page_ai_scout(df, rec, embedding_type)
    elif page == "🔍 Player Finder":
        page_player_finder(df, rec, embedding_type)
    elif page == "💰 Budget Scout":
        page_budget_scout(df, rec)
    elif page == "💎 Hidden Gem Explorer":
        page_hidden_gems(df, rec)


if __name__ == "__main__":
    main()
