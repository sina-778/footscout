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
from pathlib import Path

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

# ─── Custom CSS (Dark Glassmorphism Theme) ──────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E8EBF0;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(135deg, #0F1117 0%, #111827 40%, #0D1B2A 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 17, 26, 0.95);
        border-right: 1px solid rgba(108, 99, 255, 0.2);
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #6C63FF;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(0,210,168,0.08) 100%);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(108,99,255,0.1);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #00D2A8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }
    .hero-subtitle {
        color: rgba(232, 235, 240, 0.7);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108,99,255,0.15);
    }

    /* ── Player Card ── */
    .player-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.1) 0%, rgba(0,210,168,0.06) 100%);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .player-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #6C63FF;
        margin: 0;
    }
    .player-meta {
        color: rgba(232, 235, 240, 0.65);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* ── Metric Badge ── */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(108, 99, 255, 0.15);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #A89CFF;
        margin: 0.15rem;
    }
    .metric-badge.green {
        background: rgba(0, 210, 168, 0.12);
        border-color: rgba(0, 210, 168, 0.3);
        color: #00D2A8;
    }
    .metric-badge.gold {
        background: rgba(255, 193, 7, 0.12);
        border-color: rgba(255, 193, 7, 0.3);
        color: #FFC107;
    }

    /* ── Result Table ── */
    .rec-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 0.4rem;
    }
    .rec-table th {
        color: rgba(232,235,240,0.5);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .rec-table td {
        background: rgba(255,255,255,0.03);
        padding: 0.6rem 0.8rem;
        font-size: 0.92rem;
    }
    .rec-table tr:hover td {
        background: rgba(108,99,255,0.08);
    }

    /* ── Similarity Bar ── */
    .sim-bar-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sim-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #6C63FF, #00D2A8);
    }
    .sim-value {
        font-size: 0.8rem;
        color: rgba(232,235,240,0.7);
        white-space: nowrap;
    }

    /* ── Page Navigation Pills ── */
    .nav-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    /* ── Section Title ── */
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Streamlit overrides ── */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSlider"] > div,
    div[data-testid="stNumberInput"] > div {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #5B52E8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7C73FF 0%, #6B62F8 100%);
        box-shadow: 0 4px 20px rgba(108,99,255,0.4);
        transform: translateY(-1px);
    }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.8rem;
    }
    div[data-testid="metric-container"] label {
        color: rgba(232,235,240,0.6) !important;
        font-size: 0.8rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #A89CFF !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data & Model Loading ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_player_data() -> pd.DataFrame:
    """Load the merged player dataset."""
    path = PROJECT_ROOT / "data" / "processed" / "players_merged.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        return df
    # Demo data for development (no real data yet)
    return _generate_demo_data()


@st.cache_resource(show_spinner=False)
def load_recommender_cached(method: str = "umap", alpha: float = 0.6):
    """Load (or build) the recommender engine. Cached as a singleton."""
    try:
        from src.recommender import load_recommender
        df = load_player_data()
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

def get_flag_html(nation: str) -> str:
    """Generate HTML image tag for country flag using FlagCDN."""
    if not isinstance(nation, str) or not nation or nation == "N/A":
        return ""
    FLAG_MAP = {
        "ENG": "gb-eng", "ESP": "es", "GER": "de", "FRA": "fr", "BRA": "br", "IT": "it", "ITA": "it",
        "ARG": "ar", "NED": "nl", "POR": "pt", "BEL": "be", "SUI": "ch", "CRO": "hr",
        "USA": "us", "MAR": "ma", "JPN": "jp", "SEN": "sn", "SRB": "rs", "NOR": "no",
        "COL": "co", "URU": "uy", "DEN": "dk", "MEX": "mx", "CAN": "ca", "NGA": "ng",
        "SCO": "gb-sct", "KOR": "kr", "CIV": "ci", "AUT": "at", "POL": "pl", "TUR": "tr",
        "GEO": "ge", "AUS": "au", "GHA": "gh", "UKR": "ua", "ECU": "ec", "IRN": "ir",
        "EGY": "eg", "HUN": "hu", "SWE": "se", "CMR": "cm", "SVN": "si", "BFA": "bf",
        "ALG": "dz", "SVK": "sk", "GAB": "ga", "MLI": "ml", "WAL": "gb-wls", "ISR": "il",
        "JAM": "jm", "IRL": "ie", "FIN": "fi", "CZE": "cz", "ARM": "am", "KOS": "xk",
        "GUI": "gn", "COD": "cd"
    }
    flag_code = FLAG_MAP.get(nation.upper(), "un")
    return f"<img src='https://flagcdn.com/w20/{flag_code}.png' style='vertical-align: middle; margin-right: 4px; border-radius: 2px; border: 1px solid rgba(255,255,255,0.15);' width='16' title='{nation}'/>"


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


def get_player_avatar_svg(name: str) -> str:
    """Generate a premium base64-encoded SVG circular avatar badge with initials and gradient background."""
    import hashlib
    import base64
    
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16)
    
    gradients = [
        ("#6C63FF", "#3F37C9"), # Indigo-Purple
        ("#00D2A8", "#0077B6"), # Turquoise-Blue
        ("#FFC107", "#E63946"), # Amber-Red
        ("#F72585", "#7209B7"), # Pink-Purple
        ("#4CC9F0", "#4895EF"), # SkyBlue-Blue
        ("#FF9F1C", "#FF4000"), # Orange-Red
        ("#70E000", "#38B000"), # Green-Lime
    ]
    c1, c2 = gradients[h % len(gradients)]
    
    parts = [p for p in name.split() if p]
    if len(parts) >= 2:
        initials = (parts[0][0] + parts[-1][0]).upper()
    elif len(parts) == 1:
        initials = parts[0][:2].upper()
    else:
        initials = "FS"
        
    svg = f"""
    <svg width='44' height='44' viewBox='0 0 44 44' fill='none' xmlns='http://www.w3.org/2000/svg' style='border-radius: 50%; border: 1.5px solid rgba(255, 255, 255, 0.25); box-shadow: 0 2px 8px rgba(0,0,0,0.4);'>
        <defs>
            <linearGradient id='grad_{h}' x1='0%' y1='0%' x2='100%' y2='100%'>
                <stop offset='0%' stop-color='{c1}' />
                <stop offset='100%' stop-color='{c2}' />
            </linearGradient>
        </defs>
        <circle cx='22' cy='22' r='22' fill='url(#grad_{h})' />
        <text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle' fill='#FFFFFF' font-size='14' font-family='Outfit, Inter, sans-serif' font-weight='700' letter-spacing='0.5'>{initials}</text>
    </svg>
    """
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"


def render_player_card(row: pd.Series, rank: int = 0, show_gem_badge: bool = False) -> None:
    """Render a player recommendation card with fully inlined styles."""
    import html as _html
    mv      = format_market_value(row.get("market_value_eur"))
    sim     = float(row.get("similarity", 0) or 0)
    sim_pct = f"{sim*100:.1f}%"

    player_name = _html.escape(str(row.get("player", "N/A")))
    position    = _html.escape(str(row.get("position", row.get("pos", "N/A"))))
    squad       = _html.escape(str(row.get("squad", "N/A")))
    league      = _html.escape(str(row.get("league", "N/A")))
    nation      = _html.escape(str(row.get("nation", row.get("nationality_tm", "N/A"))))

    def _stat(key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    sim_color  = "#00D2A8" if sim >= 0.8 else ("#6C63FF" if sim >= 0.6 else "#FFC107")
    rank_str   = f"#{rank}&nbsp;" if rank > 0 else ""

    # Pure inline styles — avoids Streamlit sanitizer blocking class= attributes
    badge      = "display:inline-block;background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.3);border-radius:8px;padding:3px 10px;font-size:12px;font-weight:600;color:#A89CFF;margin:2px;"
    badge_grn  = "display:inline-block;background:rgba(0,210,168,0.12);border:1px solid rgba(0,210,168,0.3);border-radius:8px;padding:3px 10px;font-size:12px;font-weight:600;color:#00D2A8;margin:2px;"
    badge_gold = "display:inline-block;background:rgba(255,193,7,0.12);border:1px solid rgba(255,193,7,0.3);border-radius:8px;padding:3px 10px;font-size:12px;font-weight:600;color:#FFC107;margin:2px;"
    sim_badge  = f"display:inline-block;background:rgba(255,255,255,0.05);border:1px solid {sim_color}66;border-radius:8px;padding:3px 10px;font-size:12px;font-weight:700;color:{sim_color};margin:2px;"
    gem_html   = f"<span style='{badge_gold}'>&#128142; Hidden Gem</span>" if show_gem_badge else ""
    
    # World Cup Badge
    wc_html = ""
    try:
        is_wc = int(float(row.get("is_world_cup", 0)))
        if is_wc == 1:
            wc_html = f"<span style='{badge_grn}'>🏆 World Cup</span>"
    except Exception:
        pass

    avatar_url = row.get("image_url")
    if not isinstance(avatar_url, str) or pd.isna(avatar_url) or not avatar_url:
        avatar_url = get_player_avatar_url(player_name)
    flag_html = get_flag_html(nation)

    st.markdown(
        f"<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);"
        f"border-radius:14px;padding:14px 18px;margin-bottom:10px;"
        f"box-shadow:0 4px 16px rgba(0,0,0,0.2);'>"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:flex-start;'>"
        f"<div style='display:flex;align-items:center;gap:12px;'>"
        f"<img src='{avatar_url}' width='44' height='44' style='border-radius:50%; flex-shrink: 0; background: rgba(255,255,255,0.05);'/>"
        f"<div>"
        f"<div style='font-size:15px;font-weight:700;color:#E8EBF0;'>{rank_str}{player_name}</div>"
        f"<div style='font-size:12px;color:rgba(232,235,240,0.5);margin-top:2px;display:flex;align-items:center;gap:4px;'>{position} &bull; {squad} &bull; {league} &bull; {flag_html} {nation}</div>"
        f"</div>"
        f"</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:center;'>"
        f"<span style='{sim_badge}'>{sim_pct} match</span>"
        f"<span style='{badge_grn}'>{mv}</span>"
        f"{wc_html}"
        f"{gem_html}"
        f"</div></div>"
        f"<div style='margin-top:10px;display:flex;flex-wrap:wrap;gap:4px;'>"
        f"<span style='{badge}'>&#9917; {_stat('gls_per90'):.2f} G/90</span>"
        f"<span style='{badge}'>&#127170; {_stat('ast_per90'):.2f} A/90</span>"
        f"<span style='{badge}'>&#128200; {_stat('xg_per90'):.2f} xG/90</span>"
        f"<span style='{badge}'>&#128737; {_stat('tackles_tkl_per90'):.2f} Tkl/90</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


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
            avatar_url = r.get("image_url")
            if not isinstance(avatar_url, str) or pd.isna(avatar_url) or not avatar_url:
                avatar_url = get_player_avatar_url(selected_player)
            flag_html = get_flag_html(nation)

            st.markdown(f"""
            <div class='player-card' style='background:linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(0,210,168,0.08) 100%); border:1px solid rgba(108,99,255,0.45); border-radius:20px; padding:1.5rem; margin-bottom:1rem;'>
                <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; align-items:center;'>
                    <div style='display:flex; align-items:center; gap:14px;'>
                        <img src='{avatar_url}' width='56' height='56' style='border-radius:50%; background:rgba(255,255,255,0.05); border:1.5px solid rgba(108,99,255,0.3); flex-shrink:0;'/>
                        <div>
                            <div class='player-name' style='font-size:2rem;'>{r.get('player', selected_player)}</div>
                            <div class='player-meta' style='display:flex; align-items:center; gap:6px; flex-wrap:wrap; color:rgba(232, 235, 240, 0.65); font-size:0.95rem; margin-top:0.3rem;'>
                                📍 {r.get('pos','N/A')} &nbsp;&bull;&nbsp;
                                🏟 {r.get('squad','N/A')} &nbsp;&bull;&nbsp;
                                🌍 {r.get('league','N/A')} &nbsp;&bull;&nbsp;
                                🎂 Age {r.get('age','N/A')} &nbsp;&bull;&nbsp;
                                💶 {mv} &nbsp;&bull;&nbsp;
                                {flag_html} {nation}
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
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
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
                query_val = player_row.iloc[0].get("market_value_eur")
                query_str = format_market_value(query_val)
                st.markdown(f"""
                <div class='glass-card'>
                    🎯 Replacing <b>{selected_player}</b>
                    (value: <b>{query_str}</b>) with players under <b>€{budget_m}M</b>
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
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
            st.markdown(
                f"<div class='glass-card'>💎 Found <b>{len(results)}</b> hidden gems "
                f"under <b>€{max_val_m}M</b> matching the <b>{position}</b> profile"
                + (f" (reference: {reference})" if reference else "") + "</div>",
                unsafe_allow_html=True,
            )

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
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
                res = rf_process.extractOne(match_clean, player_names, scorer=rf_fuzz.token_sort_ratio)
                if res:
                    name, score, _ = res
                    if score >= 75:  # threshold of 75%
                        ref_player = name
                        extracted_name_segment = match_clean
                        break
            if ref_player:
                break

        # Fallback to scanning all n-grams in query for player name fuzzy match
        if not ref_player:
            words = q_lower.split()
            best_name = None
            best_score = 0
            best_segment = ""
            
            for n in range(1, min(4, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n])
                    if len(ngram) < 3:
                        continue
                    if ngram in ["i want a", "looking for", "under a", "under", "million"]:
                        continue
                    res = rf_process.extractOne(ngram, player_names, scorer=rf_fuzz.token_sort_ratio)
                    if res:
                        name, score, _ = res
                        if score >= 80 and score > best_score:
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
            avatar_url = r.get("image_url")
            if not isinstance(avatar_url, str) or pd.isna(avatar_url) or not avatar_url:
                avatar_url = get_player_avatar_url(ref_player)
            flag_html = get_flag_html(nation)
            
            wc_badge = ""
            try:
                if int(float(r.get("is_world_cup", 0))) == 1:
                    wc_badge = "<span style='display:inline-block;background:rgba(0,210,168,0.15);border:1px solid rgba(0,210,168,0.4);border-radius:8px;padding:3px 10px;font-size:12px;font-weight:600;color:#00D2A8;margin-left:8px;'>🏆 World Cup</span>"
            except Exception:
                pass

            st.markdown(f"""
            <div style='background:rgba(108,99,255,0.06); border:1px solid rgba(108,99,255,0.25); border-radius:14px; padding:14px 18px; margin-bottom:1.5rem;'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;'>
                    <div style='display:flex; align-items:center; gap:12px;'>
                        <img src='{avatar_url}' width='44' height='44' style='border-radius:50%; flex-shrink: 0; background: rgba(255,255,255,0.05);'/>
                        <div>
                            <div style='font-size:16px; font-weight:700; color:#E8EBF0;'>Comparing against: {ref_player} {wc_badge}</div>
                            <div style='font-size:12px; color:rgba(232,235,240,0.5); margin-top:2px; display:flex; align-items:center; gap:4px;'>
                                📍 {r.get('pos','N/A')} &nbsp;&bull;&nbsp; 🏟 {r.get('squad','N/A')} &nbsp;&bull;&nbsp; 🌍 {r.get('league','N/A')} &nbsp;&bull;&nbsp; 💶 {mv} &nbsp;&bull;&nbsp; {flag_html} {nation}
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
                
                formatted_rows = []
                for rank, (idx, crow) in enumerate(candidates.iterrows(), start=1):
                    formatted_rows.append({
                        "rank": rank,
                        "player": crow[name_col],
                        "similarity": round(float(crow["similarity"]), 4),
                        "position": crow["pos"],
                        "squad": crow["squad"],
                        "league": crow["league"],
                        "market_value_eur": crow["market_value_eur"],
                        "age": crow["age"],
                        "is_world_cup": crow["is_world_cup"],
                        "gls_per90": crow.get("gls_per90", 0.0),
                        "ast_per90": crow.get("ast_per90", 0.0),
                        "xg_per90": crow.get("xg_per90", 0.0),
                        "tackles_tkl_per90": crow.get("tackles_tkl_per90", 0.0),
                    })
                results = pd.DataFrame(formatted_rows)
        except Exception as e:
            st.error(f"Search failed: {e}")
            results = pd.DataFrame()

    if not results.empty:
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
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
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
    sample["similarity"] = np.sort(np.random.uniform(0.55, 0.97, n))[::-1].round(4)
    sample["rank"]       = range(1, n + 1)
    # Map pos → position column so render_player_card finds it
    if "pos" in sample.columns:
        sample["position"] = sample["pos"]
    return sample.reset_index(drop=True)


# ─── Main App ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    # Load data
    with st.spinner("Loading player data..."):
        df = load_player_data()

    # Sidebar
    page, embedding_type, alpha = render_sidebar(df)

    # Load recommender (may fail if embeddings not built yet)
    rec, load_error = load_recommender_cached(method="umap", alpha=alpha)

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
