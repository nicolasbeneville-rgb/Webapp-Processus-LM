# -*- coding: utf-8 -*-
"""
app.py — ML Studio — Point d'entrée.

Lance l'application avec :
    streamlit run app.py
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import APP_TITLE, APP_ICON, APP_LAYOUT

# ═══════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

# ── Navigation multi-pages avec labels personnalisés ──
_page_main = st.Page("app_pipeline.py", title="Créer un Modèle", icon="⚗️", default=True)
_page_pred = st.Page("app_prediction.py", title="Faire une Prédiction", icon="🔮")
_nav = st.navigation([_page_main, _page_pred])

# ── Custom CSS ──
st.markdown("""
<style>
/* ══════════════════════════════════════════
   PALETTE & DESIGN TOKENS
   ══════════════════════════════════════════ */
:root {
    --primary: #4F5BD5;
    --primary-hover: #3D48B0;
    --primary-light: #ECEFFE;
    --primary-subtle: #F5F6FF;
    --accent: #7C3AED;
    --success: #059669;
    --success-light: #ECFDF5;
    --warning: #D97706;
    --warning-light: #FFFBEB;
    --danger: #DC2626;
    --danger-light: #FEF2F2;
    --surface: #FFFFFF;
    --surface-raised: #FFFFFF;
    --bg: #F8F9FC;
    --text: #111827;
    --text-secondary: #4B5563;
    --text-muted: #9CA3AF;
    --border: #E5E7EB;
    --border-light: #F3F4F6;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.06);
    --shadow-lg: 0 4px 16px rgba(0,0,0,0.08);
    --radius: 10px;
    --radius-lg: 14px;
    --transition: 0.15s ease;
}

/* ══════════════════════════════════════════
   GLOBAL
   ══════════════════════════════════════════ */
.stApp {
    background-color: var(--bg);
    color: var(--text);
}
* { -webkit-font-smoothing: antialiased; }

/* ══════════════════════════════════════════
   SIDEBAR — fond bleu foncé, texte clair
   ══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2A4A 0%, #162240 100%) !important;
    border-right: none;
}
section[data-testid="stSidebar"] h1 {
    color: #E74C3C !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #CBD5E1 !important;
    font-size: 0.9rem !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] small {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    transition: background var(--transition);
    border-radius: 6px;
    padding: 1px 4px;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stRadio label span {
    color: #E2E8F0 !important;
    font-size: 0.82rem;
    font-weight: 450;
}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] span {
    color: #FFFFFF !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] .stProgress > div > div {
    background: linear-gradient(90deg, #E74C3C, #F39C12) !important;
    border-radius: 4px;
}
section[data-testid="stSidebar"] .stProgress > div {
    background: rgba(255,255,255,0.12) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.1) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.1) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.18) !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button {
    background: rgba(255,255,255,0.1) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] details {
    border-color: rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.04) !important;
}
section[data-testid="stSidebar"] summary span {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] code {
    background: rgba(255,255,255,0.08) !important;
    color: #E2E8F0 !important;
}
/* ML Studio home button — styled as title */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
}
section[data-testid="stSidebar"] button[key="home_btn"],
section[data-testid="stSidebar"] .stButton:first-child > button {
    background: transparent !important;
    border: none !important;
    color: #E74C3C !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    box-shadow: none !important;
    padding: 8px 0 !important;
    text-align: left !important;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stButton:first-child > button:hover {
    color: #FF6B6B !important;
    background: transparent !important;
}

/* ══════════════════════════════════════════
   TYPOGRAPHY
   ══════════════════════════════════════════ */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.025em;
    color: var(--text) !important;
}
h2 {
    color: #1F2937 !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    letter-spacing: -0.015em;
}
h3 {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}
.stMarkdown p { line-height: 1.6; }

/* ══════════════════════════════════════════
   METRIC CARDS
   ══════════════════════════════════════════ */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--transition), transform var(--transition);
}
div[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-weight: 800 !important;
    font-size: 1.4rem !important;
}

/* ══════════════════════════════════════════
   TAB BAR (sticky on scroll)
   ══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--primary-light);
    border-radius: var(--radius);
    padding: 3px;
    position: -webkit-sticky;
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
/* Fix parent overflow for sticky to work inside Streamlit */
.stMainBlockContainer, .block-container,
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"] {
    overflow: visible !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.8rem;
    padding: 6px 14px;
    white-space: nowrap;
    color: var(--text-secondary);
    transition: all var(--transition);
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.6);
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm);
}

/* ══════════════════════════════════════════
   DATAFRAMES
   ══════════════════════════════════════════ */
.stDataFrame {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
}

/* ══════════════════════════════════════════
   EXPANDERS
   ══════════════════════════════════════════ */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.9rem;
    border-radius: var(--radius);
    color: var(--text) !important;
}
details[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface);
}

/* ══════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════ */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.45rem 1.2rem;
    transition: all var(--transition);
    border: 1px solid var(--border);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
    border: none;
    color: white !important;
    box-shadow: 0 2px 8px rgba(79,91,213,0.3);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 12px rgba(79,91,213,0.4);
    transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--primary);
    color: var(--primary);
}
.stDownloadButton > button {
    border-radius: 8px;
    font-size: 0.82rem;
}

/* ══════════════════════════════════════════
   INPUTS & SELECTORS
   ══════════════════════════════════════════ */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    font-size: 0.88rem;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px var(--primary-light) !important;
}

/* ══════════════════════════════════════════
   ALERTS & MESSAGES
   ══════════════════════════════════════════ */
.stAlert {
    border-radius: var(--radius) !important;
    font-size: 0.88rem;
}

/* ══════════════════════════════════════════
   DIVIDERS
   ══════════════════════════════════════════ */
hr {
    border: none;
    height: 1px;
    background: var(--border-light);
    margin: 1.2rem 0;
}

/* ══════════════════════════════════════════
   GUIDE BANNER
   ══════════════════════════════════════════ */
.guide-banner {
    background: linear-gradient(135deg, var(--primary-subtle) 0%, var(--primary-light) 100%);
    border-left: 3px solid var(--primary);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.guide-banner .guide-what {
    font-size: 0.9rem;
    color: var(--text);
    font-weight: 600;
    margin-bottom: 6px;
    line-height: 1.5;
}
.guide-banner .guide-why {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 4px;
    line-height: 1.5;
}
.guide-banner .guide-tip {
    font-size: 0.82rem;
    color: var(--success);
    padding: 8px 12px;
    background: var(--success-light);
    border-radius: 8px;
    margin-top: 8px;
    line-height: 1.5;
}

/* ══════════════════════════════════════════
   RECOMMENDATION CARD
   ══════════════════════════════════════════ */
.reco-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    margin: 8px 0;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--transition);
}
.reco-card:hover { box-shadow: var(--shadow-md); }
.reco-card .reco-label {
    font-weight: 600;
    font-size: 0.88rem;
    margin-bottom: 3px;
    color: var(--text);
}
.reco-card .reco-reason {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ══════════════════════════════════════════
   NEXT STEP BOX
   ══════════════════════════════════════════ */
.next-step-box {
    background: linear-gradient(135deg, var(--success-light) 0%, #D1FAE5 100%);
    border-left: 3px solid var(--success);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 12px 16px;
    margin-top: 14px;
    font-size: 0.88rem;
}

/* ══════════════════════════════════════════
   QUALITY BADGE
   ══════════════════════════════════════════ */
.quality-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 24px;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: -0.02em;
}
.quality-good { background: var(--success-light); color: var(--success); }
.quality-medium { background: var(--warning-light); color: var(--warning); }
.quality-bad { background: var(--danger-light); color: var(--danger); }

/* ══════════════════════════════════════════
   STEP SECTION CARD (wraps main content)
   ══════════════════════════════════════════ */
.step-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
}

/* ══════════════════════════════════════════
   SCROLLBAR
   ══════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: #D1D5DB;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* ══════════════════════════════════════════
   HOME PAGE CARDS
   ══════════════════════════════════════════ */
.home-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 24px;
    text-align: center;
    min-height: 180px;
    transition: box-shadow var(--transition), transform var(--transition);
    box-shadow: var(--shadow-sm);
}
.home-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ── Style matplotlib global ──
plt.rcParams.update({
    "figure.facecolor": "#F8F9FC",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E5E7EB",
    "axes.labelcolor": "#374151",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.grid": True,
    "grid.color": "#F3F4F6",
    "grid.alpha": 0.8,
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#E5E7EB",
    "font.family": "sans-serif",
})

# ── Exécuter la page sélectionnée ──
_nav.run()
