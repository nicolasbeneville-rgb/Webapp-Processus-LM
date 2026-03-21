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

# ── Titre ML Studio tout en haut de la sidebar ──
st.sidebar.markdown(
    '<p style="margin:0;padding:4px 0 2px;font-size:1.1rem;font-weight:800;'
    'color:#F59E0B;letter-spacing:-0.01em;">⚗️ ML STUDIO</p>',
    unsafe_allow_html=True,
)

# ── Navigation multi-pages avec labels personnalisés ──
_page_main = st.Page("app_pipeline.py", title="Créer un Modèle", icon="⚗️", default=True)
_page_pred = st.Page("app_prediction.py", title="Faire une Prédiction", icon="🔮")
_nav = st.navigation([_page_main, _page_pred])

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

/* ══════════════════════════════════════════
   PALETTE & DESIGN TOKENS
   ══════════════════════════════════════════ */
:root {
    --primary: #0F766E;
    --primary-hover: #115E59;
    --primary-light: #D1FAF3;
    --primary-subtle: #ECFEFA;
    --accent: #B45309;
    --success: #166534;
    --success-light: #DCFCE7;
    --warning: #B45309;
    --warning-light: #FEF3C7;
    --danger: #B91C1C;
    --danger-light: #FEE2E2;
    --surface: #FFFFFF;
    --surface-raised: #FFFFFF;
    --bg: #F2F5F7;
    --text: #0B1623;
    --text-secondary: #344054;
    --text-muted: #9CA3AF;
    --border: #D8E0E7;
    --border-light: #E7EDF2;
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
    background:
        radial-gradient(circle at 12% 14%, #E6F7F3 0%, transparent 42%),
        radial-gradient(circle at 85% 2%, #FFF7E8 0%, transparent 34%),
        linear-gradient(180deg, #F6F8FA 0%, #EEF2F6 100%);
    color: var(--text);
}
* {
    -webkit-font-smoothing: antialiased;
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}

/* ══════════════════════════════════════════
   SIDEBAR — fond bleu foncé, texte clair, compact
   ══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, #0B1220 0%, #111827 56%, #1E293B 100%) !important;
    border-right: none;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.3rem !important;
    padding-bottom: 0.5rem !important;
}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    gap: 0.15rem !important;
}
section[data-testid="stSidebar"] .stElementContainer {
    margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] h1 {
    color: #F59E0B !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    margin: 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #CBD5E1 !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] small {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    transition: background var(--transition);
    border-radius: 4px;
    padding: 1px 4px !important;
    min-height: unset !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stRadio label span {
    color: #E2E8F0 !important;
    font-size: 0.78rem;
    font-weight: 450;
}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] span {
    color: #FFFFFF !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] .stProgress > div > div {
    background: linear-gradient(90deg, #4F5BD5, #7C3AED) !important;
    border-radius: 4px;
    height: 6px !important;
}
section[data-testid="stSidebar"] .stProgress > div {
    background: rgba(255,255,255,0.12) !important;
    height: 6px !important;
}
section[data-testid="stSidebar"] .stProgress {
    margin: 0 !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] .stProgress p {
    display: none !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.1) !important;
    margin: 2px 0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.1) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    padding: 2px 8px !important;
    min-height: unset !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.18) !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button {
    background: rgba(255,255,255,0.1) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    padding: 2px 8px !important;
    min-height: unset !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] details {
    border-color: rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.04) !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] summary {
    padding: 4px 8px !important;
    min-height: unset !important;
}
section[data-testid="stSidebar"] summary span {
    color: #CBD5E1 !important;
    font-size: 0.78rem !important;
}
section[data-testid="stSidebar"] code {
    background: rgba(255,255,255,0.08) !important;
    color: #E2E8F0 !important;
}
/* Sidebar save button compact */
section[data-testid="stSidebar"] .stDownloadButton {
    margin-top: 4px !important;
}

/* ══════════════════════════════════════════
   SIDEBAR — sous-étapes (2e radio) compact
   ══════════════════════════════════════════ */
section[data-testid="stSidebar"] .stRadio + .stMarkdown + .stRadio label {
    padding-left: 10px !important;
    margin-left: 6px;
    border-left: 2px solid rgba(255,255,255,0.15);
}
section[data-testid="stSidebar"] .stRadio + .stMarkdown + .stRadio label span {
    font-size: 0.72rem;
    font-weight: 400;
}
section[data-testid="stSidebar"] .stRadio + .stMarkdown + .stRadio [aria-checked="true"] {
    border-left-color: #F39C12;
}
section[data-testid="stSidebar"] .stRadio + .stMarkdown + .stRadio [aria-checked="true"] span {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* ══════════════════════════════════════════
   TYPOGRAPHY
   ══════════════════════════════════════════ */
h1 {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em;
    color: var(--text) !important;
}
h2 {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important;
    color: #1F2937 !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    letter-spacing: -0.015em;
}
h3 {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important;
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
    box-shadow: 0 2px 10px rgba(15,118,110,0.26);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 14px rgba(15,118,110,0.36);
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

/* ══════════════════════════════════════════
   INDUSTRIAL DASHBOARD COCKPIT
   ══════════════════════════════════════════ */
.industrial-shell {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF4F7 100%);
    border: 1px solid #D6E1E8;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    padding: 14px 16px;
    margin-bottom: 14px;
}
.industrial-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
}
.industrial-title {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
    font-size: 0.98rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #0F172A;
    text-transform: uppercase;
}
.industrial-sub {
    color: #475467;
    font-size: 0.8rem;
}
.industrial-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.industrial-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    color: #1E293B;
}
.industrial-chip.ok {
    border-color: #86EFAC;
    background: #F0FDF4;
    color: #166534;
}
.industrial-chip.warn {
    border-color: #FCD34D;
    background: #FFFBEB;
    color: #92400E;
}
.industrial-chip.critical {
    border-color: #FCA5A5;
    background: #FEF2F2;
    color: #991B1B;
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
