# -*- coding: utf-8 -*-
"""
app.py — ML Studio — Pipeline ML interactif modulaire.

Lance l'application avec :
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import APP_TITLE, APP_ICON, APP_LAYOUT, STEPS

from modules.m1_chargement import afficher_demarrage, afficher_typage, afficher_consolidation, afficher_charger_projet
from modules.m2_diagnostic import afficher_diagnostic, afficher_cible_variables
from modules.m3_nettoyage import afficher_nettoyage, afficher_transformation
from modules.m4_entrainement import afficher_entrainement
from modules.m5_evaluation import afficher_evaluation
from modules.m6_prediction import afficher_optimisation_prediction
from modules.aide_contextuelle import afficher_aide, afficher_glossaire
from utils.projet_manager import exporter_projet_zip


# ═══════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

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


# ═══════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES SIDEBAR
# ═══════════════════════════════════════════════════════════
def _etape_max_accessible() -> int:
    """Calcule le numéro de la dernière étape accessible (progression verrouillée)."""
    checks = [
        ("chargement_done", 1),
        ("typage_done", 2),
        ("consolidation_done", 3),
        ("diagnostic_done", 4),
        ("cible_done", 5),
        ("nettoyage_done", 6),
        ("transformation_done", 7),
        ("entrainement_done", 8),
        ("evaluation_done", 9),
    ]
    max_step = 0
    for key, step in checks:
        if st.session_state.get(key):
            max_step = step
        else:
            break
    return max_step


def _indicateur_statut():
    """Affiche un récap des choix réalisés à chaque étape dans la sidebar."""
    df = st.session_state.get("df_courant")
    if df is None:
        st.caption("📭 Aucun jeu de données chargé")
        return

    n_rows, n_cols = df.shape
    st.markdown(f"**📊 Données** : {n_rows} × {n_cols}")

    # Récap structuré par étape
    rapport = st.session_state.get("rapport", {})
    etape_max = rapport.get("etape_courante", 0)
    if etape_max < 1:
        return

    with st.expander("📋 Récap par étape", expanded=False):
        # Étape 1 — Typage
        if etape_max >= 2:
            tm = rapport.get("type_mapping", {})
            # Lister les conversions non triviales
            conversions = [f"{c} → {t}" for c, t in tm.items()
                           if t not in ("Texte (string)",)]
            if conversions:
                st.caption(f"**1. Typage** — {', '.join(conversions[:5])}")
                if len(conversions) > 5:
                    st.caption(f"   … et {len(conversions) - 5} autre(s)")
            else:
                st.caption(f"**1. Typage** — {len(tm)} colonnes (aucune conversion)")

        # Étape 2 — Consolidation
        if etape_max >= 3:
            st.caption("**2. Consolidation** — ✓")

        # Étape 3 — Diagnostic
        if etape_max >= 4:
            diag = rapport.get("diagnostic", {})
            score_q = diag.get("score_qualite")
            txt = f"**3. Diagnostic** — qualité {score_q:.0f} %" if score_q else "**3. Diagnostic** — ✓"
            st.caption(txt)

        # Étape 4 — Cible & variables
        if etape_max >= 5:
            cible = rapport.get("colonne_cible", "?")
            feats = rapport.get("colonnes_features", [])
            ts_dt = rapport.get("ts_datetime_col")
            if ts_dt:
                st.caption(f"**4. Cible** — {cible} | datetime : {ts_dt}")
            else:
                st.caption(f"**4. Cible** — {cible} | {len(feats)} features")

        # Étape 5 — Nettoyage
        if etape_max >= 6:
            nett = rapport.get("nettoyage", {})
            parts = []
            if nett.get("valeurs_manquantes"):
                parts.append("val. manquantes traitées")
            dup = nett.get("doublons_supprimes")
            if dup:
                parts.append(f"{dup} doublons supprimés")
            if nett.get("outliers"):
                parts.append("outliers traités")
            ts_interp = nett.get("ts_interpolation")
            if ts_interp:
                m = ts_interp.get("method", "?")
                cols = ts_interp.get("columns", [])
                parts.append(f"interpolation {m} ({len(cols)} col.)")
            # Chercher une coupure TS dans l'historique
            historique = rapport.get("historique", [])
            for entry in historique:
                action = entry.get("action", "")
                if "coupée" in action.lower() or "série coupée" in action.lower():
                    parts.append(action)
                    break
            detail = ", ".join(parts) if parts else "✓"
            st.caption(f"**5. Nettoyage** — {detail}")

        # Étape 6 — Transformation
        if etape_max >= 7:
            nett = rapport.get("nettoyage", {})
            parts = []
            if nett.get("encodage"):
                parts.append("encodage")
            ts_tf = nett.get("ts_transform")
            if ts_tf:
                tf_type = ts_tf.get("type", "?")
                tf_cols = ts_tf.get("columns", [])
                parts.append(f"{tf_type} sur {', '.join(tf_cols[:3])}")
            if nett.get("normalisation"):
                scaled_cols = nett.get("scaled_columns", [])
                parts.append(f"scaling {nett['normalisation']} "
                             f"({len(scaled_cols)} col.)")
            detail = ", ".join(parts) if parts else "✓ (aucune)"
            st.caption(f"**6. Transformation** — {detail}")

        # Étape 7 — Entraînement
        if etape_max >= 8:
            modele = rapport.get("modele", {})
            nom_m = modele.get("nom", "?")
            score_t = modele.get("score_test")
            txt = f"**7. Entraînement** — {nom_m}"
            if score_t is not None:
                txt += f" (test {score_t:.3f})"
            st.caption(txt)

        # Étape 8 — Évaluation
        if etape_max >= 9:
            st.caption("**8. Évaluation** — ✓")

        # Étape 9 — Optimisation
        opt = rapport.get("optimisation")
        if opt:
            best_p = opt.get("best_params", {})
            n_params = len(best_p)
            st.caption(f"**9. Optimisation** — {n_params} hyper-param.")


# ═══════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION
# ═══════════════════════════════════════════════════════════

# Déterminer si on est sur l'accueil
_projet_actif = st.session_state.get("df_courant") is not None
_accueil_action = st.session_state.get("accueil_action")
_on_accueil = (not _projet_actif) and (_accueil_action is None)

step_idx = 0  # valeur par défaut

with st.sidebar:
    # ── Titre ML Studio = retour accueil ──
    if st.button("⚗️ ML STUDIO", use_container_width=True, key="home_btn",
                 type="primary" if _on_accueil else "secondary"):
        # Retour à l'accueil : tout réinitialiser
        keys_to_keep = set()
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()

    if _projet_actif:
        # ── Projet actif : afficher la navigation complète ──
        st.divider()
        _indicateur_statut()
        st.divider()

        # Progression
        max_step = _etape_max_accessible()
        progress_pct = max_step / max(len(STEPS) - 1, 1)
        st.progress(progress_pct, text=f"Progression : {max_step}/{len(STEPS) - 1}")

        # Navigation avec verrouillage
        step_labels = []
        for i, label in enumerate(STEPS):
            if i <= max_step:
                step_labels.append(label)
            else:
                step_labels.append(f"🔒 {label}")

        # Auto-avancement : si une étape a demandé la navigation
        pending = st.session_state.pop("_pending_step", None)
        if pending is not None and pending <= max_step:
            default_idx = pending
            st.session_state.pop("nav_step", None)
        else:
            default_idx = min(max_step, len(STEPS) - 1)

        choix = st.radio("Étape", step_labels, index=default_idx,
                          key="nav_step", label_visibility="collapsed")

        step_idx = step_labels.index(choix)

        if step_idx > max_step:
            step_idx = max_step
            st.warning(f"🔒 Étape verrouillée. Complétez d'abord l'étape {max_step}.")

        st.divider()

        # Aide contextuelle
        afficher_aide(step_idx)

        # Glossaire
        with st.expander("📖 Glossaire ML"):
            afficher_glossaire()

        # Indicateur projet + export ZIP
        rapport = st.session_state.get("rapport")
        if rapport and rapport.get("chemin"):
            st.divider()
            st.caption(f"📁 **Projet :** {rapport.get('nom', '?')}")
            st.caption(f"Étape {rapport.get('etape_courante', 0)}/{len(STEPS) - 1}")
            with st.expander("📂 Emplacement des fichiers"):
                st.code(rapport["chemin"], language=None)
                st.caption("Ce dossier contient : rapport.json, "
                           "data_raw.csv, data_cleaned.csv, modèles (.pkl), etc.")
            zip_bytes = exporter_projet_zip(rapport["chemin"])
            st.download_button("📦 Télécharger le projet (ZIP)",
                               zip_bytes,
                               f"{rapport.get('dossier', 'projet')}.zip",
                               "application/zip",
                               key="dl_zip")
    else:
        # ── Pas de projet actif : sidebar minimale ──
        st.divider()
        st.caption("Créez ou chargez un projet pour commencer le pipeline ML.")


# ═══════════════════════════════════════════════════════════
# PAGE D'ACCUEIL
# ═══════════════════════════════════════════════════════════
def afficher_accueil():
    """Page d'accueil — choix entre nouveau projet ou chargement."""
    st.markdown("""
    <div style="text-align:center; margin: 2rem 0 1rem;">
        <span style="font-size:3rem;">⚗️</span>
        <h1 style="margin:0.2rem 0; font-size:2.2rem; font-weight:800; letter-spacing:-0.03em;">
            ML Studio
        </h1>
        <p style="color:#6B7280; font-size:1rem; margin-top:0.3rem;">
            Pipeline Machine Learning interactif & pédagogique
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="home-card">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">🆕</div>
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.4rem;">Nouveau projet</div>
            <div style="font-size:0.85rem; color:#6B7280; line-height:1.5;">
                Importez vos données et laissez-vous guider pas à pas :
                typage, nettoyage, modélisation, évaluation, prédiction.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🆕  Créer un nouveau projet", type="primary",
                      use_container_width=True, key="btn_nouveau"):
            st.session_state["accueil_action"] = "nouveau"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="home-card">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">📂</div>
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.4rem;">Charger un projet</div>
            <div style="font-size:0.85rem; color:#6B7280; line-height:1.5;">
                Reprenez un projet sauvegardé là où vous l'aviez laissé,
                avec tous vos paramètres et modèles.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📂  Charger un projet existant",
                      use_container_width=True, key="btn_charger"):
            st.session_state["accueil_action"] = "charger"
            st.rerun()


# ═══════════════════════════════════════════════════════════
# ZONE PRINCIPALE — ROUTAGE
# ═══════════════════════════════════════════════════════════
STEP_FUNCTIONS = {
    0: afficher_demarrage,
    1: afficher_typage,
    2: afficher_consolidation,
    3: afficher_diagnostic,
    4: afficher_cible_variables,
    5: afficher_nettoyage,
    6: afficher_transformation,
    7: afficher_entrainement,
    8: afficher_evaluation,
    9: afficher_optimisation_prediction,
}

if _on_accueil:
    afficher_accueil()
elif _accueil_action == "nouveau":
    afficher_demarrage()
elif _accueil_action == "charger":
    afficher_charger_projet()
else:
    handler = STEP_FUNCTIONS.get(step_idx)
    if handler:
        handler()
    else:
        st.error(f"Étape {step_idx} non implémentée.")
