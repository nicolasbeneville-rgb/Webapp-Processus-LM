# -*- coding: utf-8 -*-
"""
app_pipeline.py — Page principale : Pipeline ML (Créer un Modèle).

Contient toute la logique du pipeline : sidebar, navigation par étapes, routage.
Appelé depuis app.py via st.navigation.
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import STEPS

from modules.m1_chargement import afficher_demarrage, afficher_typage, afficher_consolidation, afficher_charger_projet
from modules.m2_diagnostic import afficher_diagnostic, afficher_cible_variables
from modules.m3_nettoyage import afficher_nettoyage, afficher_transformation
from modules.m4_entrainement import afficher_entrainement
from modules.m5_evaluation import afficher_evaluation
from modules.m6_prediction import afficher_optimisation_prediction
from modules.aide_contextuelle import afficher_aide, afficher_glossaire
from utils.projet_manager import exporter_projet_zip, exporter_projet_portable, importer_projet_portable

import os
_IS_CLOUD = os.path.exists("/mount/src")


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
    # ── 1. Titre ML Studio = retour accueil (tout en haut) ──
    if st.button("⚗️ ML STUDIO", use_container_width=True, key="home_btn",
                 type="primary" if _on_accueil else "secondary"):
        keys_to_keep = set()
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()

    if _projet_actif:
        # ── 2. Barre de progression simple ──
        max_step = _etape_max_accessible()
        progress_pct = max_step / max(len(STEPS) - 1, 1)
        st.progress(progress_pct)
        st.caption(f"Progression : **{max_step}/{len(STEPS) - 1}**")

        # ── 3. Sauvegarde (accessible immédiatement) ──
        rapport = st.session_state.get("rapport")
        if rapport:
            project_bytes = exporter_projet_portable(st.session_state)
            nom_safe = rapport.get("nom", "projet").replace(" ", "_").lower()
            st.download_button(
                "💾 Sauvegarder mon projet",
                project_bytes,
                f"{nom_safe}.mlproject",
                "application/octet-stream",
                key="dl_mlproject",
                type="primary",
            )
            if rapport.get("chemin") and not _IS_CLOUD:
                zip_bytes = exporter_projet_zip(rapport["chemin"])
                st.download_button("📦 Exporter en ZIP",
                                   zip_bytes,
                                   f"{rapport.get('dossier', 'projet')}.zip",
                                   "application/zip",
                                   key="dl_zip")

        st.divider()

        # ── 4. Navigation étapes ──
        step_labels = []
        for i, label in enumerate(STEPS):
            if i <= max_step:
                step_labels.append(label)
            else:
                step_labels.append(f"🔒 {label}")

        pending = st.session_state.pop("_pending_step", None)
        if pending is not None and pending <= max_step:
            default_idx = pending
            st.session_state.pop("nav_step", None)
        elif "nav_step" in st.session_state:
            # Garder l'étape actuelle (ne pas sauter à max_step)
            current_choice = st.session_state["nav_step"]
            if current_choice in step_labels:
                default_idx = step_labels.index(current_choice)
            else:
                default_idx = min(max_step, len(STEPS) - 1)
        else:
            default_idx = min(max_step, len(STEPS) - 1)

        choix = st.radio("Étape", step_labels, index=default_idx,
                          key="nav_step", label_visibility="collapsed")

        step_idx = step_labels.index(choix)

        if step_idx > max_step:
            step_idx = max_step
            st.warning(f"🔒 Étape verrouillée. Complétez d'abord l'étape {max_step}.")

        # ── 5. Sous-étapes (si étape 5 ou 6) ──
        _sub_step_labels = None
        problem_type = st.session_state.get("problem_type", "Régression")
        is_ts = problem_type == "Série temporelle" or st.session_state.get("ts_horizon_mode")

        if step_idx == 5:
            if is_ts:
                _sub_step_labels = [
                    "👯 Doublons de date",
                    "📊 Continuité & Gaps",
                    "🔧 Interpolation",
                    "🔬 Valeurs aberrantes",
                    "✅ Validation",
                ]
            else:
                _sub_step_labels = [
                    "🕳️ Valeurs manquantes",
                    "👯 Doublons",
                    "📊 Outliers",
                ]
            sub_choice = st.radio(
                "Sous-étape", _sub_step_labels,
                key="nav_sub_step_5",
                label_visibility="collapsed")
            st.session_state["_current_sub_step"] = _sub_step_labels.index(sub_choice)

        elif step_idx == 6:
            if is_ts:
                _sub_step_labels = [
                    "📊 Analyse & Recommandations",
                    "🔧 Transformations",
                    "📐 Scaling",
                    "🎯 Prédiction horizon",
                    "✅ Valider",
                ]
            else:
                _sub_step_labels = [
                    "🏷️ Encoding",
                    "📐 Scaling",
                    "🔧 Feature Engineering",
                ]
            sub_choice = st.radio(
                "Sous-étape", _sub_step_labels,
                key="nav_sub_step_6",
                label_visibility="collapsed")
            st.session_state["_current_sub_step"] = _sub_step_labels.index(sub_choice)
        else:
            st.session_state.pop("_current_sub_step", None)

        st.divider()

        # ── 6. Aide contextuelle + Glossaire ──
        afficher_aide(step_idx)

        with st.expander("📖 Glossaire ML"):
            afficher_glossaire()

        # ── 7. Récap par étape (tout en bas) ──
        _indicateur_statut()

    else:
        st.divider()
        st.caption("Créez ou chargez un projet pour commencer le pipeline ML.")


# ═══════════════════════════════════════════════════════════
# PAGE D'ACCUEIL
# ═══════════════════════════════════════════════════════════
def afficher_accueil():
    """Page d'accueil — choix entre nouveau projet ou chargement."""
    if _IS_CLOUD:
        st.info(
            "☁️ **Version en ligne** — Vos données ne sont pas conservées sur le serveur. "
            "Pensez à **sauvegarder votre projet** (bouton dans la barre latérale) "
            "pour pouvoir le reprendre plus tard."
        )
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

        # Upload .mlproject (fonctionne partout)
        uploaded_project = st.file_uploader(
            "Charger un fichier .mlproject",
            type=["mlproject"],
            key="upload_mlproject",
            label_visibility="collapsed",
        )
        if uploaded_project is not None:
            try:
                nom = importer_projet_portable(uploaded_project.read(), st.session_state)
                st.session_state["accueil_action"] = None
                st.session_state["_reload_success"] = True
                st.session_state["_reload_success_msg"] = f"✅ Projet « {nom} » rechargé !"
                st.rerun()
            except Exception as e:
                st.error(f"❌ Impossible de charger le projet : {e}")

        # Charger depuis le disque local (uniquement en local)
        if not _IS_CLOUD:
            if st.button("📂  Charger un projet local",
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
