# -*- coding: utf-8 -*-
"""
m1_chargement.py — Module 1 : Chargement des fichiers et configuration projet.

Étapes 0 (Démarrage), 1 (Typage), 2 (Consolidation).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    SEPARATORS, ENCODINGS, MAX_FILES, SUPPORTED_FILE_TYPES,
    TARGET_TYPES, JOIN_TYPES, AGGREGATION_FUNCTIONS,
    PROBLEM_TYPES, MAX_NAN_AFTER_CONVERSION_PCT, MAX_JOIN_LOSS_PCT,
)
from src.data_loader import load_file, get_file_info, detect_types, apply_typing
from src.consolidation import preview_join, perform_join, get_join_stats, aggregate
from src.validators import validate_loaded_file, validate_after_conversion, validate_join
from utils.projet_manager import (
    creer_projet, lister_projets, supprimer_projet, charger_rapport,
    sauvegarder_rapport, sauvegarder_csv, ajouter_historique,
    charger_modele, charger_objet, charger_csv,
)

CHART_COLORS = {
    "primary": "#4F5BD5", "secondary": "#818CF8", "accent": "#A5B4FC",
    "success": "#059669", "warning": "#D97706", "danger": "#DC2626",
    "palette": ["#4F5BD5", "#059669", "#D97706", "#DC2626", "#0891B2",
                "#7C3AED", "#EA580C", "#2563EB", "#14B8A6", "#E11D48"],
}


def _restaurer_projet(projet: dict):
    """Restaure complètement un projet sauvegardé dans session_state."""
    import os

    chemin = projet.get("chemin", "")
    etape = projet.get("etape_courante", 0)

    st.session_state["rapport"] = projet
    st.session_state["projet_charge"] = True
    st.session_state["etape_courante"] = etape

    # ── Restaurer le type de problème ──
    if projet.get("type_ml"):
        st.session_state["problem_type"] = projet["type_ml"]
    if projet.get("colonne_cible"):
        st.session_state["target_col"] = projet["colonne_cible"]
    if projet.get("colonnes_features"):
        st.session_state["feature_cols"] = projet["colonnes_features"]
    if projet.get("ts_datetime_col"):
        st.session_state["ts_datetime_col"] = projet["ts_datetime_col"]
    if projet.get("parcours") == "ts" and projet.get("colonne_cible"):
        st.session_state["ts_value_col"] = projet["colonne_cible"]

    # ── Restaurer le DataFrame (le plus avancé disponible) ──
    df = None
    cleaned_path = os.path.join(chemin, "data_cleaned.csv")
    typed_path = os.path.join(chemin, "data_typed.csv")
    raw_path = os.path.join(chemin, "data_raw.csv")
    if os.path.isfile(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.isfile(typed_path):
        df = pd.read_csv(typed_path)
    elif os.path.isfile(raw_path):
        df = pd.read_csv(raw_path)

    if df is not None:
        # Ré-appliquer les conversions de types sauvegardées (sinon CSV perd les datetime etc.)
        type_mapping = projet.get("type_mapping", {})
        if type_mapping:
            from src.data_loader import apply_typing
            try:
                df = apply_typing(df, type_mapping)
            except Exception:
                pass  # en cas de colonne absente ou conversion impossible

        st.session_state["df_courant"] = df
        # Recréer raw_dataframes / typed_dataframes pour que les étapes 1-2
        # puissent afficher le fichier même après un rechargement.
        nom_fichier = projet.get("nom", "data") + ".csv"
        st.session_state["raw_dataframes"] = {nom_fichier: df.copy()}
        if etape >= 2:
            st.session_state["typed_dataframes"] = {nom_fichier: df.copy()}

    # ── Restaurer les drapeaux de progression selon etape_courante ──
    # Mapping : etape N terminée ⟹ tous les flags ≤ N sont True
    flags_par_etape = {
        1: ["chargement_done"],
        2: ["chargement_done", "typage_done"],
        3: ["chargement_done", "typage_done", "consolidation_done"],
        4: ["chargement_done", "typage_done", "consolidation_done",
            "diagnostic_done"],
        5: ["chargement_done", "typage_done", "consolidation_done",
            "diagnostic_done", "cible_done"],
        6: ["chargement_done", "typage_done", "consolidation_done",
            "diagnostic_done", "cible_done",
            "nettoyage_done", "manquantes_done", "doublons_done", "outliers_done"],
        7: ["chargement_done", "typage_done", "consolidation_done",
            "diagnostic_done", "cible_done",
            "nettoyage_done", "manquantes_done", "doublons_done", "outliers_done",
            "encoding_done", "scaling_done", "transformation_done"],
        8: ["chargement_done", "typage_done", "consolidation_done",
            "diagnostic_done", "cible_done",
            "nettoyage_done", "manquantes_done", "doublons_done", "outliers_done",
            "encoding_done", "scaling_done", "transformation_done",
            "entrainement_done"],
        9: ["chargement_done", "typage_done", "consolidation_done",
            "diagnostic_done", "cible_done",
            "nettoyage_done", "manquantes_done", "doublons_done", "outliers_done",
            "encoding_done", "scaling_done", "transformation_done",
            "entrainement_done", "evaluation_done"],
    }

    flags = flags_par_etape.get(etape, [])
    for flag in flags:
        st.session_state[flag] = True

    # ── Restaurer le modèle si disponible ──
    modele_info = projet.get("modele")
    if modele_info and chemin:
        nom_modele = modele_info.get("nom", "")
        model = charger_modele(chemin, f"{nom_modele}.joblib")
        if model is not None:
            # Nom d'affichage : remettre un nom lisible
            display_name = nom_modele.replace("_optimise", " (optimisé)")
            st.session_state["meilleur_modele"] = {
                "name": display_name,
                "model": model,
                "test_score": modele_info.get("score_test"),
                "train_score": modele_info.get("score_train"),
                "best_params": modele_info.get("best_params"),
            }

            # Restaurer opt_result si le modèle a été optimisé
            opt_info = projet.get("optimisation")
            if opt_info:
                st.session_state["opt_result"] = {
                    "best_params": opt_info.get("best_params"),
                    "best_score": opt_info.get("best_score_cv"),
                    "best_model": model,
                }

    # ── Restaurer les splits train/test si disponibles ──
    X_train_df = charger_csv(chemin, "X_train.csv")
    X_test_df = charger_csv(chemin, "X_test.csv")
    y_train_df = charger_csv(chemin, "y_train.csv")
    y_test_df = charger_csv(chemin, "y_test.csv")

    if X_train_df is not None and y_train_df is not None:
        st.session_state["X_train"] = X_train_df.values
        st.session_state["X_test"] = X_test_df.values
        st.session_state["y_train"] = y_train_df["target"].values
        st.session_state["y_test"] = y_test_df["target"].values
        # Restaurer les noms de features
        st.session_state["feature_cols_used"] = X_train_df.columns.tolist()

    # Restaurer colonnes_features_used depuis rapport
    feat_used = projet.get("colonnes_features_used")
    if feat_used and "feature_cols_used" not in st.session_state:
        st.session_state["feature_cols_used"] = feat_used

    # ── Restaurer les objets de preprocessing si disponibles ──
    encoders = charger_objet(chemin, "encoders.pkl")
    if encoders is not None:
        st.session_state["encoders"] = encoders

    scaler = charger_objet(chemin, "scaler.pkl")
    if scaler is not None:
        st.session_state["scaler"] = scaler

    # Restaurer la liste des colonnes scalées depuis le rapport
    nettoyage = projet.get("nettoyage", {})
    if nettoyage.get("scaled_columns"):
        st.session_state["scaled_columns"] = nettoyage["scaled_columns"]

    # ── Reconstruire la série temporelle si parcours TS ──
    if projet.get("parcours") == "ts" and df is not None:
        dt_col = projet.get("ts_datetime_col")
        val_col = projet.get("colonne_cible")
        if dt_col and val_col and dt_col in df.columns and val_col in df.columns:
            try:
                from src.timeseries import prepare_timeseries
                ts_series = prepare_timeseries(df, dt_col, val_col)
                st.session_state["ts_series"] = ts_series
            except Exception:
                pass


def afficher_charger_projet():
    """Page de chargement d'un projet existant."""
    projets = lister_projets()
    if not projets:
        st.info("📭 Aucun projet sauvegardé.")
        if st.button("⬅️ Retour à l'accueil"):
            st.session_state.pop("accueil_action", None)
            st.rerun()
        return

    st.subheader("📂 Vos projets sauvegardés")

    noms = [f"{p['nom']} (étape {p.get('etape_courante', 0)})" for p in projets]
    idx = st.selectbox("Sélectionnez un projet", range(len(noms)),
                       format_func=lambda i: noms[i],
                       help="Choisissez le projet sauvegardé à reprendre")
    projet = projets[idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Étape", f"{projet.get('etape_courante', 0)}/9")
    c2.metric("Créé le", projet.get("date_creation", "—"))
    c3.metric("Type", projet.get("type_ml", "—") or "—")

    # Afficher le message de succès après rerun si le flag est posé
    if st.session_state.pop("_reload_success", None):
        st.success(st.session_state.pop("_reload_success_msg", "✅ Projet rechargé !"))

    col_load, col_del = st.columns([3, 1])
    with col_load:
        if st.button("🔄 Recharger ce projet", type="primary"):
            _restaurer_projet(projet)
            st.session_state.pop("accueil_action", None)
            st.session_state["_reload_success"] = True
            st.session_state["_reload_success_msg"] = f"✅ Projet « {projet['nom']} » rechargé !"
            st.rerun()
    with col_del:
        if st.button("🗑️ Supprimer"):
            st.session_state["_confirm_delete"] = projet.get("chemin")

    if st.session_state.get("_confirm_delete") == projet.get("chemin"):
        st.warning("⚠️ Suppression irréversible. Confirmer ?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Oui, supprimer", type="primary"):
                supprimer_projet(projet["chemin"])
                st.session_state.pop("_confirm_delete", None)
                st.success("Projet supprimé.")
                st.rerun()
        with c2:
            if st.button("❌ Annuler"):
                st.session_state.pop("_confirm_delete", None)
                st.rerun()

    st.divider()
    if st.button("⬅️ Retour à l'accueil"):
        st.session_state.pop("accueil_action", None)
        st.rerun()


def afficher_demarrage():
    """Étape 0 — Configuration et création d'un nouveau projet."""
    if st.button("⬅️ Retour à l'accueil"):
        st.session_state.pop("accueil_action", None)
        st.rerun()
    st.subheader("⚙️ Configuration du projet")

    # ── Arbre de décision pédagogique ──
    with st.expander("🎓 Guide : comment choisir le bon type de projet ?", expanded=False):
        st.markdown("""
**Posez-vous cette question : que voulez-vous prédire ?**

```
                    Que voulez-vous prédire ?
                            │
             ┌──────────────┼──────────────┐
             │              │              │
        Un nombre ?    Une catégorie ?  Une évolution
        (prix, durée,  (oui/non,       dans le temps ?
         température)   type, classe)   (niveau, ventes)
             │              │              │
         RÉGRESSION   CLASSIFICATION  SÉRIE TEMPORELLE
             │              │              │
          Exemples :     Exemples :     Exemples :
          · prix d'un    · risque de    · niveau d'un
            appartement    défaut (O/N)   barrage à J+15
          · durée d'un   · espèce d'un  · ventes du
            trajet          animal          mois prochain
          · note d'un    · type de      · température
            examen         client          demain
```

**📐 Régression** → le résultat est un **nombre continu** (1.5, 42, 1203.7…)

**🏷️ Classification** → le résultat est une **étiquette** parmi un ensemble fini ("chat"/"chien", "spam"/"ok"…)

**📈 Série temporelle** → vous avez des données **datées** et voulez **prédire la suite** :
- *ARIMA* : basé uniquement sur l'historique de la variable (simple, univarié)
- *Prédiction horizon* : utilise d'autres variables (météo, débits…) pour prédire à N jours

> **💡 Pas sûr ?** Si votre cible est un nombre → **Régression**. Si c'est du texte ou des catégories → **Classification**.
> Si vous avez une colonne de dates et voulez prédire le futur → **Série temporelle**.
""")

    col1, col2 = st.columns(2)
    with col1:
        nom_projet = st.text_input("📝 Nom du projet",
                                    value=st.session_state.get("nom_projet", ""))

        problem_help = {
            "Régression": "📐 Prédire un **nombre** (prix, durée, température…)",
            "Classification": "🏷️ Prédire une **catégorie** (oui/non, type…)",
            "Série temporelle": "📈 **Prédire l'évolution** dans le temps",
        }
        problem_type = st.radio("Que voulez-vous prédire ?", PROBLEM_TYPES)
        st.caption(problem_help.get(problem_type, ""))

    with col2:
        st.markdown("**Détection automatique**")
        st.caption("Le séparateur et l'encodage seront détectés automatiquement.")

    st.divider()
    st.subheader("📂 Chargement des fichiers")

    uploaded_files = st.file_uploader(
        "📎 Glissez vos fichiers (CSV ou Excel, max 3)",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    if len(uploaded_files) > MAX_FILES:
        st.error(f"❌ Maximum {MAX_FILES} fichiers autorisés.")
        return

    dataframes = {}
    all_valid = True

    for i, f in enumerate(uploaded_files):
        st.subheader(f"📄 Fichier {i + 1} : {f.name}")

        with st.expander("⚙️ Options de lecture", expanded=(i == 0)):
            c1, c2, c3 = st.columns(3)
            with c1:
                sep_label = st.selectbox("Séparateur", list(SEPARATORS.keys()),
                                         key=f"sep_{i}",
                                         help="Caractère séparant les colonnes (virgule, point-virgule, tabulation…)")
            with c2:
                enc = st.selectbox("Encodage", ENCODINGS, key=f"enc_{i}",
                                   help="Encodage du fichier (utf-8 par défaut, latin-1 pour les fichiers français anciens)")
            with c3:
                header = st.number_input("Ligne en-têtes", 0, 10, 0, key=f"header_{i}",
                                         help="Numéro de la ligne contenant les noms de colonnes (0 = première ligne)")

        try:
            df = load_file(f, separator=SEPARATORS[sep_label], encoding=enc,
                           header_row=header)
            info = get_file_info(df)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Lignes", f"{info['nb_rows']:,}")
            mc2.metric("Colonnes", info["nb_cols"])
            mc3.metric("Mémoire", f"{info['memory_mb']} Mo")

            st.markdown("**Aperçu (5 lignes) :**")
            st.dataframe(df.head(), use_container_width=True)

            v = validate_loaded_file(df, f.name)
            if v["passed"]:
                st.success(v["message"])
            else:
                st.error(v["message"])
                all_valid = False

            dataframes[f.name] = df

        except Exception as e:
            st.error(f"❌ Impossible de lire « {f.name} » : {e}")
            all_valid = False

    if dataframes and all_valid:
        # ── Garde-fous : vérifier la compatibilité colonnes / type de problème ──
        merged_df = list(dataframes.values())[0] if len(dataframes) == 1 else None
        if merged_df is not None:
            n_num = len(merged_df.select_dtypes(include="number").columns)
            n_total = len(merged_df.columns)

            if problem_type in ("Régression", "Classification") and n_total < 3:
                st.warning(
                    f"⚠️ Seulement **{n_total} colonne(s)** détectée(s). "
                    f"En {problem_type} il faut au minimum **1 cible + 1 variable "
                    f"explicative**. Vérifiez votre fichier.")

            if problem_type == "Série temporelle" and n_num == 0:
                st.error("❌ Aucune colonne numérique détectée. "
                         "Une série temporelle nécessite au moins **1 colonne "
                         "numérique** (valeur à prédire).")
                all_valid = False

            if problem_type == "Série temporelle" and n_num == 1:
                st.info(
                    "💡 **1 seule colonne numérique** → parcours **ARIMA** (univarié). "
                    "Pour exploiter d'autres variables (météo, débits…), "
                    "ajoutez-les au fichier et utilisez le mode **Prédiction horizon**.")

        if st.button("🚀 Démarrer le projet", type="primary"):
            rapport = creer_projet(nom_projet or "Mon_projet")
            rapport["type_ml"] = problem_type
            rapport["parcours"] = "ts" if problem_type == "Série temporelle" else "ml_classique"
            rapport["etape_courante"] = 1
            ajouter_historique(rapport, f"Projet créé — {len(dataframes)} fichier(s)")

            st.session_state["rapport"] = rapport
            st.session_state["problem_type"] = problem_type
            st.session_state["raw_dataframes"] = dataframes
            st.session_state["chargement_done"] = True
            st.session_state["projet_charge"] = True
            st.session_state["etape_courante"] = 1
            st.session_state.pop("accueil_action", None)

            # Sauvegarder les fichiers bruts
            if len(dataframes) == 1:
                first_df = list(dataframes.values())[0]
                sauvegarder_csv(rapport, first_df, "data_raw.csv")
                st.session_state["df_courant"] = first_df.copy()

            sauvegarder_rapport(rapport)
            st.session_state["_pending_step"] = 1
            st.rerun()


def afficher_typage():
    """Étape 1 — Typage et conversion des colonnes."""
    st.caption("ÉTAPE 1")

    with st.expander("🎓 Pourquoi cette étape ?", expanded=False):
        st.markdown("""
L'ordinateur doit savoir si chaque colonne contient des **nombres**, du **texte**,
des **dates** ou des **catégories**.

| Ce que vous voyez | Ce que l'ordi comprend | Problème si mal typé |
|---|---|---|
| `1 200 €` | Texte | Impossible de calculer une moyenne |
| `75001` | Nombre | Le code postal est une catégorie, pas un nombre ! |
| `12/03/2024` | Texte | Ne peut pas trier par date |

> **💡 L'app détecte automatiquement les types.** Vérifiez simplement que les suggestions
> sont correctes et corrigez manuellement si nécessaire.
""")

    dataframes = st.session_state.get("raw_dataframes", {})
    if not dataframes:
        st.warning("⚠️ Chargez d'abord des fichiers à l'étape 0.")
        return

    for fname, df in dataframes.items():
        st.subheader(f"📄 {fname}")

        # ── Suppression de colonnes ──
        with st.expander("🗑️ Supprimer des colonnes inutiles", expanded=False):
            cols_to_drop = st.multiselect(
                "Colonnes à supprimer", df.columns.tolist(),
                key=f"drop_cols_{fname}",
                help="Sélectionnez les colonnes que vous ne voulez pas garder "
                     "(identifiants, colonnes vides, doublons…)")
            if cols_to_drop and st.button(f"🗑️ Supprimer {len(cols_to_drop)} colonne(s)",
                                          key=f"apply_drop_{fname}"):
                df = df.drop(columns=cols_to_drop)
                dataframes[fname] = df
                st.session_state["raw_dataframes"] = dataframes
                st.success(f"✅ {len(cols_to_drop)} colonne(s) supprimée(s). "
                           f"Restant : {len(df.columns)} colonnes.")
                st.rerun()

        detected = detect_types(df)

        key_types = f"type_choices_{fname}"
        if key_types not in st.session_state:
            st.session_state[key_types] = detected.copy()

        type_choices = {}

        for col_name in df.columns:
            detected_type = detected.get(col_name, "Texte (string)")
            with st.expander(f"📌 **{col_name}** — détecté : *{detected_type}*"):
                c1, c2 = st.columns([1, 1])
                with c1:
                    sample = df[col_name].dropna().head(8).tolist()
                    st.markdown(f"**Exemples :** {', '.join(str(v) for v in sample)}")
                    st.markdown(f"**Uniques :** {df[col_name].nunique()} | "
                                f"**Trous :** {df[col_name].isna().sum()}")

                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        fig, ax = plt.subplots(figsize=(3, 1.2))
                        fig.set_facecolor("#F8F9FC")
                        df[col_name].dropna().hist(ax=ax, bins=20,
                                                    color=CHART_COLORS["primary"],
                                                    edgecolor="white", alpha=0.85)
                        ax.set_facecolor("#FFFFFF")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.set_title("Distribution", fontsize=9, fontweight="bold")
                        ax.tick_params(labelsize=7)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                with c2:
                    chosen = st.selectbox(
                        f"Type pour « {col_name} »", TARGET_TYPES,
                        index=TARGET_TYPES.index(detected_type) if detected_type in TARGET_TYPES else 2,
                        key=f"type_{fname}_{col_name}")
                type_choices[col_name] = chosen

        st.session_state[key_types] = type_choices

    if st.button("🔄 Appliquer les conversions", type="primary"):
        converted_dfs = {}
        for fname, df in dataframes.items():
            type_choices = st.session_state[f"type_choices_{fname}"]
            try:
                df_after = apply_typing(df, type_choices)
                converted_dfs[fname] = df_after
                v = validate_after_conversion(df, df_after,
                                              max_nan_pct=MAX_NAN_AFTER_CONVERSION_PCT)
                if v["passed"]:
                    st.success(f"✅ {fname} — OK")
                else:
                    st.warning(f"⚠️ {fname} — {v['message']}")
            except Exception as e:
                st.error(f"❌ {fname} : {e}")
                converted_dfs[fname] = df

        st.session_state["typed_dataframes"] = converted_dfs
        st.session_state["typage_done"] = True

        if len(converted_dfs) == 1:
            first_df = list(converted_dfs.values())[0]
            st.session_state["df_courant"] = first_df.copy()
            st.session_state["consolidation_done"] = True

        rapport = st.session_state.get("rapport", {})
        if rapport:
            # Sauvegarder le mapping de types pour pouvoir le ré-appliquer au rechargement
            all_type_choices = {}
            for fname in dataframes:
                tc = st.session_state.get(f"type_choices_{fname}", {})
                all_type_choices.update(tc)
            rapport["type_mapping"] = all_type_choices

            # Sauvegarder le DataFrame typé pour qu'il soit disponible au rechargement
            df_typed = list(converted_dfs.values())[0] if len(converted_dfs) == 1 else None
            if df_typed is not None:
                sauvegarder_csv(rapport, df_typed, "data_typed.csv")

            rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 2)
            ajouter_historique(rapport, "Types vérifiés")
            sauvegarder_rapport(rapport)

        if st.session_state.get("consolidation_done"):
            st.session_state["_pending_step"] = 3
        else:
            st.session_state["_pending_step"] = 2
        st.rerun()


def afficher_consolidation():
    """Étape 2 — Consolidation / jointure de fichiers."""
    st.caption("ÉTAPE 2")

    dataframes = st.session_state.get("typed_dataframes",
                                       st.session_state.get("raw_dataframes", {}))
    if not dataframes:
        st.warning("⚠️ Complétez d'abord le typage (étape 1).")
        return

    if len(dataframes) == 1:
        fname = list(dataframes.keys())[0]
        st.session_state["df_courant"] = dataframes[fname].copy()
        st.session_state["consolidation_done"] = True
        st.info("📄 Un seul fichier — pas besoin de fusion.")
        st.dataframe(st.session_state["df_courant"].head(), use_container_width=True)
        st.success("✅ Prêt pour la suite.")
        return

    # Afficher les fichiers
    for fname, df in dataframes.items():
        st.write(f"**{fname}** — {len(df)} lignes × {len(df.columns)} colonnes")

    st.divider()

    # Agrégation optionnelle
    with st.expander("📊 Regrouper des lignes avant fusion", expanded=False):
        agg_file = st.selectbox("Fichier", list(dataframes.keys()), key="agg_file",
                                help="Fichier dont vous souhaitez regrouper les lignes")
        agg_df = dataframes[agg_file]
        agg_group = st.selectbox("Colonne de regroupement",
                                  agg_df.columns.tolist(), key="agg_group",
                                  help="Colonne servant de clé pour le regroupement")
        agg_cols = st.multiselect("Colonnes à résumer",
                                   [c for c in agg_df.columns if c != agg_group],
                                   key="agg_cols",
                                   help="Colonnes numériques à agréger avec la fonction choisie")
        agg_func = st.selectbox("Fonction", AGGREGATION_FUNCTIONS, key="agg_func",
                                help="Opération d'agrégation : somme, moyenne, min, max…")

        if agg_cols and st.button("Appliquer le regroupement"):
            try:
                result = aggregate(agg_df, agg_group, {c: agg_func for c in agg_cols})
                dataframes[agg_file] = result
                st.session_state["typed_dataframes"] = dataframes
                st.success(f"✅ {len(result)} lignes après regroupement.")
                st.dataframe(result.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()
    st.subheader("🔗 Fusionner deux fichiers")

    file_names = list(dataframes.keys())
    c1, c2 = st.columns(2)
    with c1:
        left_file = st.selectbox("1er fichier", file_names, key="join_left",
                                 help="Premier fichier à fusionner")
        left_key = st.selectbox("Colonne commune",
                                 dataframes[left_file].columns.tolist(), key="join_left_key",
                                 help="Colonne clé du 1er fichier pour l'appariement")
    with c2:
        right_file = st.selectbox("2ème fichier",
                                   [f for f in file_names if f != left_file], key="join_right",
                                   help="Deuxième fichier à fusionner")
        right_key = st.selectbox("Colonne commune",
                                  dataframes[right_file].columns.tolist(), key="join_right_key",
                                  help="Colonne clé du 2ème fichier pour l'appariement")

    join_type = st.selectbox("Type de fusion", list(JOIN_TYPES.keys()),
                             help="Interne = lignes communes, Gauche = tout du 1er, Externe = tous")

    if st.button("🔍 Prévisualiser"):
        preview = preview_join(dataframes[left_file], dataframes[right_file],
                               left_key, right_key, how=join_type.lower())
        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes estimées", preview["estimated_rows"])
        c2.metric("Clés communes", preview["common_keys"])
        c3.metric("Colonnes", preview["total_cols"])

    if st.button("▶️ Fusionner", type="primary"):
        try:
            result = perform_join(dataframes[left_file], dataframes[right_file],
                                  left_key, right_key, how=join_type.lower())
            stats = get_join_stats(dataframes[left_file], result)

            c1, c2, c3 = st.columns(3)
            c1.metric("Avant", stats["rows_before"])
            c2.metric("Après", stats["rows_after"])
            c3.metric("Perdues", stats.get("rows_lost", 0))

            st.dataframe(result.head(), use_container_width=True)

            st.session_state["df_courant"] = result
            st.session_state["consolidation_done"] = True

            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 3)
                ajouter_historique(rapport, "Fusion réalisée")
                sauvegarder_rapport(rapport)
                sauvegarder_csv(rapport, result, "data_raw.csv")

            st.success("✅ Fusion réussie !")
        except Exception as e:
            st.error(f"❌ {e}")
