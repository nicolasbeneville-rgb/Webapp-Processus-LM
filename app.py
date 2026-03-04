# -*- coding: utf-8 -*-
"""
app.py — Point d'entrée Streamlit du Pipeline ML interactif.

Lance l'application avec :
    streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config import (
    APP_TITLE, APP_ICON, APP_LAYOUT, STEPS, PROBLEM_TYPES,
    SEPARATORS, ENCODINGS, MAX_FILES, SUPPORTED_FILE_TYPES,
    TARGET_TYPES, JOIN_TYPES, AGGREGATION_FUNCTIONS,
    REGRESSION_MODELS, CLASSIFICATION_MODELS,
    DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_CV_FOLDS,
    DEFAULT_MIN_SCORE, DEFAULT_MAX_OVERFIT_PCT,
    MAX_NAN_AFTER_CONVERSION_PCT, MAX_JOIN_LOSS_PCT,
    DEFAULT_QUALITY_THRESHOLD, CORRELATION_THRESHOLD,
    MAX_ONEHOT_CARDINALITY, MIN_ROWS_AFTER_CLEANING, MIN_FEATURES,
    DATA_OUTPUT_DIR, MODELS_DIR,
)

from src.data_loader import load_file, get_file_info, detect_types, apply_typing
from src.consolidation import preview_join, perform_join, get_join_stats, aggregate
from src.audit import (
    quality_table, descriptive_stats, detect_anomalies, get_anomaly_actions,
    correlation_matrix, high_correlations, top_correlations_with_target,
    plot_correlation_heatmap, compute_quality_score, check_target_imbalance,
)
from src.preprocessing import (
    handle_missing, detect_outliers_iqr, handle_outliers,
    normalize_columns, encode_categorical,
    get_categorical_columns, get_numeric_columns,
)
from src.feature_engineering import (
    combine_columns, transform_column, discretize_column,
    rename_column, drop_column, auto_select_features, get_modification_summary,
)
from src.models import (
    get_model, train_model, train_multiple, split_data,
    optimize_model, save_model,
)
from src.evaluation import (
    results_table, plot_real_vs_pred, plot_residuals,
    plot_residual_distribution, get_top_errors, auto_comment_residuals,
    plot_confusion_matrix, plot_roc_curve, get_classification_report,
    get_misclassified, plot_feature_importance,
    plot_histogram, plot_boxplot, plot_scatter,
    plot_target_distribution, auto_comment_distribution,
    fig_to_png_bytes, generate_html_report,
)
from src.validators import (
    validate_loaded_file, validate_after_conversion,
    validate_join, validate_data_quality, validate_prepared_data,
    validate_model_scores, validate_optimization,
    validate_residuals, validation_dashboard,
)


# ═══════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)


def _check_step(required_key: str, step_label: str) -> bool:
    """Vérifie qu'une étape précédente a été complétée."""
    if required_key not in st.session_state or st.session_state[required_key] is None:
        st.warning(f"⚠️ Veuillez d'abord compléter l'étape **{step_label}**.")
        return False
    return True


# ═══════════════════════════════════════════════════════════
# BARRE LATÉRALE — NAVIGATION
# ═══════════════════════════════════════════════════════════
st.sidebar.title(APP_TITLE)
current_step = st.sidebar.radio("Étape", STEPS, key="nav_step")
step_idx = STEPS.index(current_step)
st.sidebar.progress((step_idx + 1) / len(STEPS))
st.sidebar.caption(f"Étape {step_idx + 1} / {len(STEPS)}")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 0 — CONFIGURATION DU PROJET
# ═══════════════════════════════════════════════════════════
if current_step == STEPS[0]:
    st.header("⚙️ Configuration du projet")
    st.markdown("Définissez les paramètres généraux de votre projet avant de commencer.")

    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Nom du projet", value=st.session_state.get("project_name", "Mon projet ML"))
        problem_type = st.radio("Type de problème", PROBLEM_TYPES,
                                index=PROBLEM_TYPES.index(st.session_state.get("problem_type", "Régression")))

    with col2:
        max_missing_pct = st.slider(
            "% max de valeurs manquantes par colonne",
            0, 100, st.session_state.get("max_missing_pct", 20), 5
        )
        min_score = st.slider(
            "Score minimum attendu (R² ou Accuracy)",
            0.0, 1.0, st.session_state.get("min_score", DEFAULT_MIN_SCORE), 0.05
        )
        max_overfit_pct = st.slider(
            "% max d'écart train/test (overfitting)",
            0, 50, st.session_state.get("max_overfit_pct", int(DEFAULT_MAX_OVERFIT_PCT)), 1
        )

    if st.button("🚀 Démarrer le projet", type="primary"):
        st.session_state["project_name"] = project_name
        st.session_state["problem_type"] = problem_type
        st.session_state["max_missing_pct"] = max_missing_pct
        st.session_state["min_score"] = min_score
        st.session_state["max_overfit_pct"] = max_overfit_pct
        st.session_state["project_configured"] = True
        st.success("✅ Projet configuré !")

    if st.session_state.get("project_configured"):
        st.divider()
        st.subheader("Récapitulatif")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Projet", st.session_state.get("project_name", ""))
        c2.metric("Type", st.session_state.get("problem_type", ""))
        c3.metric("Score min", st.session_state.get("min_score", 0))
        c4.metric("Max overfit", f"{st.session_state.get('max_overfit_pct', 0)}%")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 1 — CHARGEMENT DES FICHIERS
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[1]:
    st.header("📂 Chargement des fichiers")
    st.markdown("Importez vos fichiers de données (CSV ou Excel). Vous pouvez charger jusqu'à 3 fichiers.")

    if not _check_step("project_configured", STEPS[0]):
        st.stop()

    uploaded_files = st.file_uploader(
        "Glissez-déposez vos fichiers ici",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"❌ Maximum {MAX_FILES} fichiers autorisés.")
            st.stop()

        dataframes = {}
        all_valid = True

        for i, f in enumerate(uploaded_files):
            st.subheader(f"Fichier {i+1} : {f.name}")
            c1, c2, c3 = st.columns(3)
            with c1:
                sep_label = st.selectbox(f"Séparateur ({f.name})", list(SEPARATORS.keys()), key=f"sep_{i}")
            with c2:
                enc = st.selectbox(f"Encodage ({f.name})", ENCODINGS, key=f"enc_{i}")
            with c3:
                header = st.number_input(f"Ligne d'en-tête ({f.name})", 0, 10, 0, key=f"header_{i}")

            try:
                df = load_file(f, separator=SEPARATORS[sep_label], encoding=enc, header_row=header)
                info = get_file_info(df)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Lignes", info["nb_rows"])
                mc2.metric("Colonnes", info["nb_cols"])
                mc3.metric("Mémoire", f"{info['memory_mb']} Mo")

                st.dataframe(df.head(), use_container_width=True)

                # Validation
                v = validate_loaded_file(df, f.name,
                                         min_rows=st.session_state.get("min_rows", 50))
                if v["passed"]:
                    st.success(v["message"])
                else:
                    st.error(v["message"])
                    for d in v["details"]:
                        st.caption(d)
                    all_valid = False

                dataframes[f.name] = df

            except Exception as e:
                st.error(f"❌ Erreur lors du chargement de « {f.name} » : {e}")
                all_valid = False

        if dataframes and all_valid:
            if st.button("✅ Valider les fichiers chargés", type="primary"):
                st.session_state["raw_dataframes"] = dataframes
                st.session_state["files_loaded"] = True
                st.success(f"✅ {len(dataframes)} fichier(s) chargé(s) avec succès !")
                st.rerun()
        elif dataframes and not all_valid:
            st.warning("⚠️ Corrigez les erreurs ci-dessus avant de continuer.")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 2 — TYPAGE ET CONVERSION
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[2]:
    st.header("🔤 Typage et conversion des variables")
    st.markdown("Vérifiez et corrigez le type de chaque colonne. L'application propose des conversions automatiques.")

    if not _check_step("files_loaded", STEPS[1]):
        st.stop()

    dataframes = st.session_state["raw_dataframes"]

    for fname, df in dataframes.items():
        st.subheader(f"📄 {fname}")
        detected = detect_types(df)

        # Initialiser les choix dans session_state
        key_types = f"type_choices_{fname}"
        if key_types not in st.session_state:
            st.session_state[key_types] = detected.copy()

        type_choices = {}
        cols = st.columns([2, 1, 1, 1])
        cols[0].markdown("**Colonne**")
        cols[1].markdown("**Type détecté**")
        cols[2].markdown("**Type cible**")
        cols[3].markdown("**Suggestion**")

        for col_name in df.columns:
            detected_type = detected.get(col_name, "Texte (string)")
            cols = st.columns([2, 1, 1, 1])
            cols[0].text(col_name)
            cols[1].text(detected_type)
            chosen = cols[2].selectbox(
                f"Type pour {col_name}",
                TARGET_TYPES,
                index=TARGET_TYPES.index(detected_type) if detected_type in TARGET_TYPES else 2,
                key=f"type_{fname}_{col_name}",
                label_visibility="collapsed",
            )
            type_choices[col_name] = chosen

            # Suggestion
            suggestion = ""
            if detected_type != chosen:
                suggestion = "✏️ Modifié"
            elif detected_type == "Booléen (bool)":
                suggestion = "💡 Auto-détecté"
            cols[3].text(suggestion)

        st.session_state[key_types] = type_choices

    if st.button("🔄 Appliquer les conversions", type="primary"):
        converted_dfs = {}
        conversion_ok = True

        for fname, df in dataframes.items():
            type_choices = st.session_state[f"type_choices_{fname}"]
            try:
                df_before = df.copy()
                df_after = apply_typing(df, type_choices)
                converted_dfs[fname] = df_after

                v = validate_after_conversion(df_before, df_after,
                                              max_nan_pct=st.session_state.get("max_missing_pct", MAX_NAN_AFTER_CONVERSION_PCT))
                if v["passed"]:
                    st.success(f"✅ {fname} — Conversions appliquées.")
                else:
                    st.warning(v["message"])
                    for d in v["details"]:
                        st.caption(d)
                    conversion_ok = False
                    converted_dfs[fname] = df_after  # Garder quand même

            except Exception as e:
                st.error(f"❌ Erreur sur « {fname} » : {e}")
                conversion_ok = False

        if converted_dfs:
            st.session_state["typed_dataframes"] = converted_dfs
            st.session_state["typing_done"] = True
            if conversion_ok:
                st.success("✅ Toutes les conversions ont été appliquées avec succès !")
            else:
                st.info("ℹ️ Les conversions ont été appliquées malgré les avertissements.")
            st.rerun()


# ═══════════════════════════════════════════════════════════
# ÉTAPE 3 — CONSOLIDATION ET JOINTURES
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[3]:
    st.header("🔗 Consolidation et jointures")
    st.markdown("Combinez vos fichiers en une seule base de données si nécessaire.")

    if not _check_step("typing_done", STEPS[2]):
        st.stop()

    dataframes = st.session_state["typed_dataframes"]

    if len(dataframes) == 1:
        st.info("ℹ️ Un seul fichier chargé — pas de jointure nécessaire.")
        fname = list(dataframes.keys())[0]
        st.session_state["consolidated_df"] = dataframes[fname].copy()
        st.session_state["consolidation_done"] = True
        st.dataframe(dataframes[fname].head(), use_container_width=True)
        st.success("✅ Base consolidée prête (fichier unique).")

    else:
        # Afficher les fichiers disponibles
        st.subheader("Fichiers disponibles")
        for fname, df in dataframes.items():
            st.write(f"**{fname}** — {len(df)} lignes × {len(df.columns)} colonnes")

        st.divider()

        # Agrégation optionnelle
        with st.expander("📊 Agrégation avant jointure (optionnel)"):
            agg_file = st.selectbox("Fichier à agréger", list(dataframes.keys()), key="agg_file")
            agg_df = dataframes[agg_file]
            agg_group = st.selectbox("Grouper par", agg_df.columns.tolist(), key="agg_group")
            agg_cols = st.multiselect("Colonnes à agréger", 
                                      [c for c in agg_df.columns if c != agg_group], key="agg_cols")
            agg_func = st.selectbox("Fonction", AGGREGATION_FUNCTIONS, key="agg_func")

            if agg_cols and st.button("Appliquer l'agrégation"):
                try:
                    agg_result = aggregate(agg_df, agg_group, {c: agg_func for c in agg_cols})
                    dataframes[agg_file] = agg_result
                    st.session_state["typed_dataframes"] = dataframes
                    st.success(f"✅ Agrégation appliquée : {len(agg_result)} lignes.")
                    st.dataframe(agg_result.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Erreur d'agrégation : {e}")

        st.divider()
        st.subheader("Jointure")

        file_names = list(dataframes.keys())
        c1, c2 = st.columns(2)
        with c1:
            left_file = st.selectbox("Table de gauche", file_names, key="join_left")
            left_key = st.selectbox("Clé de gauche",
                                    dataframes[left_file].columns.tolist(), key="join_left_key")
        with c2:
            right_file = st.selectbox("Table de droite",
                                      [f for f in file_names if f != left_file], key="join_right")
            right_key = st.selectbox("Clé de droite",
                                     dataframes[right_file].columns.tolist(), key="join_right_key")

        join_type = st.selectbox(
            "Type de jointure",
            list(JOIN_TYPES.keys()),
            help="\n".join([f"**{k}** : {v}" for k, v in JOIN_TYPES.items()]),
        )

        # Prévisualisation
        if st.button("🔍 Prévisualiser la jointure"):
            preview = preview_join(
                dataframes[left_file], dataframes[right_file],
                left_key, right_key, how=join_type.lower()
            )
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Lignes estimées", preview["estimated_rows"])
            mc2.metric("Clés communes", preview["common_keys"])
            mc3.metric("Colonnes résultat", preview["total_cols"])
            for w in preview["warnings"]:
                st.warning(w)

        if st.button("▶️ Exécuter la jointure", type="primary"):
            try:
                result = perform_join(
                    dataframes[left_file], dataframes[right_file],
                    left_key, right_key, how=join_type.lower()
                )
                stats = get_join_stats(dataframes[left_file], result)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Lignes avant", stats["rows_before"])
                mc2.metric("Lignes après", stats["rows_after"])
                mc3.metric("Lignes perdues", stats.get("rows_lost", 0))

                st.dataframe(result.head(), use_container_width=True)

                v = validate_join(dataframes[left_file], result,
                                  max_loss_pct=MAX_JOIN_LOSS_PCT)
                if v["passed"]:
                    st.success(v["message"])
                else:
                    st.warning(v["message"])
                    for d in v["details"]:
                        st.caption(d)

                st.session_state["consolidated_df"] = result
                st.session_state["consolidation_done"] = True
                st.success("✅ Base consolidée sauvegardée !")

            except Exception as e:
                st.error(f"❌ Erreur de jointure : {e}")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 4 — AUDIT ET ANALYSE EXPLORATOIRE (EDA)
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[4]:
    st.header("🔍 Audit et analyse exploratoire")
    st.markdown("Explorez la qualité de vos données et identifiez les problèmes potentiels.")

    if not _check_step("consolidation_done", STEPS[3]):
        st.stop()

    df = st.session_state["consolidated_df"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Qualité", "📈 Statistiques", "⚠️ Anomalies", "🔗 Corrélations", "📋 Rapport"
    ])

    # ─── Tab 1 : Qualité ───
    with tab1:
        st.subheader("Tableau de qualité des données")
        max_pct = st.session_state.get("max_missing_pct", 20)
        qt = quality_table(df, max_missing_pct=max_pct)
        st.dataframe(qt, use_container_width=True)

    # ─── Tab 2 : Statistiques descriptives ───
    with tab2:
        st.subheader("Statistiques descriptives")
        num_stats, cat_stats = descriptive_stats(df)

        if num_stats is not None and not num_stats.empty:
            st.markdown("**Variables numériques**")
            st.dataframe(num_stats, use_container_width=True)
        if cat_stats is not None and not cat_stats.empty:
            st.markdown("**Variables catégorielles**")
            st.dataframe(cat_stats, use_container_width=True)

    # ─── Tab 3 : Anomalies ───
    with tab3:
        st.subheader("Détection automatique des anomalies")
        anomalies = detect_anomalies(df)
        actions = get_anomaly_actions(anomalies)

        if anomalies.get("constant_cols"):
            st.error(f"🔴 Colonnes constantes : {', '.join(anomalies['constant_cols'])}")
        if anomalies.get("quasi_constant_cols"):
            st.warning(f"🟠 Colonnes quasi-constantes : {', '.join(anomalies['quasi_constant_cols'])}")
        if anomalies.get("high_cardinality_cols"):
            st.warning(f"🟠 Colonnes à très haute cardinalité (identifiants ?) : "
                       f"{', '.join(anomalies['high_cardinality_cols'])}")
        if anomalies.get("outlier_counts"):
            st.info("ℹ️ Outliers détectés (méthode IQR) :")
            for col, count in anomalies["outlier_counts"].items():
                st.write(f"  • **{col}** : {count} outlier(s)")

        if actions:
            st.divider()
            st.markdown("**Actions recommandées :**")
            for a in actions:
                st.write(f"  → {a}")

        st.session_state["anomalies"] = anomalies

    # ─── Tab 4 : Corrélations ───
    with tab4:
        st.subheader("Matrice de corrélation")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr = correlation_matrix(df)
            fig = plot_correlation_heatmap(corr)
            st.pyplot(fig)
            plt.close()

            high_corr = high_correlations(corr, threshold=CORRELATION_THRESHOLD)
            if not high_corr.empty:
                st.warning("⚠️ Corrélations élevées détectées (risque de multicolinéarité) :")
                st.dataframe(high_corr, use_container_width=True)
            else:
                st.success("✅ Aucune multicolinéarité détectée.")
        else:
            st.info("ℹ️ Pas assez de variables numériques pour calculer les corrélations.")

    # ─── Tab 5 : Rapport qualité ───
    with tab5:
        st.subheader("Rapport de qualité automatique")
        anomalies = st.session_state.get("anomalies", detect_anomalies(df))
        quality = compute_quality_score(df, anomalies)

        st.metric("Score de qualité global", f"{quality['score']} / 100")

        if quality.get("issues"):
            st.markdown("**Points d'attention :**")
            for issue in quality["issues"]:
                st.write(f"  → {issue}")

        multicollinearity = len(high_correlations(correlation_matrix(df), CORRELATION_THRESHOLD)) > 0 if len(df.select_dtypes(include=[np.number]).columns) >= 2 else False

        v = validate_data_quality(
            quality["score"], multicollinearity, False,
            threshold=DEFAULT_QUALITY_THRESHOLD
        )
        if v["passed"]:
            st.success(v["message"])
        else:
            st.warning(v["message"])
        for d in v["details"]:
            st.caption(d)

        st.session_state["quality_score"] = quality["score"]
        st.session_state["audit_done"] = True


# ═══════════════════════════════════════════════════════════
# ÉTAPE 5 — PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[5]:
    st.header("🧹 Préparation des données")
    st.markdown("Nettoyez vos données, traitez les valeurs manquantes, normalisez et encodez les variables.")

    if not _check_step("consolidation_done", STEPS[3]):
        st.stop()

    df = st.session_state["consolidated_df"].copy()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🕳️ Manquants", "📊 Outliers", "📐 Normalisation",
        "🏷️ Encodage", "🎯 Variable cible", "✅ Validation"
    ])

    # ─── Tab 1 : Valeurs manquantes ───
    with tab1:
        st.subheader("Traitement des valeurs manquantes")
        cols_with_na = [c for c in df.columns if df[c].isna().sum() > 0]

        if not cols_with_na:
            st.success("✅ Aucune valeur manquante détectée.")
        else:
            missing_strategies = {}
            fixed_values = {}

            for col in cols_with_na:
                na_count = df[col].isna().sum()
                na_pct = round(na_count / len(df) * 100, 1)
                st.write(f"**{col}** — {na_count} manquant(s) ({na_pct}%)")

                is_num = pd.api.types.is_numeric_dtype(df[col])
                options = ["Conserver", "Supprimer la colonne", "Supprimer les lignes"]
                if is_num:
                    options += ["Imputer (moyenne)", "Imputer (médiane)"]
                options += ["Imputer (mode)", "Valeur fixe", "Indicatrice manquant"]

                choice = st.selectbox(f"Stratégie pour {col}", options, key=f"missing_{col}")

                strategy_map = {
                    "Conserver": None,
                    "Supprimer la colonne": "drop_column",
                    "Supprimer les lignes": "drop_rows",
                    "Imputer (moyenne)": "mean",
                    "Imputer (médiane)": "median",
                    "Imputer (mode)": "mode",
                    "Valeur fixe": "fixed",
                    "Indicatrice manquant": "indicator",
                }
                strategy = strategy_map.get(choice)
                if strategy:
                    missing_strategies[col] = strategy
                if choice == "Valeur fixe":
                    fixed_values[col] = st.text_input(f"Valeur pour {col}", key=f"fix_{col}")

            if missing_strategies and st.button("Appliquer le traitement des manquants"):
                df = handle_missing(df, missing_strategies, fixed_values)
                st.session_state["consolidated_df"] = df
                st.success(f"✅ Traitement appliqué. Lignes restantes : {len(df)}")
                st.rerun()

    # ─── Tab 2 : Outliers ───
    with tab2:
        st.subheader("Traitement des outliers")
        num_cols = get_numeric_columns(df)
        outlier_strategies = {}

        for col in num_cols:
            mask = detect_outliers_iqr(df[col])
            n_outliers = mask.sum()
            if n_outliers > 0:
                st.write(f"**{col}** — {n_outliers} outlier(s) détecté(s)")
                choice = st.selectbox(
                    f"Action pour {col}",
                    ["Conserver", "Supprimer les lignes", "Plafonner (capping)", "Transformer (log)"],
                    key=f"outlier_{col}",
                )
                strategy_map = {
                    "Conserver": None,
                    "Supprimer les lignes": "drop",
                    "Plafonner (capping)": "cap",
                    "Transformer (log)": "log",
                }
                strategy = strategy_map.get(choice)
                if strategy:
                    outlier_strategies[col] = strategy

        if not outlier_strategies and num_cols:
            st.success("✅ Aucun outlier significatif détecté.")

        if outlier_strategies and st.button("Appliquer le traitement des outliers"):
            df = handle_outliers(df, outlier_strategies)
            st.session_state["consolidated_df"] = df
            st.success(f"✅ Traitement des outliers appliqué. Lignes : {len(df)}")
            st.rerun()

    # ─── Tab 3 : Normalisation ───
    with tab3:
        st.subheader("Normalisation / Standardisation")
        st.info("ℹ️ La normalisation met toutes les variables numériques sur une même échelle, "
                "ce qui améliore les performances de nombreux algorithmes.")
        num_cols = get_numeric_columns(df)
        if num_cols:
            cols_to_norm = st.multiselect("Colonnes à normaliser", num_cols, key="norm_cols")
            method = st.radio("Méthode", ["StandardScaler (moyenne=0, écart-type=1)",
                                          "Min-Max (entre 0 et 1)", "Aucune"], key="norm_method")
            method_map = {
                "StandardScaler (moyenne=0, écart-type=1)": "standard",
                "Min-Max (entre 0 et 1)": "minmax",
                "Aucune": None,
            }
            if cols_to_norm and method_map[method] and st.button("Appliquer la normalisation"):
                df, scaler = normalize_columns(df, cols_to_norm, method=method_map[method])
                st.session_state["consolidated_df"] = df
                st.session_state["scaler"] = scaler
                st.session_state["scaled_columns"] = cols_to_norm
                st.success("✅ Normalisation appliquée.")
                st.rerun()
        else:
            st.info("ℹ️ Aucune variable numérique à normaliser.")

    # ─── Tab 4 : Encodage ───
    with tab4:
        st.subheader("Encodage des variables catégorielles")
        st.info("ℹ️ Les modèles ML ne comprennent que les nombres. "
                "Il faut convertir les variables texte/catégorielles en valeurs numériques.")
        cat_cols = get_categorical_columns(df)

        if cat_cols:
            encode_strategies = {}
            for col in cat_cols:
                n_unique = df[col].nunique()
                st.write(f"**{col}** — {n_unique} valeur(s) unique(s)")
                if n_unique > MAX_ONEHOT_CARDINALITY:
                    st.warning(f"⚠️ Cardinalité élevée ({n_unique}). One-Hot créera beaucoup de colonnes.")

                choice = st.selectbox(
                    f"Encodage pour {col}",
                    ["One-Hot Encoding", "Label Encoding", "Target Encoding", "Supprimer"],
                    key=f"encode_{col}",
                    help=("**One-Hot** : une colonne binaire par catégorie. "
                          "**Label** : une seule colonne avec des entiers. "
                          "**Target** : encode selon la moyenne de la cible."),
                )
                encode_strategies[col] = choice.lower().replace(" encoding", "").replace("-", "_")

            if st.button("Appliquer l'encodage"):
                target_col = st.session_state.get("target_col")
                df, encoders = encode_categorical(df, encode_strategies, target_col=target_col)
                st.session_state["consolidated_df"] = df
                st.session_state["encoders"] = encoders
                st.success("✅ Encodage appliqué.")
                st.rerun()
        else:
            st.success("✅ Aucune variable catégorielle à encoder.")

    # ─── Tab 5 : Variable cible ───
    with tab5:
        st.subheader("🎯 Sélection de la variable cible")
        all_cols = df.columns.tolist()
        target = st.selectbox("Variable cible (Y)", all_cols, key="target_select")

        if target:
            st.write("**Distribution de la variable cible :**")
            fig = plot_target_distribution(df[target])
            st.pyplot(fig)
            plt.close()

            comment = auto_comment_distribution(df[target])
            st.info(f"💡 {comment}")

            problem_type = st.session_state.get("problem_type", "Régression")
            if problem_type == "Classification":
                imbalance = check_target_imbalance(df[target])
                if imbalance.get("is_imbalanced"):
                    st.warning(f"⚠️ Classes déséquilibrées : {imbalance.get('message', '')}")

            features = st.multiselect(
                "Variables explicatives (X)",
                [c for c in all_cols if c != target],
                default=[c for c in all_cols if c != target],
                key="feature_select",
            )

            # Corrélation avec la cible
            if features and pd.api.types.is_numeric_dtype(df[target]):
                num_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
                if num_features:
                    with st.expander("📊 Corrélation avec la cible"):
                        corr_target = df[num_features + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
                        st.bar_chart(corr_target)

            n_auto = st.slider("Sélection auto (top N)", 2, max(2, len(features)), len(features), key="auto_n")
            if st.button("🔄 Sélection automatique"):
                try:
                    auto_feats = auto_select_features(df, target, n=n_auto)
                    st.session_state["feature_select"] = auto_feats
                    st.info(f"Variables sélectionnées automatiquement : {', '.join(auto_feats)}")
                except Exception as e:
                    st.error(f"Erreur : {e}")

            if st.button("✅ Valider la cible et les variables", type="primary"):
                st.session_state["target_col"] = target
                st.session_state["feature_cols"] = features
                st.success(f"✅ Cible : **{target}** | Variables : {len(features)}")

    # ─── Tab 6 : Validation ───
    with tab6:
        st.subheader("Validation du dataset préparé")

        target_col = st.session_state.get("target_col")
        feature_cols = st.session_state.get("feature_cols")

        if not target_col or not feature_cols:
            st.warning("⚠️ Définissez d'abord la variable cible et les variables explicatives (onglet 🎯).")
        else:
            v = validate_prepared_data(df, target_col, feature_cols,
                                       min_rows=MIN_ROWS_AFTER_CLEANING,
                                       min_features=MIN_FEATURES)
            if v["passed"]:
                st.success(v["message"])
            else:
                st.error(v["message"])
            for d in v["details"]:
                st.caption(d)

            # Récapitulatif
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Lignes", len(df))
            mc2.metric("Variables X", len(feature_cols))
            mc3.metric("Cible Y", target_col)
            mc4.metric("NaN restants", int(df[feature_cols + [target_col]].isna().sum().sum()))

            st.session_state["preparation_done"] = True
            st.session_state["prepared_df"] = df


# ═══════════════════════════════════════════════════════════
# ÉTAPE 6 — GRAPHIQUES EXPLORATOIRES
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[6]:
    st.header("📊 Graphiques exploratoires")
    st.markdown("Visualisez vos données sous différents angles pour mieux les comprendre.")

    if not _check_step("consolidation_done", STEPS[3]):
        st.stop()

    df = st.session_state.get("prepared_df", st.session_state.get("consolidated_df"))

    chart_type = st.selectbox("Type de graphique", [
        "Histogramme", "Boxplot", "Scatter plot",
        "Heatmap de corrélation", "Distribution de la cible",
        "Importance des variables (après modélisation)",
    ])

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    all_cols = df.columns.tolist()

    if chart_type == "Histogramme":
        col = st.selectbox("Variable", all_cols, key="hist_col")
        bins = st.slider("Nombre de bins", 5, 100, 30)
        if col:
            fig = plot_histogram(df, col, bins=bins)
            st.pyplot(fig)
            comment = auto_comment_distribution(df[col])
            st.info(f"💡 {comment}")
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), f"hist_{col}.png", "image/png")
            plt.close()

    elif chart_type == "Boxplot":
        col = st.selectbox("Variable", num_cols, key="box_col")
        group = st.selectbox("Grouper par (optionnel)", ["Aucun"] + cat_cols, key="box_group")
        if col:
            group_col = None if group == "Aucun" else group
            fig = plot_boxplot(df, col, group_col=group_col)
            st.pyplot(fig)
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), f"boxplot_{col}.png", "image/png")
            plt.close()

    elif chart_type == "Scatter plot":
        c1, c2, c3 = st.columns(3)
        with c1:
            col_x = st.selectbox("Axe X", num_cols, key="sc_x")
        with c2:
            col_y = st.selectbox("Axe Y", num_cols, key="sc_y", index=min(1, len(num_cols)-1))
        with c3:
            color_col = st.selectbox("Couleur (optionnel)", ["Aucun"] + cat_cols, key="sc_color")
        if col_x and col_y:
            cc = None if color_col == "Aucun" else color_col
            fig = plot_scatter(df, col_x, col_y, color_col=cc)
            st.pyplot(fig)
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "scatter.png", "image/png")
            plt.close()

    elif chart_type == "Heatmap de corrélation":
        if len(num_cols) >= 2:
            corr = correlation_matrix(df)
            fig = plot_correlation_heatmap(corr)
            st.pyplot(fig)
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "heatmap.png", "image/png")
            plt.close()
        else:
            st.info("ℹ️ Pas assez de variables numériques.")

    elif chart_type == "Distribution de la cible":
        target = st.session_state.get("target_col")
        if target and target in df.columns:
            fig = plot_target_distribution(df[target])
            st.pyplot(fig)
            comment = auto_comment_distribution(df[target])
            st.info(f"💡 {comment}")
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "target_dist.png", "image/png")
            plt.close()
        else:
            st.warning("⚠️ Variable cible non définie. Allez à l'étape 5.")

    elif chart_type == "Importance des variables (après modélisation)":
        best_model = st.session_state.get("best_model")
        feature_cols = st.session_state.get("feature_cols", [])
        if best_model and feature_cols:
            try:
                fig = plot_feature_importance(best_model, feature_cols)
                st.pyplot(fig)
                st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "importance.png", "image/png")
                plt.close()
            except Exception as e:
                st.warning(f"⚠️ Ce modèle ne supporte pas l'importance des variables : {e}")
        else:
            st.info("ℹ️ Veuillez d'abord entraîner un modèle (étape 7).")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 7 — MODÉLISATION
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[7]:
    st.header("🤖 Modélisation")
    st.markdown("Entraînez et comparez plusieurs modèles de Machine Learning.")

    if not _check_step("preparation_done", STEPS[5]):
        st.stop()

    df = st.session_state["prepared_df"]
    target_col = st.session_state["target_col"]
    feature_cols = st.session_state["feature_cols"]
    problem_type = st.session_state["problem_type"]

    # 7.1 Paramètres
    st.subheader("Paramètres généraux")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Taille du jeu de test (%)", 10, 40,
                               int(DEFAULT_TEST_SIZE * 100)) / 100
    with c2:
        random_state = st.number_input("Random state", 0, 9999, DEFAULT_RANDOM_STATE)
    with c3:
        use_cv = st.checkbox("Cross-validation", value=True)
        cv_folds = st.slider("K (folds)", 3, 10, DEFAULT_CV_FOLDS) if use_cv else None

    # 7.2 Sélection des modèles
    st.subheader("Modèles à tester")
    available_models = REGRESSION_MODELS if problem_type == "Régression" else CLASSIFICATION_MODELS

    selected_models = []
    cols = st.columns(4)
    for i, model_name in enumerate(available_models):
        with cols[i % 4]:
            if st.checkbox(model_name, value=(i == 0), key=f"model_{model_name}"):
                selected_models.append(model_name)

    # Hyperparamètres personnalisés
    model_params = {}
    if selected_models:
        with st.expander("⚙️ Hyperparamètres personnalisés"):
            for model_name in selected_models:
                st.markdown(f"**{model_name}**")
                params = {}
                if model_name in ["Ridge", "Lasso"]:
                    params["alpha"] = st.slider(f"Alpha ({model_name})", 0.01, 10.0, 1.0,
                                                 key=f"hp_{model_name}_alpha")
                elif model_name == "ElasticNet":
                    params["alpha"] = st.slider(f"Alpha ({model_name})", 0.01, 10.0, 1.0,
                                                 key=f"hp_{model_name}_alpha")
                    params["l1_ratio"] = st.slider(f"L1 ratio ({model_name})", 0.0, 1.0, 0.5,
                                                    key=f"hp_{model_name}_l1")
                elif model_name == "Arbre de décision":
                    params["max_depth"] = st.slider(f"Profondeur max ({model_name})", 1, 30, 5,
                                                     key=f"hp_{model_name}_depth")
                elif model_name == "Random Forest":
                    params["n_estimators"] = st.slider(f"Nb arbres ({model_name})", 10, 500, 100,
                                                        key=f"hp_{model_name}_n")
                    params["max_depth"] = st.slider(f"Profondeur max ({model_name})", 1, 30, 10,
                                                     key=f"hp_{model_name}_depth")
                elif model_name == "Gradient Boosting":
                    params["learning_rate"] = st.slider(f"Learning rate ({model_name})", 0.01, 1.0, 0.1,
                                                         key=f"hp_{model_name}_lr")
                    params["n_estimators"] = st.slider(f"Nb estimateurs ({model_name})", 10, 500, 100,
                                                        key=f"hp_{model_name}_n")
                elif model_name == "KNN":
                    params["n_neighbors"] = st.slider(f"Nb voisins ({model_name})", 1, 50, 5,
                                                       key=f"hp_{model_name}_k")
                if params:
                    model_params[model_name] = params

    # 7.3 Entraînement
    if selected_models and st.button("🚀 Lancer l'entraînement", type="primary"):
        try:
            X_train, X_test, y_train, y_test = split_data(
                df, target_col, feature_cols,
                test_size=test_size, random_state=random_state
            )
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test

            progress = st.progress(0)
            status = st.empty()

            results = train_multiple(
                selected_models, X_train, y_train, X_test, y_test,
                problem_type=problem_type,
                model_params=model_params,
                cv_folds=cv_folds,
                progress_callback=lambda i, n, name: (
                    progress.progress((i + 1) / n),
                    status.write(f"Entraînement de **{name}**... ({i+1}/{n})")
                ),
            )

            st.session_state["model_results"] = results
            progress.progress(1.0)
            status.success("✅ Tous les modèles ont été entraînés !")

            # 7.4 Tableau comparatif
            st.subheader("📊 Tableau comparatif")
            rt = results_table(results, problem_type)
            st.dataframe(rt, use_container_width=True)

            # Meilleur modèle
            best = max(results, key=lambda r: r.get("test_score", 0))
            st.session_state["best_model"] = best.get("model")
            st.session_state["best_model_name"] = best.get("name")
            st.session_state["best_model_result"] = best

            st.metric("🏆 Meilleur modèle", f"{best['name']} — Score : {best['test_score']:.4f}")

            # Validation
            min_score = st.session_state.get("min_score", DEFAULT_MIN_SCORE)
            max_overfit = st.session_state.get("max_overfit_pct", DEFAULT_MAX_OVERFIT_PCT)
            v = validate_model_scores(results, min_score=min_score, max_overfit_pct=max_overfit)
            if v["passed"]:
                st.success(v["message"])
            else:
                st.warning(v["message"])
            for d in v["details"]:
                st.caption(d)

            st.session_state["modeling_done"] = True

        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement : {e}")

    # Afficher les résultats précédents si disponibles
    elif st.session_state.get("model_results"):
        st.subheader("📊 Résultats précédents")
        rt = results_table(st.session_state["model_results"], problem_type)
        st.dataframe(rt, use_container_width=True)
        best = st.session_state.get("best_model_result", {})
        if best:
            st.metric("🏆 Meilleur modèle", f"{best.get('name', '?')} — Score : {best.get('test_score', 0):.4f}")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 8 — OPTIMISATION DES HYPERPARAMÈTRES
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[8]:
    st.header("🔧 Optimisation des hyperparamètres")
    st.markdown("Affinez les paramètres du modèle sélectionné pour améliorer ses performances.")

    if not _check_step("modeling_done", STEPS[7]):
        st.stop()

    results = st.session_state["model_results"]
    model_names = [r["name"] for r in results if r.get("model") is not None]
    problem_type = st.session_state["problem_type"]

    selected = st.selectbox("Modèle à optimiser", model_names)
    method = st.radio("Méthode de recherche", ["Grid Search (exhaustif)", "Random Search (rapide)"])

    # Grille de paramètres
    st.subheader("Grille de paramètres")
    param_grid = {}

    if selected in ["Ridge", "Lasso"]:
        alpha_values = st.text_input("Alpha (séparés par des virgules)", "0.01, 0.1, 1.0, 10.0")
        param_grid["alpha"] = [float(x.strip()) for x in alpha_values.split(",")]
    elif selected == "ElasticNet":
        alpha_values = st.text_input("Alpha", "0.01, 0.1, 1.0")
        l1_values = st.text_input("L1 ratio", "0.2, 0.5, 0.8")
        param_grid["alpha"] = [float(x.strip()) for x in alpha_values.split(",")]
        param_grid["l1_ratio"] = [float(x.strip()) for x in l1_values.split(",")]
    elif selected in ["Arbre de décision"]:
        depth_values = st.text_input("Profondeur max", "3, 5, 10, 15, 20")
        param_grid["max_depth"] = [int(x.strip()) for x in depth_values.split(",")]
    elif selected == "Random Forest":
        n_est = st.text_input("Nb arbres", "50, 100, 200")
        depth = st.text_input("Profondeur max", "5, 10, 15")
        param_grid["n_estimators"] = [int(x.strip()) for x in n_est.split(",")]
        param_grid["max_depth"] = [int(x.strip()) for x in depth.split(",")]
    elif selected == "Gradient Boosting":
        lr = st.text_input("Learning rate", "0.01, 0.05, 0.1, 0.2")
        n_est = st.text_input("Nb estimateurs", "50, 100, 200")
        param_grid["learning_rate"] = [float(x.strip()) for x in lr.split(",")]
        param_grid["n_estimators"] = [int(x.strip()) for x in n_est.split(",")]
    elif selected == "KNN":
        k_values = st.text_input("Nb voisins", "3, 5, 7, 11, 15")
        param_grid["n_neighbors"] = [int(x.strip()) for x in k_values.split(",")]
    else:
        st.info("ℹ️ Pas de grille prédéfinie pour ce modèle. Vous pouvez quand même lancer l'entraînement.")

    n_iter = None
    if "Random" in method:
        n_iter = st.slider("Nombre d'itérations", 5, 200, 50)

    if param_grid and st.button("🚀 Lancer l'optimisation", type="primary"):
        try:
            # Retrouver le modèle
            model_result = next(r for r in results if r["name"] == selected)
            model = model_result["model"]
            score_before = model_result["test_score"]

            X_train = st.session_state["X_train"]
            y_train = st.session_state["y_train"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]

            with st.spinner("Optimisation en cours..."):
                search_method = "grid" if "Grid" in method else "random"
                opt_result = optimize_model(
                    model, X_train, y_train, param_grid,
                    method=search_method, cv=DEFAULT_CV_FOLDS, n_iter=n_iter,
                )

            best_model = opt_result["best_model"]
            best_params = opt_result["best_params"]

            # Score après
            if problem_type == "Régression":
                from sklearn.metrics import r2_score
                y_pred = best_model.predict(X_test)
                score_after = r2_score(y_test, y_pred)
            else:
                from sklearn.metrics import accuracy_score
                y_pred = best_model.predict(X_test)
                score_after = accuracy_score(y_test, y_pred)

            c1, c2, c3 = st.columns(3)
            c1.metric("Score avant", f"{score_before:.4f}")
            c2.metric("Score après", f"{score_after:.4f}")
            c3.metric("Gain", f"{(score_after - score_before):.4f}",
                       delta=f"{(score_after - score_before) / max(abs(score_before), 0.001) * 100:.1f}%")

            st.write("**Meilleurs paramètres :**", best_params)

            if "cv_results" in opt_result:
                with st.expander("📋 Détail de tous les essais"):
                    st.dataframe(pd.DataFrame(opt_result["cv_results"]), use_container_width=True)

            v = validate_optimization(score_before, score_after)
            if v["passed"]:
                st.success(v["message"])
            else:
                st.warning(v["message"])

            if st.button("✅ Adopter ces paramètres"):
                st.session_state["best_model"] = best_model
                st.session_state["best_model_result"]["model"] = best_model
                st.session_state["best_model_result"]["test_score"] = score_after
                st.session_state["optimization_done"] = True
                st.success("✅ Paramètres adoptés !")

        except Exception as e:
            st.error(f"❌ Erreur : {e}")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 9 — MODIFICATION DES VARIABLES
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[9]:
    st.header("🔬 Modification des variables")
    st.markdown("Créez, transformez ou supprimez des variables puis relancez la modélisation.")

    if not _check_step("modeling_done", STEPS[7]):
        st.stop()

    df = st.session_state["prepared_df"].copy()
    target_col = st.session_state["target_col"]
    feature_cols = st.session_state["feature_cols"]

    # Historique
    if "modification_history" not in st.session_state:
        st.session_state["modification_history"] = []

    st.dataframe(df.head(), use_container_width=True)

    action = st.selectbox("Action", [
        "Ajouter une variable (combinaison)",
        "Supprimer une variable",
        "Transformer une variable",
        "Renommer une variable",
        "Discrétiser une variable numérique",
    ])

    num_cols = get_numeric_columns(df)
    all_cols = [c for c in df.columns.tolist() if c != target_col]

    if action == "Ajouter une variable (combinaison)":
        c1, c2, c3 = st.columns(3)
        with c1:
            col_a = st.selectbox("Colonne A", num_cols, key="comb_a")
        with c2:
            operation = st.selectbox("Opération", ["Somme", "Différence", "Produit", "Ratio"])
        with c3:
            col_b = st.selectbox("Colonne B", [c for c in num_cols if c != col_a], key="comb_b")

        op_map = {"Somme": "sum", "Différence": "diff", "Produit": "product", "Ratio": "ratio"}
        new_name = st.text_input("Nom de la nouvelle variable", f"{col_a}_{op_map[operation]}_{col_b}")

        if st.button("Ajouter"):
            try:
                df = combine_columns(df, col_a, col_b, op_map[operation], new_name)
                st.session_state["prepared_df"] = df
                feature_cols.append(new_name)
                st.session_state["feature_cols"] = feature_cols
                st.session_state["modification_history"].append(f"Ajout : {new_name}")
                st.success(f"✅ Variable « {new_name} » ajoutée.")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

    elif action == "Supprimer une variable":
        col = st.selectbox("Variable à supprimer", all_cols, key="drop_col")
        if st.button("Supprimer"):
            df = drop_column(df, col)
            st.session_state["prepared_df"] = df
            if col in feature_cols:
                feature_cols.remove(col)
                st.session_state["feature_cols"] = feature_cols
            st.session_state["modification_history"].append(f"Suppression : {col}")
            st.success(f"✅ Variable « {col} » supprimée.")
            st.rerun()

    elif action == "Transformer une variable":
        col = st.selectbox("Variable", num_cols, key="transform_col")
        transform = st.selectbox("Transformation", ["Log", "Racine carrée", "Carré"])
        t_map = {"Log": "log", "Racine carrée": "sqrt", "Carré": "square"}
        if st.button("Appliquer"):
            try:
                df = transform_column(df, col, t_map[transform])
                st.session_state["prepared_df"] = df
                st.session_state["modification_history"].append(f"Transformation {transform} : {col}")
                st.success(f"✅ Transformation « {transform} » appliquée à « {col} ».")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

    elif action == "Renommer une variable":
        col = st.selectbox("Variable", all_cols, key="rename_col")
        new_name = st.text_input("Nouveau nom", col)
        if st.button("Renommer") and new_name != col:
            df = rename_column(df, col, new_name)
            st.session_state["prepared_df"] = df
            if col in feature_cols:
                idx = feature_cols.index(col)
                feature_cols[idx] = new_name
                st.session_state["feature_cols"] = feature_cols
            st.session_state["modification_history"].append(f"Renommage : {col} → {new_name}")
            st.success(f"✅ « {col} » renommé en « {new_name} ».")
            st.rerun()

    elif action == "Discrétiser une variable numérique":
        col = st.selectbox("Variable", num_cols, key="disc_col")
        n_bins = st.slider("Nombre de tranches", 2, 10, 4)
        if st.button("Discrétiser"):
            try:
                df = discretize_column(df, col, n_bins=n_bins)
                st.session_state["prepared_df"] = df
                st.session_state["modification_history"].append(f"Discrétisation ({n_bins} bins) : {col}")
                st.success(f"✅ « {col} » discrétisé en {n_bins} tranches.")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

    # Historique et annulation
    st.divider()
    history = st.session_state.get("modification_history", [])
    if history:
        st.subheader("📜 Historique des modifications")
        for h in history:
            st.write(f"  • {h}")

    # Relancer la modélisation
    st.divider()
    if st.button("🔄 Relancer la modélisation avec ces variables", type="primary"):
        try:
            df = st.session_state["prepared_df"]
            X_train, X_test, y_train, y_test = split_data(
                df, target_col, feature_cols,
                test_size=st.session_state.get("test_size", DEFAULT_TEST_SIZE),
                random_state=st.session_state.get("random_state", DEFAULT_RANDOM_STATE),
            )

            # Ré-entraîner le meilleur modèle
            best_name = st.session_state.get("best_model_name", "Régression Linéaire")
            problem_type = st.session_state["problem_type"]
            model = get_model(best_name, problem_type)
            result = train_model(model, X_train, y_train, X_test, y_test, problem_type)

            old_score = st.session_state.get("best_model_result", {}).get("test_score", 0)
            new_score = result["test_score"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Score avant", f"{old_score:.4f}")
            c2.metric("Score après", f"{new_score:.4f}")
            delta = new_score - old_score
            c3.metric("Delta", f"{delta:+.4f}", delta=f"{'Amélioration' if delta > 0 else 'Dégradation'}")

        except Exception as e:
            st.error(f"❌ Erreur : {e}")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 10 — ANALYSE DES ERREURS ET RÉSIDUS
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[10]:
    st.header("📉 Analyse des erreurs et résidus")
    st.markdown("Examinez la qualité des prédictions de votre meilleur modèle.")

    if not _check_step("modeling_done", STEPS[7]):
        st.stop()

    best_model = st.session_state.get("best_model")
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    problem_type = st.session_state["problem_type"]

    if best_model is None:
        st.error("❌ Aucun modèle disponible.")
        st.stop()

    y_pred = best_model.predict(X_test)

    if problem_type == "Régression":
        tab1, tab2, tab3, tab4 = st.tabs([
            "Réel vs Prédit", "Résidus", "Distribution résidus", "Top erreurs"
        ])

        with tab1:
            fig = plot_real_vs_pred(y_test, y_pred)
            st.pyplot(fig)
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "real_vs_pred.png", "image/png")
            plt.close()

        with tab2:
            fig = plot_residuals(y_test, y_pred)
            st.pyplot(fig)
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "residuals.png", "image/png")
            plt.close()

        with tab3:
            fig = plot_residual_distribution(y_test, y_pred)
            st.pyplot(fig)
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "resid_dist.png", "image/png")
            plt.close()

        with tab4:
            top_err = get_top_errors(y_test, y_pred, n=10)
            st.dataframe(top_err, use_container_width=True)

        # Commentaire auto
        comment = auto_comment_residuals(y_test, y_pred)
        st.info(f"💡 {comment}")

        # Validation
        residuals = np.array(y_test) - np.array(y_pred)
        v = validate_residuals(residuals, y_pred)
        if v["passed"]:
            st.success(v["message"])
        else:
            st.warning(v["message"])
        for d in v["details"]:
            st.caption(d)

    else:  # Classification
        tab1, tab2, tab3, tab4 = st.tabs([
            "Matrice de confusion", "Courbe ROC", "Rapport", "Cas mal classés"
        ])

        with tab1:
            fig = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig)
            st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "confusion.png", "image/png")
            plt.close()

        with tab2:
            try:
                y_proba = best_model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    fig = plot_roc_curve(y_test, y_proba[:, 1])
                    st.pyplot(fig)
                    st.download_button("📥 Télécharger", fig_to_png_bytes(fig), "roc.png", "image/png")
                    plt.close()
                else:
                    st.info("ℹ️ Courbe ROC disponible uniquement pour la classification binaire.")
            except AttributeError:
                st.warning("⚠️ Ce modèle ne supporte pas predict_proba.")

        with tab3:
            report = get_classification_report(y_test, y_pred)
            st.dataframe(report, use_container_width=True)

        with tab4:
            misclass = get_misclassified(y_test, y_pred, X=X_test, n=10)
            st.dataframe(misclass, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# ÉTAPE 11 — VALIDATION FINALE ET RAPPORT
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[11]:
    st.header("📋 Validation finale et rapport")
    st.markdown("Récapitulatif complet de votre projet et export des résultats.")

    if not _check_step("modeling_done", STEPS[7]):
        st.stop()

    # Récapitulatif
    st.subheader("🏗️ Configuration du projet")
    c1, c2, c3 = st.columns(3)
    c1.metric("Projet", st.session_state.get("project_name", "?"))
    c2.metric("Type", st.session_state.get("problem_type", "?"))
    c3.metric("Score min requis", st.session_state.get("min_score", "?"))

    st.subheader("📊 Données")
    df = st.session_state.get("prepared_df", st.session_state.get("consolidated_df"))
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes", len(df))
        c2.metric("Colonnes", len(df.columns))
        c3.metric("Variable cible", st.session_state.get("target_col", "?"))

    st.subheader("🏆 Meilleur modèle")
    best_result = st.session_state.get("best_model_result", {})
    if best_result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Modèle", best_result.get("name", "?"))
        c2.metric("Score test", f"{best_result.get('test_score', 0):.4f}")
        c3.metric("Score train", f"{best_result.get('train_score', 0):.4f}")

    # Tableau de bord des validations
    st.subheader("✅ Tableau de bord des validations")
    validations = {}
    validations["V1 - Chargement"] = "✅" if st.session_state.get("files_loaded") else "❌"
    validations["V2 - Typage"] = "✅" if st.session_state.get("typing_done") else "❌"
    validations["V3 - Consolidation"] = "✅" if st.session_state.get("consolidation_done") else "❌"
    validations["V4 - Qualité"] = "✅" if st.session_state.get("audit_done") else "⚠️"
    validations["V5 - Préparation"] = "✅" if st.session_state.get("preparation_done") else "❌"
    validations["V6 - Modélisation"] = "✅" if st.session_state.get("modeling_done") else "❌"
    validations["V7 - Optimisation"] = "✅" if st.session_state.get("optimization_done") else "⚠️ (optionnel)"
    validations["V8 - Résidus"] = "ℹ️ Voir étape 10"

    for k, v in validations.items():
        st.write(f"  {v} {k}")

    # Exports
    st.divider()
    st.subheader("💾 Exports")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if df is not None:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Dataset final (CSV)", csv, "dataset_final.csv", "text/csv")

    with c2:
        best_model = st.session_state.get("best_model")
        if best_model:
            model_bytes = io.BytesIO()
            joblib.dump(best_model, model_bytes)
            model_bytes.seek(0)
            st.download_button("📥 Modèle (.pkl)", model_bytes, "model.pkl", "application/octet-stream")

    with c3:
        if best_model and "X_test" in st.session_state:
            y_pred = best_model.predict(st.session_state["X_test"])
            pred_df = pd.DataFrame({
                "Réel": st.session_state["y_test"],
                "Prédit": y_pred,
            })
            csv_pred = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Prédictions (CSV)", csv_pred, "predictions.csv", "text/csv")

    with c4:
        try:
            report_html = generate_html_report(
                project_config={
                    "name": st.session_state.get("project_name", ""),
                    "type": st.session_state.get("problem_type", ""),
                    "min_score": st.session_state.get("min_score", 0),
                },
                data_summary={
                    "rows": len(df) if df is not None else 0,
                    "cols": len(df.columns) if df is not None else 0,
                    "target": st.session_state.get("target_col", ""),
                },
                model_summary=best_result,
                validations=validations,
            )
            st.download_button("📥 Rapport (HTML)", report_html, "rapport.html", "text/html")
        except Exception:
            st.info("ℹ️ Rapport non disponible.")


# ═══════════════════════════════════════════════════════════
# ÉTAPE 12 — PRÉDICTION SUR NOUVELLES DONNÉES
# ═══════════════════════════════════════════════════════════
elif current_step == STEPS[12]:
    st.header("🔮 Prédiction sur nouvelles données")
    st.markdown("Chargez un nouveau fichier et appliquez votre modèle entraîné pour générer des prédictions.")

    if not _check_step("modeling_done", STEPS[7]):
        st.stop()

    best_model = st.session_state.get("best_model")
    feature_cols = st.session_state.get("feature_cols", [])

    if best_model is None:
        st.error("❌ Aucun modèle entraîné disponible.")
        st.stop()

    uploaded = st.file_uploader("Fichier à prédire (CSV)", type=["csv"], key="predict_upload")

    if uploaded:
        try:
            sep_label = st.selectbox("Séparateur", list(SEPARATORS.keys()), key="pred_sep")
            enc = st.selectbox("Encodage", ENCODINGS, key="pred_enc")
            new_df = load_file(uploaded, separator=SEPARATORS[sep_label], encoding=enc)

            st.dataframe(new_df.head(), use_container_width=True)

            # Vérification des colonnes
            missing_cols = [c for c in feature_cols if c not in new_df.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
                st.stop()

            st.success("✅ Toutes les colonnes requises sont présentes.")

            # Appliquer les mêmes transformations
            scaler = st.session_state.get("scaler")
            scaled_cols = st.session_state.get("scaled_columns", [])
            if scaler and scaled_cols:
                cols_to_scale = [c for c in scaled_cols if c in new_df.columns]
                if cols_to_scale:
                    new_df[cols_to_scale] = scaler.transform(new_df[cols_to_scale])

            if st.button("🚀 Générer les prédictions", type="primary"):
                X_new = new_df[feature_cols]
                predictions = best_model.predict(X_new)
                new_df["Prédiction"] = predictions

                st.subheader("Résultats")
                st.dataframe(new_df, use_container_width=True)

                csv = new_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Télécharger les prédictions",
                    csv, "predictions_nouvelles.csv", "text/csv",
                )

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
