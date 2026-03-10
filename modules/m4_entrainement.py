# -*- coding: utf-8 -*-
"""
m4_entrainement.py — Module 4 : Modélisation.

Étape 7 : Entraînement des modèles.
Utilise les modèles pré-recommandés par le diagnostic (step 3).
Split → Sélection modèles → Entraînement → Comparaison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    REGRESSION_MODELS, CLASSIFICATION_MODELS,
    DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS,
)
from src.models import split_data, split_data_chronological, train_multiple, get_model
from src.evaluation import results_table
from src.timeseries import (
    prepare_timeseries, suggest_arima_order, fit_arima,
    arima_grid_search, fit_sarima, detect_seasonality,
)
from utils.projet_manager import (
    sauvegarder_rapport, sauvegarder_csv, sauvegarder_modele,
    ajouter_historique, sauvegarder_objet,
)
from utils.data_utils import recommend_models


def _sauvegarder_splits(rapport: dict):
    """Sauvegarde X_train, X_test, y_train, y_test sur disque pour restauration."""
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")
    feature_names = st.session_state.get("feature_cols_used", [])

    if X_train is None or y_train is None:
        return

    # Convertir en DataFrame pour CSV (X peut être ndarray après scaling)
    if not isinstance(X_train, pd.DataFrame):
        cols = feature_names if len(feature_names) == X_train.shape[1] \
            else [f"Feature_{i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)

    sauvegarder_csv(rapport, X_train, "X_train.csv")
    sauvegarder_csv(rapport, X_test, "X_test.csv")
    sauvegarder_csv(rapport, pd.DataFrame(y_train, columns=["target"]), "y_train.csv")
    sauvegarder_csv(rapport, pd.DataFrame(y_test, columns=["target"]), "y_test.csv")


def afficher_entrainement():
    """Étape 7 — Modélisation : split, choix modèles, entraînement."""
    st.caption("ÉTAPE 7")

    df = st.session_state.get("df_courant")
    target_col = st.session_state.get("target_col")
    feature_cols = st.session_state.get("feature_cols")
    problem_type = st.session_state.get("problem_type", "")
    is_ts = problem_type == "Série temporelle" or st.session_state.get("ts_horizon_mode")

    if df is None or not target_col:
        st.warning("⚠️ Complétez d'abord les étapes précédentes (cible + variables).")
        return

    # feature_cols peut être vide pour les séries temporelles (ARIMA univarié)
    if not is_ts and not feature_cols:
        st.warning("⚠️ Complétez d'abord les étapes précédentes (cible + variables).")
        return

    if not st.session_state.get("transformation_done"):
        st.info("🔒 **Verrouillé** — Terminez d'abord les transformations (étape 6).")
        return

    with st.expander("🎓 Comment fonctionne l'entraînement ?", expanded=False):
        st.markdown("""
L'entraînement d'un modèle suit toujours le même principe :

```
  DONNÉES
    │
    ├── 80% → JEU D'ENTRAÎNEMENT  ←  Le modèle apprend sur ces données
    │              │
    │         Algorithme apprend les règles :
    │         "quand surface ↑ et quartier = centre → prix ↑"
    │
    └── 20% → JEU DE TEST          ←  On vérifie sur des données jamais vues
                   │
              Score = combien de bonnes prédictions ?
```

**Pourquoi séparer ?** Pour vérifier que le modèle a **compris les règles générales**
et ne fait pas que "réciter" les données apprises (sur-apprentissage).

| Méthode de split | Quand l'utiliser |
|---|---|
| **Aléatoire** | Données sans ordre particulier (par défaut) |
| **Chronologique** | Données temporelles (on prédit le futur, pas le passé) |

| Score | Interprétation |
|---|---|
| **> 0.80** | 🟢 Bon modèle |
| **0.60 – 0.80** | 🟠 Acceptable, peut être amélioré |
| **< 0.60** | 🔴 Le modèle a du mal → revoir les variables ou le nettoyage |
""")

    # Déterminer le type de problème
    problem_type = st.session_state.get("problem_type")
    if not problem_type:
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 10:
            problem_type = "Régression"
        else:
            problem_type = "Classification"
        st.session_state["problem_type"] = problem_type

    st.info(f"**Type de problème détecté :** {problem_type}")

    # ═══════════════════════════════════════
    # Parcours Série temporelle
    # ═══════════════════════════════════════
    if problem_type == "Série temporelle":
        _afficher_entrainement_ts()
        return

    # Notice mode horizon
    if st.session_state.get("ts_horizon_mode"):
        h_val = st.session_state.get("ts_horizon_value", "?")
        st.info(
            f"🎯 **Mode prédiction horizon t+{h_val}** — Le modèle de "
            f"régression sera entraîné sur les features temporelles "
            f"construites. Utilisez un split **chronologique** pour "
            f"respecter l'ordre temporel des données.")

    # ═══════════════════════════════════════
    # Parcours ML classique
    # ═══════════════════════════════════════

    # ═══════════════════════════════════════
    # 1. Split train / test
    # ═══════════════════════════════════════
    st.subheader("1. Découpage des données")

    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Part du jeu de test (%)", 10, 40,
                               int(DEFAULT_TEST_SIZE * 100), step=5,
                               key="test_pct",
                               help="Pourcentage de données réservées au test (20% = bon compromis)") / 100
    with c2:
        default_split = 1 if st.session_state.get("ts_horizon_mode") else 0
        split_method = st.radio("Méthode de split",
                                 ["Aléatoire", "Chronologique"],
                                 index=default_split,
                                 key="split_method")

    # Effectuer le split
    if "X_train" not in st.session_state:
        try:
            valid_features = [c for c in feature_cols if c in df.columns and c != target_col]
            if not valid_features:
                st.error("❌ Aucune colonne de features valide.")
                return

            if split_method == "Chronologique":
                X_train, X_test, y_train, y_test = split_data_chronological(
                    df, target_col, valid_features, test_size=test_size)
            else:
                X_train, X_test, y_train, y_test = split_data(
                    df, target_col, valid_features, test_size=test_size)

            st.session_state.update({
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "feature_cols_used": valid_features,
            })
        except Exception as e:
            st.error(f"❌ Erreur de split : {e}")
            return

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("🏋️ Train", f"{len(X_train)} lignes")
    with c2:
        st.metric("🧪 Test", f"{len(X_test)} lignes")

    # Bouton pour refaire le split
    if st.button("🔄 Refaire le split", key="redo_split"):
        for k in ["X_train", "X_test", "y_train", "y_test",
                   "resultats_modeles", "meilleur_modele"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()

    # ═══════════════════════════════════════
    # 2. Sélection des modèles
    # ═══════════════════════════════════════
    st.subheader("2. Sélection des modèles")

    model_list = REGRESSION_MODELS if problem_type == "Régression" else CLASSIFICATION_MODELS

    # Pré-sélection intelligente
    recommandes = st.session_state.get("modeles_recommandes", [])
    if not recommandes:
        reco = recommend_models(df, target_col, problem_type)
        recommandes = reco.get("principaux", [])

    # Pré-cocher les recommandés
    default_sel = [m for m in recommandes if m in model_list]
    if not default_sel:
        default_sel = model_list[:2]

    selected_models = st.multiselect(
        "Modèles à entraîner",
        model_list,
        default=default_sel,
        help="Les modèles pré-cochés viennent du diagnostic.",
        key="model_selection",
    )

    if recommandes:
        st.caption(f"💡 Recommandés par le diagnostic : {', '.join(recommandes)}")

    # Cross-validation
    use_cv = st.checkbox("Validation croisée (plus lent mais plus fiable)",
                          value=False, key="use_cv")
    cv_folds = DEFAULT_CV_FOLDS
    if use_cv:
        cv_folds = st.slider("Nombre de folds", 3, 10, DEFAULT_CV_FOLDS, key="cv_folds",
                              help="Nombre de divisions pour la validation croisée (5-10 = robuste)")

    # Pondération des classes (classification uniquement)
    use_class_weight = False
    if problem_type == "Classification":
        use_class_weight = st.checkbox(
            "⚖️ Pondérer par le poids des classes (class_weight='balanced')",
            value=False, key="use_class_weight",
            help="Utile quand les classes sont déséquilibrées : "
                 "donne plus d'importance aux classes rares.")

    st.divider()

    # ═══════════════════════════════════════
    # 3. Entraînement
    # ═══════════════════════════════════════
    st.subheader("3. Entraînement")

    if not selected_models:
        st.info("Sélectionnez au moins un modèle.")
        return

    if st.button("🚀 Lancer l'entraînement", type="primary", key="train_btn"):
        progress_bar = st.progress(0, text="Entraînement en cours…")

        def update_progress(i, total, model_name):
            progress_bar.progress(
                (i + 1) / total,
                text=f"Entraînement : {model_name} ({i + 1}/{total})"
            )

        try:
            # Préparer les paramètres avec class_weight si demandé
            model_params = {}
            if use_class_weight and problem_type == "Classification":
                for m in selected_models:
                    model_params[m] = {"class_weight": "balanced"}

            results = train_multiple(
                selected_models, X_train, y_train, X_test, y_test,
                problem_type, model_params=model_params,
                cv_folds=cv_folds if use_cv else 0,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0, text="✅ Entraînement terminé !")

            st.session_state["resultats_modeles"] = results

            # Trier et trouver le meilleur (None → -999 pour les modèles en erreur)
            metric_key = "test_score"
            results_sorted = sorted(
                results,
                key=lambda r: r.get(metric_key) if r.get(metric_key) is not None else -999,
                reverse=True)
            best = results_sorted[0]
            st.session_state["meilleur_modele"] = best
            st.session_state["trained_models"] = {r["name"]: r for r in results}

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            return

    # ═══════════════════════════════════════
    # 4. Résultats
    # ═══════════════════════════════════════
    results = st.session_state.get("resultats_modeles")
    if not results:
        return

    st.subheader("4. Résultats")

    # Tableau de comparaison
    df_results = results_table(results, problem_type)
    st.dataframe(df_results, use_container_width=True)

    # Meilleur modèle
    best = st.session_state.get("meilleur_modele")
    if best and best.get("test_score") is not None:
        st.success(f"🏆 **Meilleur modèle : {best['name']}**  "
                   f"(score test : {best.get('test_score', 0):.4f})")

        # Alerte overfitting
        train_s = best.get("train_score") or 0
        test_s = best.get("test_score") or 0
        if train_s > 0 and test_s > 0:
            overfit = (train_s - test_s) / max(train_s, 0.0001) * 100
            if overfit > 10:
                st.warning(f"⚠️ Overfitting possible : écart train/test = {overfit:.1f}%")

    # ═══════════════════════════════════════
    # 5. Analyse et propositions
    # ═══════════════════════════════════════
    st.divider()
    with st.expander("💡 Analyse et recommandations", expanded=True):
        best_r = st.session_state.get("meilleur_modele", {})
        score = best_r.get("test_score") or 0
        train_s = best_r.get("train_score") or 0
        gap = abs(train_s - score) * 100 if train_s > 0 else 0

        # Diagnostic
        if score > 0.85:
            st.success(f"🟢 Très bon score ({score:.1%}). Le modèle est performant.")
        elif score > 0.65:
            st.warning(f"🟡 Score correct ({score:.1%}) mais améliorable.")
        else:
            st.error(f"🔴 Score faible ({score:.1%}). Revoir les données ou features.")

        if gap > 15:
            st.warning(f"⚠️ Sur-apprentissage détecté (écart {gap:.0f}%). "
                       "Essayez un modèle plus simple ou plus de données.")

        # Propositions
        st.markdown("**Propositions pour la suite :**")
        suggestions = []
        if score < 0.85:
            suggestions.append("- Retournez à l'étape 6 pour créer de nouvelles variables (feature engineering)")
            suggestions.append("- Essayez d'autres modèles (Gradient Boosting, Random Forest)")
        if gap > 10:
            suggestions.append("- Réduisez la complexité du modèle (moins de profondeur, régularisation)")
            suggestions.append("- Augmentez la taille du jeu d'entraînement")
        if problem_type == "Classification" and not use_class_weight:
            suggestions.append("- Activez la pondération des classes si elles sont déséquilibrées")
        suggestions.append("- Validez le modèle et passez à l'évaluation détaillée (étape 8)")
        for s in suggestions:
            st.markdown(s)

    # Validation
    st.divider()
    model_choice = st.selectbox("Modèle à conserver pour l'évaluation",
                                 [r["name"] for r in results], key="keep_model")

    if st.button("✅ Valider ce modèle", type="primary", key="validate_model"):
        chosen = next((r for r in results if r["name"] == model_choice), best)
        st.session_state["meilleur_modele"] = chosen
        st.session_state["entrainement_done"] = True

        rapport = st.session_state.get("rapport", {})
        if rapport:
            rapport["modele"] = {
                "nom": chosen["name"],
                "score_train": chosen.get("train_score"),
                "score_test": chosen.get("test_score"),
                "temps": chosen.get("train_time"),
            }
            # Sauvegarder les colonnes utilisées
            feat_used = st.session_state.get("feature_cols_used", [])
            if feat_used:
                rapport["colonnes_features_used"] = feat_used

            rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 8)
            ajouter_historique(rapport, f"Modèle validé : {chosen['name']}")
            sauvegarder_rapport(rapport)

            # Sauvegarder le modèle
            if "model" in chosen:
                sauvegarder_modele(rapport, chosen["model"], f"{chosen['name']}.joblib")

            # Sauvegarder les splits train/test pour pouvoir les restaurer
            _sauvegarder_splits(rapport)

        st.session_state["_pending_step"] = 8
        st.rerun()

    # ═══════════════════════════════════════
    # 6. Télécharger le modèle
    # ═══════════════════════════════════════
    best = st.session_state.get("meilleur_modele")
    if best and "model" in best and st.session_state.get("entrainement_done"):
        st.divider()
        st.subheader("6. Télécharger le modèle")
        st.info("💡 Téléchargez votre modèle pour le réutiliser plus tard, "
                "même depuis un autre ordinateur.")

        # Sérialiser le modèle + métadonnées
        export_data = {
            "model": best["model"],
            "name": best["name"],
            "problem_type": problem_type,
            "feature_cols": st.session_state.get("feature_cols_used",
                            st.session_state.get("feature_cols", [])),
            "target_col": st.session_state.get("target_col", ""),
            "encoders": st.session_state.get("encoders", {}),
            "scaler": st.session_state.get("scaler"),
            "scaled_columns": st.session_state.get("scaled_columns", []),
            "fe_operations": st.session_state.get("fe_operations", []),
            "test_score": best.get("test_score"),
            "train_score": best.get("train_score"),
        }
        buffer = io.BytesIO()
        pickle.dump(export_data, buffer)
        buffer.seek(0)

        model_filename = f"modele_{best['name'].replace(' ', '_').lower()}.mlmodel"
        st.download_button(
            label="📥 Télécharger le modèle (.mlmodel)",
            data=buffer,
            file_name=model_filename,
            mime="application/octet-stream",
            type="primary",
        )


def _afficher_entrainement_ts():
    """Sous-routine pour l'entraînement de séries temporelles (ARIMA / SARIMA)."""
    ts_series = st.session_state.get("ts_series")
    dt_col = st.session_state.get("ts_datetime_col")
    val_col = st.session_state.get("ts_value_col")

    if ts_series is None:
        df = st.session_state.get("df_courant")
        if df is not None and dt_col and val_col:
            try:
                ts_series = prepare_timeseries(df, dt_col, val_col)
                st.session_state["ts_series"] = ts_series
            except Exception as e:
                st.error(f"❌ Impossible de préparer la série : {e}")
                return
        else:
            st.warning("⚠️ Série temporelle non définie. Retournez au diagnostic.")
            return

    # Rappel : test = dernières données (split chronologique)
    st.info("📌 Le découpage train/test est **chronologique** : "
            "le test correspond toujours aux **dernières données** de la série.")

    st.subheader("1. Configuration du modèle")

    # Détection de saisonnalité
    seasonal = st.session_state.get("ts_seasonality")
    if seasonal is None:
        with st.spinner("Détection de la saisonnalité…"):
            seasonal = detect_seasonality(ts_series)
            st.session_state["ts_seasonality"] = seasonal

    # Choix ARIMA vs SARIMA
    has_season = seasonal.get("has_seasonality", False)
    season_period = seasonal.get("period") or 0
    # SARIMA avec m > 52 est trop lent (matrices énormes) — on propose
    # un sous-multiple réaliste ou des features calendaires.
    sarima_feasible = has_season and season_period <= 52

    if has_season and sarima_feasible:
        st.success(f"🌊 Saisonnalité détectée (période={season_period}, "
                   f"force={seasonal['strength']:.1%}). SARIMA recommandé.")
        default_model = "SARIMA (avec saisonnalité)"
    elif has_season and not sarima_feasible:
        st.success(f"🌊 Saisonnalité détectée (période={season_period}, "
                   f"force={seasonal['strength']:.1%}).")
        st.warning(
            f"⚠️ La période saisonnière ({season_period}) est trop grande pour "
            f"SARIMA (lent/instable au-delà de ~52). Options :\n\n"
            f"- **ARIMA classique** sur les données — souvent suffisant si la tendance est capturée\n"
            f"- **Sous-échantillonner** en données hebdo/mensuel (étape 5) pour réduire la période\n"
            f"- **Mode horizon** (étape 6d) : ajouter des features calendaires "
            f"(mois, saison) + régression supervisée")
        default_model = "ARIMA (classique)"
    else:
        st.info("Pas de saisonnalité significative détectée. ARIMA classique recommandé.")
        default_model = "ARIMA (classique)"

    model_choice = st.radio("Type de modèle", [
        "ARIMA (classique)",
        "SARIMA (avec saisonnalité)",
    ], index=0 if default_model.startswith("ARIMA") else 1,
        key="ts_model_type")

    use_sarima = model_choice.startswith("SARIMA")

    # Suggestion automatique d'ordre
    suggested = st.session_state.get("ts_suggested_order")
    if suggested is None:
        with st.spinner("Détection automatique de l'ordre ARIMA…"):
            suggested = suggest_arima_order(ts_series)
            st.session_state["ts_suggested_order"] = suggested

    st.info(f"💡 Ordre suggéré automatiquement : **ARIMA{suggested}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        p = st.number_input("p (AR)", 0, 10, int(suggested[0]), key="arima_p",
                            help="Paramètre autorégressif : nombre de valeurs passées utilisées")
    with c2:
        d = st.number_input("d (Intégration)", 0, 3, int(suggested[1]), key="arima_d",
                            help="Différenciation : 0 ou 1 pour enlever la tendance")
    with c3:
        q = st.number_input("q (MA)", 0, 10, int(suggested[2]), key="arima_q",
                            help="Moyenne mobile : nombre de résidus passés utilisés")
    order = (p, d, q)

    # Paramètres saisonniers SARIMA
    seasonal_order = None
    if use_sarima:
        st.markdown("#### Composante saisonnière (P, D, Q, m)")
        default_m = seasonal.get("period", 12) or 12
        # Plafonner le défaut à 52 max pour éviter les calculs interminables
        if default_m > 52:
            st.warning(f"⚠️ La période détectée ({default_m}) est réduite à 52 max "
                       f"pour SARIMA. Pour la saisonnalité longue, préférez le mode horizon.")
            default_m = min(default_m, 52)
        cs1, cs2, cs3, cs4 = st.columns(4)
        with cs1:
            P = st.number_input("P (AR saisonnier)", 0, 5, 1, key="sarima_P",
                                help="Composante AR saisonnière : 1 suffit généralement")
        with cs2:
            D = st.number_input("D (Diff. saisonnière)", 0, 2, 1, key="sarima_D",
                                help="Différenciation saisonnière : 0=pas, 1=idéal")
        with cs3:
            Q = st.number_input("Q (MA saisonnier)", 0, 5, 1, key="sarima_Q",
                                help="Composante MA saisonnière : 0 ou 1")
        with cs4:
            m = st.number_input("m (période)", 2, 52, int(default_m), key="sarima_m",
                                help="Période saisonnière (12=mensuel, 4=trimestriel, 52=hebdo…)")
        seasonal_order = (P, D, Q, m)

    train_ratio = st.slider("Part d'entraînement (%)", 60, 90, 80, step=5,
                             key="ts_train_pct",
                             help="Pourcentage des données pour l'entraînement (80% conseillé)") / 100

    # Affichage du split chronologique
    n_total = len(ts_series.dropna())
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    split_date = ts_series.dropna().index[n_train] if n_train < n_total else None
    if split_date:
        st.caption(f"🔀 Split : Train = {n_train} pts (avant {split_date:%Y-%m-%d}) | "
                   f"Test = {n_test} pts (après {split_date:%Y-%m-%d}) — **dernières données**")

    st.divider()

    # ── Entraînement ──
    model_label = f"SARIMA{order}×{seasonal_order}" if use_sarima else f"ARIMA{order}"
    st.subheader(f"2. Entraînement {model_label}")

    if st.button(f"🚀 Entraîner {model_label}", type="primary", key="train_arima"):
        with st.spinner(f"Entraînement {model_label}…"):
            if use_sarima:
                result = fit_sarima(ts_series, order=order,
                                     seasonal_order=seasonal_order,
                                     train_ratio=train_ratio)
            else:
                result = fit_arima(ts_series, order=order, train_ratio=train_ratio)

        if "error" in result:
            st.error(f"❌ Erreur : {result['error']}")
        else:
            st.session_state["ts_arima_result"] = result
            st.success(f"✅ {model_label} entraîné !")

    result = st.session_state.get("ts_arima_result")
    if result and "error" not in result:
        # Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{result['mae']:.4f}")
        c2.metric("RMSE", f"{result['rmse']:.4f}")
        c3.metric("MAPE", f"{result['mape']:.1f}%")
        c4.metric("AIC", f"{result['aic']:.0f}")

        # Graphique train/test/prévision
        st.pyplot(result["figure"])
        plt.close()

        st.divider()

        # ── Grid Search optionnel (ARIMA uniquement) ──
        if not result.get("is_sarima"):
            st.subheader("3. Recherche de grille (optionnel)")
            if st.button("🔍 Rechercher le meilleur ordre ARIMA", key="arima_grid"):
                with st.spinner("Grid search ARIMA en cours…"):
                    grid_results = arima_grid_search(
                        ts_series,
                        p_range=range(0, 4), d_range=range(0, 3), q_range=range(0, 4),
                        train_ratio=train_ratio,
                    )
                if grid_results:
                    st.session_state["ts_grid_results"] = grid_results
                    st.success(f"✅ {len(grid_results)} configurations testées !")

            grid_results = st.session_state.get("ts_grid_results")
            if grid_results:
                df_grid = pd.DataFrame(grid_results[:10])
                df_grid["order"] = df_grid["order"].astype(str)
                st.dataframe(df_grid, use_container_width=True)

                best_order = grid_results[0]["order"]
                st.info(f"💡 Meilleur ordre par AIC : **ARIMA{best_order}**")

        # ── Validation ──
        st.divider()
        is_sarima = result.get("is_sarima", False)
        result_label = (f"SARIMA{result['order']}×{result['seasonal_order']}"
                        if is_sarima else f"ARIMA{result['order']}")

        if st.button(f"✅ Valider le modèle {result_label}", type="primary",
                      key="validate_arima"):
            arima_model = result["model"]
            model_info = {
                "name": result_label,
                "model": arima_model,
                "test_score": -result["mae"],
                "train_score": None,
                "order": result["order"],
                "mae": result["mae"],
                "rmse": result["rmse"],
                "mape": result["mape"],
                "aic": result["aic"],
            }
            if is_sarima:
                model_info["seasonal_order"] = result["seasonal_order"]

            st.session_state["meilleur_modele"] = model_info
            st.session_state["entrainement_done"] = True

            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["modele"] = {
                    "nom": result_label,
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "mape": result["mape"],
                    "aic": result["aic"],
                }
                rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 8)
                ajouter_historique(rapport, f"Modèle validé : {result_label}")
                sauvegarder_rapport(rapport)
                sauvegarder_modele(rapport, arima_model, "arima_model.joblib")

            st.session_state["_pending_step"] = 8
            st.rerun()
