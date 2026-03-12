# -*- coding: utf-8 -*-
"""
m6_prediction.py — Module 6 : Optimisation & Prédiction.

Étape 9 : Optimisation (GridSearch / RandomSearch) + Prédiction sur nouvelles données.
3 modes d'inférence : Upload CSV, saisie manuelle, série temporelle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import pickle
import joblib
from datetime import datetime

from config import DEFAULT_CV_FOLDS, OPTIMIZATION_DEFAULT_ITERATIONS
from src.models import optimize_model, DEFAULT_PARAM_GRIDS
from src.timeseries import forecast_future, prepare_timeseries, fit_arima
from utils.model_utils import replay_pipeline
from utils.projet_manager import (
    sauvegarder_rapport, sauvegarder_modele, sauvegarder_csv,
    ajouter_historique,
)


def _appliquer_preprocessing_prediction(df_new: pd.DataFrame) -> pd.DataFrame:
    """Applique les mêmes transformations que le pipeline au nouveau dataframe."""
    session = {
        "encoders": st.session_state.get("encoders", {}),
        "scaler": st.session_state.get("scaler"),
        "scaled_columns": st.session_state.get("scaled_columns", []),
        "fe_operations": st.session_state.get("fe_operations", []),
    }
    df_new, _transforms = replay_pipeline(df_new, session)
    return df_new


def _generer_grille_affinee(best_val, default_values):
    """Génère une grille de valeurs affinée autour de la meilleure valeur trouvée."""
    if isinstance(best_val, int):
        step = max(1, best_val // 4)
        refined = sorted(set([
            max(1, best_val - 2 * step),
            max(1, best_val - step),
            best_val,
            best_val + step,
            best_val + 2 * step,
        ]))
        return refined
    elif isinstance(best_val, float):
        if best_val == 0:
            return default_values
        refined = sorted(set([
            round(best_val * 0.1, 6),
            round(best_val * 0.5, 6),
            round(best_val, 6),
            round(best_val * 2, 6),
            round(best_val * 5, 6),
        ]))
        return refined
    return default_values


def _parser_valeurs_grille(text: str, fallback: list) -> list:
    """Parse une chaîne de valeurs séparées par des virgules en liste typée."""
    if not text.strip():
        return fallback
    parts = [p.strip() for p in text.split(",") if p.strip()]
    result = []
    for p in parts:
        if p.lower() == "none":
            result.append(None)
        elif p.lower() in ("true", "false"):
            result.append(p.lower() == "true")
        else:
            try:
                if "." in p:
                    result.append(float(p))
                else:
                    result.append(int(p))
            except ValueError:
                result.append(p)  # string (ex: "rbf", "linear")
    return result if result else fallback


def afficher_optimisation_prediction():
    """Étape 9 — Optimisation & Prédiction."""
    best = st.session_state.get("meilleur_modele")
    if not best or not st.session_state.get("evaluation_done"):
        st.info("🔒 **Verrouillé** — Validez d'abord l'évaluation (étape 8).")
        return

    model = best.get("model")
    model_name = best.get("name", "?")
    problem_type = st.session_state.get("problem_type", "Régression")

    with st.expander("🎓 Optimisation & Prédiction : comment ça marche ?", expanded=False):
        st.markdown("""
**Optimisation** = trouver les meilleurs **réglages** (hyperparamètres) du modèle.

Chaque modèle a des "boutons" à régler, comme un appareil photo :
- Trop de profondeur → le modèle sur-apprend (mémorise au lieu de comprendre)
- Pas assez → il sous-apprend (trop simple pour capter les patterns)

L'app teste automatiquement des centaines de combinaisons et garde la meilleure.

---

**Prédiction** = utiliser le modèle entraîné sur de **nouvelles données**.
- **Upload CSV** : chargez un fichier avec les mêmes colonnes (sans la cible)
- **Saisie manuelle** : entrez les valeurs une par une

> **💡** Si le score baisse après optimisation, l'app garde les paramètres d'origine.
""")

    # Résumé du modèle actuel (visible immédiatement au chargement)
    st.markdown(f"**Modèle actuel :** `{model_name}` — "
                f"Score test : **{best.get('test_score', 0):.4f}**")
    if best.get("best_params"):
        with st.expander("📋 Paramètres optimisés retenus", expanded=False):
            st.json(best["best_params"])

    # ═══════════════════════════════════════
    # Parcours Série temporelle
    # ═══════════════════════════════════════
    if problem_type == "Série temporelle":
        _afficher_prediction_ts(best, model_name)
        return

    tab_opt, tab_pred, tab_api = st.tabs([
        "⚙️ Optimisation", "🔮 Prédiction", "🚀 Déployer API"
    ])

    # ═══════════════════════════════════════
    # Onglet 1 : Optimisation
    # ═══════════════════════════════════════
    with tab_opt:
        st.subheader("Optimisation des hyperparamètres")

        X_train = st.session_state.get("X_train")
        y_train = st.session_state.get("y_train")
        X_test = st.session_state.get("X_test")
        y_test = st.session_state.get("y_test")

        if X_train is None:
            st.warning("Données d'entraînement non disponibles.")
            return

        # ── Déterminer la grille de paramètres ──
        # Si on a déjà optimisé, proposer une grille affinée
        prev_opt = st.session_state.get("opt_result")
        prev_params = prev_opt.get("best_params", {}) if prev_opt else {}
        base_grid = DEFAULT_PARAM_GRIDS.get(model_name.replace(" (optimisé)", ""), {})

        if not base_grid:
            st.info(f"Pas de grille de paramètres prédéfinie pour « {model_name} ».")
            st.caption("Vous pouvez passer directement à la prédiction.")
        else:
            # ── Grille éditable ──
            st.markdown("##### 📋 Grille de paramètres")
            if prev_params:
                st.info("💡 Grille pré-remplie autour des meilleurs paramètres "
                        "de la dernière optimisation. Modifiez les valeurs pour affiner.")

            param_grid = {}
            for param_name, default_values in base_grid.items():
                best_val = prev_params.get(param_name)

                # Générer une grille affinée autour de la meilleure valeur
                if best_val is not None and isinstance(best_val, (int, float)):
                    refined = _generer_grille_affinee(best_val, default_values)
                    default_text = ", ".join(str(v) for v in refined)
                else:
                    default_text = ", ".join(str(v) for v in default_values)

                user_input = st.text_input(
                    f"**{param_name}**"
                    + (f"  *(meilleur précédent : {best_val})*" if best_val is not None else ""),
                    value=default_text,
                    key=f"grid_{param_name}",
                    help="Séparez les valeurs par des virgules. "
                         "Exemples : 0.01, 0.1, 1  ou  rbf, linear  ou  None")

                param_grid[param_name] = _parser_valeurs_grille(user_input, default_values)

            c1, c2 = st.columns(2)
            with c1:
                method = st.radio("Méthode", ["Grid Search (exhaustif)",
                                                "Random Search (rapide)"],
                                   key="opt_method")
            with c2:
                cv = st.slider("Folds CV", 3, 10, DEFAULT_CV_FOLDS, key="opt_cv",
                               help="Nombre de folds pour la validation croisée pendant l'optimisation")
                if "Random" in method:
                    n_iter = st.slider("Itérations", 10, 200,
                                        OPTIMIZATION_DEFAULT_ITERATIONS, key="opt_iter",
                                        help="Nombre de combinaisons aléatoires à tester (plus = meilleur mais lent)")
                else:
                    n_iter = 50

            # Nombre de combinaisons
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)
            st.caption(f"🔢 {n_combos} combinaison(s) à tester")

            if st.button("🚀 Lancer l'optimisation", type="primary", key="opt_btn"):
                with st.spinner("Optimisation en cours…"):
                    opt_method = "grid" if "Grid" in method else "random"
                    try:
                        result = optimize_model(
                            model, X_train, y_train, param_grid,
                            method=opt_method, n_iter=n_iter, cv=cv,
                            problem_type=problem_type,
                        )

                        st.session_state["opt_result"] = result
                        st.success("✅ Optimisation terminée !")

                    except Exception as e:
                        st.error(f"❌ Erreur d'optimisation : {e}")

            # ── Afficher les résultats ──
            result = st.session_state.get("opt_result")
            if result:
                st.divider()
                st.subheader("📊 Résultats de l'optimisation")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Score avant", f"{best.get('test_score', 0):.4f}")
                with c2:
                    new_score = result.get("best_score", 0)
                    st.metric("Meilleur score CV", f"{new_score:.4f}")
                with c3:
                    new_model = result.get("best_model", model)
                    new_test_score = float(new_model.score(X_test, y_test))
                    delta = new_test_score - best.get("test_score", 0)
                    st.metric("Score test réel", f"{new_test_score:.4f}",
                              delta=f"{delta:+.4f}")

                st.markdown("**Meilleurs paramètres :**")
                st.json(result.get("best_params", {}))

                # Top 5 des combinaisons
                cv_results = result.get("cv_results")
                if cv_results is not None and len(cv_results) > 0:
                    with st.expander("📋 Top combinaisons testées", expanded=False):
                        cols_show = [c for c in cv_results.columns
                                     if c.startswith("param_") or c in
                                     ("mean_test_score", "std_test_score", "rank_test_score")]
                        st.dataframe(cv_results[cols_show].head(10),
                                     use_container_width=True)

                # Boutons d'action
                c_adopt, c_redo = st.columns(2)
                with c_adopt:
                    if st.button("✅ Adopter le modèle optimisé", key="adopt_opt"):
                        base_name = model_name.replace(" (optimisé)", "")
                        opt_name = f"{base_name} (optimisé)"
                        save_name = f"{base_name}_optimise"

                        st.session_state["meilleur_modele"] = {
                            **best,
                            "model": new_model,
                            "test_score": new_test_score,
                            "name": opt_name,
                            "best_params": result.get("best_params"),
                        }

                        rapport = st.session_state.get("rapport", {})
                        if rapport:
                            rapport["modele"] = {
                                "nom": save_name,
                                "score_train": best.get("train_score"),
                                "score_test": new_test_score,
                                "best_params": result.get("best_params"),
                            }
                            rapport["optimisation"] = {
                                "best_params": result.get("best_params"),
                                "best_score_cv": new_score,
                                "score_test_apres": new_test_score,
                            }
                            rapport["etape_courante"] = max(
                                rapport.get("etape_courante", 0), 9)
                            ajouter_historique(rapport, "Modèle optimisé adopté")
                            sauvegarder_modele(rapport, new_model,
                                                f"{save_name}.joblib")
                            sauvegarder_rapport(rapport)

                        st.success("✅ Modèle optimisé adopté !")
                        st.rerun()

                with c_redo:
                    if st.button("🔄 Relancer avec grille affinée", key="redo_opt"):
                        # Garder le résultat pour pré-remplir la grille, forcer rerun
                        st.rerun()

        # ── Seuil de classification ──
        if problem_type == "Classification" and hasattr(model, "predict_proba"):
            st.divider()
            st.subheader("🎚️ Ajustement du seuil de décision")
            st.caption("Par défaut le seuil est 0.5. Ajustez-le pour équilibrer "
                       "précision et rappel selon votre besoin.")

            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                threshold = st.slider("Seuil", 0.1, 0.9, 0.5, 0.05,
                                       key="threshold_slider")
                y_pred_thresh = (y_proba[:, 1] >= threshold).astype(int)

                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred_thresh):.3f}")
                c2.metric("F1", f"{f1_score(y_test, y_pred_thresh, zero_division=0):.3f}")
                c3.metric("Precision", f"{precision_score(y_test, y_pred_thresh, zero_division=0):.3f}")
                c4.metric("Recall", f"{recall_score(y_test, y_pred_thresh, zero_division=0):.3f}")
            else:
                st.info("Ajustement du seuil disponible uniquement pour la classification binaire.")

    # ═══════════════════════════════════════
    # Onglet 2 : Prédiction
    # ═══════════════════════════════════════
    with tab_pred:
        st.subheader("Prédire sur de nouvelles données")

        model = st.session_state["meilleur_modele"].get("model")
        feature_cols = st.session_state.get("feature_cols_used", [])

        if model is None:
            st.warning("Aucun modèle disponible.")
            return

        mode = st.radio("Mode d'entrée",
                         ["📤 Upload CSV", "✏️ Saisie manuelle"],
                         key="pred_mode", horizontal=True)

        # ─── Upload CSV ───
        if mode == "📤 Upload CSV":
            uploaded = st.file_uploader("Fichier CSV de prédiction",
                                         type=["csv"], key="pred_upload")
            if uploaded:
                try:
                    df_new = pd.read_csv(uploaded)
                    st.dataframe(df_new.head(), use_container_width=True)

                    missing_cols = [c for c in feature_cols if c not in df_new.columns]
                    if missing_cols:
                        st.warning(f"⚠️ Colonnes manquantes : {', '.join(missing_cols)}")
                    else:
                        if st.button("🔮 Prédire", type="primary", key="pred_csv"):
                            try:
                                df_prep = _appliquer_preprocessing_prediction(df_new)
                                X_new = df_prep[feature_cols]
                                preds = model.predict(X_new)
                                df_new["prédiction"] = preds

                                if hasattr(model, "predict_proba"):
                                    probas = model.predict_proba(X_new)
                                    if probas.shape[1] == 2:
                                        df_new["probabilité"] = probas[:, 1]

                                st.success(f"✅ {len(preds)} prédictions générées !")
                                st.dataframe(df_new, use_container_width=True)

                                # Téléchargement
                                csv_bytes = df_new.to_csv(index=False).encode("utf-8")
                                st.download_button("📥 Télécharger les résultats",
                                                    csv_bytes, "predictions.csv",
                                                    "text/csv", key="dl_pred")

                                rapport = st.session_state.get("rapport", {})
                                if rapport:
                                    sauvegarder_csv(rapport, df_new, "predictions.csv")
                                    ajouter_historique(rapport,
                                                       f"{len(preds)} prédictions (CSV)")
                                    sauvegarder_rapport(rapport)

                            except Exception as e:
                                st.error(f"❌ Erreur : {e}")

                except Exception as e:
                    st.error(f"❌ Impossible de lire le fichier : {e}")

        # ─── Saisie manuelle ───
        elif mode == "✏️ Saisie manuelle":
            st.markdown("Renseignez les valeurs pour chaque variable :")

            X_train = st.session_state.get("X_train")
            input_values = {}

            for i, col in enumerate(feature_cols):
                # X_train peut être un ndarray ou un DataFrame
                col_data = None
                if X_train is not None:
                    if hasattr(X_train, "columns") and col in X_train.columns:
                        col_data = X_train[col]
                    elif isinstance(X_train, np.ndarray) and i < X_train.shape[1]:
                        col_data = pd.Series(X_train[:, i], name=col)

                if col_data is not None:
                    if pd.api.types.is_numeric_dtype(col_data):
                        min_v = float(col_data.min())
                        max_v = float(col_data.max())
                        mean_v = float(col_data.mean())
                        input_values[col] = st.number_input(
                            f"{col} (min={min_v:.2f}, max={max_v:.2f})",
                            value=mean_v, key=f"manual_{col}")
                    else:
                        uniques = col_data.unique().tolist()
                        if len(uniques) <= 50:
                            input_values[col] = st.selectbox(
                                col, uniques, key=f"manual_{col}")
                        else:
                            input_values[col] = st.text_input(
                                col, key=f"manual_{col}")
                else:
                    input_values[col] = st.text_input(col, key=f"manual_{col}")

            if st.button("🔮 Prédire", type="primary", key="pred_manual"):
                try:
                    df_single = pd.DataFrame([input_values])
                    df_prep = _appliquer_preprocessing_prediction(df_single)
                    X_new = df_prep[feature_cols]
                    pred = model.predict(X_new)[0]

                    st.success(f"🎯 **Prédiction : {pred}**")

                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(X_new)[0]
                        st.markdown("**Probabilités :**")
                        for i, p in enumerate(probas):
                            st.write(f"  Classe {i} : {p:.4f}")

                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

    # ═══════════════════════════════════════
    # Onglet 3 : Déployer comme API
    # ═══════════════════════════════════════
    with tab_api:
        _afficher_export_api(best, model_name, problem_type)

    # ═══════════════════════════════════════
    # Sauvegarde définitive du modèle final
    # ═══════════════════════════════════════
    st.divider()
    st.subheader("💾 Enregistrer le modèle final")
    st.caption("Sauvegardez le modèle actuel (optimisé ou non) de façon définitive "
               "dans le dossier du projet. Il sera restauré automatiquement au prochain chargement.")

    if st.button("💾 Enregistrer définitivement", type="primary", key="save_final"):
        rapport = st.session_state.get("rapport", {})
        current_best = st.session_state.get("meilleur_modele", {})
        if rapport and current_best.get("model"):
            base_name = current_best["name"].replace(" (optimisé)", "")
            save_name = f"{base_name}_optimise" if "(optimisé)" in current_best["name"] \
                else current_best["name"]

            rapport["modele"] = {
                "nom": save_name,
                "score_train": current_best.get("train_score"),
                "score_test": current_best.get("test_score"),
                "best_params": current_best.get("best_params"),
            }
            rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 9)
            sauvegarder_modele(rapport, current_best["model"], f"{save_name}.joblib")

            # Re-sauvegarder les splits
            from modules.m4_entrainement import _sauvegarder_splits
            _sauvegarder_splits(rapport)

            ajouter_historique(rapport, f"Modèle final enregistré : {current_best['name']}")
            sauvegarder_rapport(rapport)

            st.success(f"✅ Modèle « {current_best['name']} » enregistré définitivement !\n\n"
                       f"Il sera restauré automatiquement au prochain chargement du projet.")
        else:
            st.warning("⚠️ Aucun modèle à sauvegarder.")

    # ═══════════════════════════════════════
    # Lien vers la page autonome de prédiction
    # ═══════════════════════════════════════
    st.divider()
    st.subheader("🔗 Prédiction autonome (lien partageable)")
    st.markdown(
        "Partagez ce lien pour permettre à d'autres utilisateurs de faire "
        "des prédictions **sans accéder au pipeline** — ils n'ont besoin que "
        "du fichier modèle `.pkl` et de leurs propres données."
    )
    st.markdown(
        '<a href="/Prediction" target="_blank">'
        '🔮 <strong>Ouvrir la page de prédiction autonome</strong></a>',
        unsafe_allow_html=True,
    )
    st.caption(
        "💡 L'URL complète est `http://<votre-machine>:8501/Prediction` — "
        "envoyez-la directement aux utilisateurs finaux."
    )


def _afficher_export_api(best: dict, model_name: str, problem_type: str):
    """Onglet Déployer API : export du modèle + guide Cloud Run + code AppScript."""
    st.subheader("🚀 Déployer le modèle comme API")
    st.markdown(
        "Exportez votre modèle pour le rendre accessible depuis **Google Sheets**, "
        "**AppScript**, ou toute autre application via une API REST."
    )

    model = best.get("model")
    feature_cols = st.session_state.get("feature_cols_used",
                                         st.session_state.get("feature_cols", []))
    target_col = st.session_state.get("target_col", "cible")
    scaler = st.session_state.get("scaler")
    rapport = st.session_state.get("rapport", {})
    project_name = rapport.get("nom", "mon_projet")

    # Générer un ID propre pour le modèle
    model_id = project_name.replace(" ", "_").lower()
    model_id = st.text_input("Identifiant du modèle (sans espaces)",
                              value=model_id, key="api_model_id",
                              help="Cet ID sera utilisé dans les appels API")

    if not model:
        st.warning("Aucun modèle disponible.")
        return

    # ── Déterminer les types de features ──
    X_train = st.session_state.get("X_train")
    feature_types = {}
    if X_train is not None:
        if hasattr(X_train, "dtypes"):
            for col in feature_cols:
                if col in X_train.columns:
                    dt = str(X_train[col].dtype)
                    feature_types[col] = "number" if "float" in dt or "int" in dt else "string"
        else:
            for col in feature_cols:
                feature_types[col] = "number"

    # ── Construire le package d'export ──
    st.divider()
    st.markdown("### 📦 1. Exporter le modèle")

    if st.button("📦 Générer le package API", type="primary", key="export_api_btn"):
        try:
            # Métadonnées
            metadata = {
                "project_name": project_name,
                "model_id": model_id,
                "model_name": model_name.replace(" (optimisé)", ""),
                "problem_type": problem_type,
                "target_name": target_col,
                "feature_names": feature_cols,
                "feature_types": feature_types,
                "test_score": best.get("test_score"),
                "train_score": best.get("train_score"),
                "best_params": best.get("best_params"),
                "created_at": datetime.now().isoformat(),
            }

            # Créer un ZIP en mémoire
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # model.joblib
                model_buf = io.BytesIO()
                joblib.dump(model, model_buf)
                zf.writestr(f"{model_id}/model.joblib", model_buf.getvalue())

                # scaler.joblib (si présent)
                if scaler is not None:
                    scaler_buf = io.BytesIO()
                    joblib.dump(scaler, scaler_buf)
                    zf.writestr(f"{model_id}/scaler.joblib", scaler_buf.getvalue())

                # metadata.json
                zf.writestr(f"{model_id}/metadata.json",
                            json.dumps(metadata, indent=2, ensure_ascii=False))

            buf.seek(0)
            st.session_state["_api_package"] = buf.getvalue()
            st.session_state["_api_metadata"] = metadata
            st.success("✅ Package généré !")

        except Exception as e:
            st.error(f"❌ Erreur : {e}")

    # Bouton de téléchargement
    pkg = st.session_state.get("_api_package")
    meta = st.session_state.get("_api_metadata")
    if pkg:
        st.download_button(
            "📥 Télécharger le package API",
            pkg,
            f"{model_id}_api.zip",
            "application/zip",
            key="dl_api_pkg",
        )
        st.caption(
            f"Contient : `{model_id}/model.joblib` + `metadata.json`"
            + (" + `scaler.joblib`" if scaler is not None else "")
        )

    # ── Guide de déploiement ──
    st.divider()
    st.markdown("### ☁️ 2. Déployer sur Google Cloud Run")

    with st.expander("📖 Guide pas à pas", expanded=False):
        st.markdown(f"""
**Prérequis** : un compte Google Cloud (gratuit : 2M requêtes/mois).

**Étape 1** — Préparez les fichiers :
```
mon-api/
├── api/
│   ├── main.py          ← (déjà dans le repo ML Studio)
│   ├── requirements.txt  ← (déjà dans le repo)
│   ├── __init__.py
│   └── models/
│       └── {model_id}/
│           ├── model.joblib    ← (du ZIP téléchargé)
│           ├── metadata.json
│           └── scaler.joblib   (si présent)
├── Dockerfile            ← (api/Dockerfile du repo)
```

**Étape 2** — Déployez en une commande :
```bash
# Installer Google Cloud CLI si pas fait
# https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project VOTRE_PROJET_GCP

# Construire et déployer
gcloud run deploy ml-studio-api \\
    --source . \\
    --region europe-west1 \\
    --allow-unauthenticated \\
    --memory 512Mi \\
    --port 8080
```

**Étape 3** — Notez l'URL affichée (ex: `https://ml-studio-api-xxxx-ew.a.run.app`)

**Ajouter un nouveau modèle** : dézipper le package dans `api/models/` et redéployer.
""")

    # ── Code AppScript ──
    st.divider()
    st.markdown("### 📋 3. Code AppScript pour Google Sheets")

    api_url = st.text_input(
        "URL de votre API Cloud Run",
        value="https://ml-studio-api-XXXXX-ew.a.run.app",
        key="api_url_input",
        help="Collez ici l'URL fournie par Cloud Run après le déploiement",
    )

    if meta:
        features_obj = ", ".join(
            f'"{f}": row[{i}]' for i, f in enumerate(meta.get("feature_names", []))
        )
        features_header = ", ".join(
            f'"{f}"' for f in meta.get("feature_names", [])
        )
        n_feats = len(meta.get("feature_names", []))

        appscript_code = f'''// ═══════════════════════════════════════════════
// ML Studio — Prédiction depuis Google Sheets
// Modèle : {meta.get("project_name", model_id)}
// ═══════════════════════════════════════════════

var API_URL = "{api_url}";
var MODEL_ID = "{model_id}";

/**
 * Prédit une valeur à partir d'une ligne de données.
 * Utilisation dans une cellule :
 *   =PREDICT(A2:{"ABCDEFGHIJKLMNOPQRSTUVWXYZ"[n_feats-1] if n_feats <= 26 else "Z"}2)
 *
 * Colonnes attendues : {", ".join(meta.get("feature_names", []))}
 */
function PREDICT(range) {{
  var row = Array.isArray(range[0]) ? range[0] : range;
  var payload = {{
    "model": MODEL_ID,
    "features": {{ {features_obj} }}
  }};
  var options = {{
    "method": "post",
    "contentType": "application/json",
    "payload": JSON.stringify(payload),
    "muteHttpExceptions": true
  }};
  var response = UrlFetchApp.fetch(API_URL + "/predict", options);
  var result = JSON.parse(response.getContentText());
  if (result.error) throw new Error(result.detail);
  return result.prediction;
}}


/**
 * Prédit en lot pour une plage de données.
 * Usage : sélectionnez la plage de données, le script remplit la colonne résultat.
 *
 * Menu : ML Studio > Prédire la sélection
 */
function predictSelection() {{
  var sheet = SpreadsheetApp.getActiveSheet();
  var range = sheet.getActiveRange();
  var data = range.getValues();

  var features = [{features_header}];
  var rows = data.map(function(row) {{
    var obj = {{}};
    for (var i = 0; i < features.length; i++) {{
      obj[features[i]] = row[i];
    }}
    return obj;
  }});

  var payload = {{
    "model": MODEL_ID,
    "data": rows
  }};
  var options = {{
    "method": "post",
    "contentType": "application/json",
    "payload": JSON.stringify(payload),
    "muteHttpExceptions": true
  }};

  var response = UrlFetchApp.fetch(API_URL + "/predict/batch", options);
  var result = JSON.parse(response.getContentText());

  // Écrire les résultats dans la colonne suivante
  var startRow = range.getRow();
  var startCol = range.getColumn() + range.getNumColumns();
  sheet.getRange(startRow, startCol).setValue("Prédiction");
  for (var i = 0; i < result.predictions.length; i++) {{
    sheet.getRange(startRow + 1 + i, startCol).setValue(result.predictions[i]);
  }}
  SpreadsheetApp.getUi().alert(
    "✅ " + result.count + " prédictions générées !"
  );
}}


/**
 * Ajoute un menu personnalisé dans Google Sheets.
 */
function onOpen() {{
  SpreadsheetApp.getUi()
    .createMenu("🔮 ML Studio")
    .addItem("Prédire la sélection", "predictSelection")
    .addToUi();
}}
'''
        st.code(appscript_code, language="javascript")

        # Bouton copier
        st.download_button(
            "📋 Télécharger le code AppScript",
            appscript_code.encode("utf-8"),
            f"ml_studio_{model_id}.gs",
            "text/plain",
            key="dl_appscript",
        )

        st.info(
            "**Comment l'utiliser :**\n"
            "1. Dans Google Sheets → Extensions → Apps Script\n"
            "2. Collez le code ci-dessus\n"
            "3. Sauvegardez et fermez l'éditeur\n"
            "4. Rechargez la feuille → menu **ML Studio** apparaît\n"
            "5. Utilisez `=PREDICT(A2:X2)` dans une cellule ou le menu pour prédire en lot"
        )
    else:
        st.caption("⬆️ Générez d'abord le package API pour voir le code AppScript.")


def _afficher_prediction_ts(best: dict, model_name: str):
    """Prédiction / prévision pour les séries temporelles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts_series = st.session_state.get("ts_series")
    order = best.get("order")

    if ts_series is None:
        df = st.session_state.get("df_courant")
        dt_col = st.session_state.get("ts_datetime_col")
        val_col = st.session_state.get("ts_value_col")
        if df is not None and dt_col and val_col:
            try:
                ts_series = prepare_timeseries(df, dt_col, val_col)
                st.session_state["ts_series"] = ts_series
            except Exception:
                pass

    if ts_series is None or order is None:
        st.warning("⚠️ Série temporelle ou ordre ARIMA non disponible.")
        return

    st.subheader(f"📈 Prévision — {model_name}")

    steps = st.slider("Horizon de prévision (nombres de pas)", 1, 365, 30,
                       key="ts_forecast_steps",
                       help="Nombre de périodes à prédire (jours, mois… selon vos données)")

    if st.button("🔮 Générer la prévision", type="primary", key="forecast_btn"):
        with st.spinner(f"Prévision sur {steps} pas…"):
            result = forecast_future(ts_series, order, steps=steps)

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.session_state["ts_forecast_result"] = result
            st.success(f"✅ Prévision sur {steps} pas générée !")

    result = st.session_state.get("ts_forecast_result")
    if result and "error" not in result:
        # Graphique avec intervalle de confiance
        st.pyplot(result["figure"])
        plt.close()

        # Tableau des prévisions
        with st.expander("📋 Tableau des prévisions"):
            df_forecast = pd.DataFrame({
                "Date": result["forecast"].index,
                "Prévision": result["forecast"].values,
                "Borne basse (95%)": result["conf_int"].iloc[:, 0].values,
                "Borne haute (95%)": result["conf_int"].iloc[:, 1].values,
            })
            st.dataframe(df_forecast, use_container_width=True)

            # Téléchargement
            csv_bytes = df_forecast.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger les prévisions", csv_bytes,
                                "previsions_ts.csv", "text/csv",
                                key="dl_forecast")

            rapport = st.session_state.get("rapport", {})
            if rapport:
                sauvegarder_csv(rapport, df_forecast, "previsions_ts.csv")
                ajouter_historique(rapport,
                                   f"Prévision {steps} pas — {model_name}")
                sauvegarder_rapport(rapport)
