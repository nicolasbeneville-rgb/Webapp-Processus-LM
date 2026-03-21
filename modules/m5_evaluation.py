# -*- coding: utf-8 -*-
"""
m5_evaluation.py — Module 5 : Évaluation.

Étape 8 : Diagnostic complet du modèle.
- Régression : Réel vs Prédit, résidus, distribution résidus, top erreurs, rapport.
- Classification : matrice confusion, ROC, precision-recall, rapport par classe.
- Feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import (
    plot_real_vs_pred, plot_residuals, plot_residual_distribution,
    get_top_errors, auto_comment_residuals, get_regression_report,
    plot_confusion_matrix, plot_roc_curve, get_classification_report,
    get_misclassified, plot_precision_recall_curve,
    plot_classification_metrics_bar,
    compute_mase,
    plot_real_vs_pred_interactive,
)
from src.guide import interpret_model_results
from src.timeseries import walk_forward_validation, plot_timeseries
from utils.projet_manager import sauvegarder_rapport, ajouter_historique
from modules.aide_contextuelle import afficher_aide_graphique


def _feature_importance_chart(model, feature_names: list):
    """Affiche un diagramme en barres de l'importance des features."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        importances = np.abs(coefs)

    if importances is None:
        st.info("Ce modèle ne fournit pas d'importance des variables.")
        return

    df_imp = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(6, max(3, len(df_imp) * 0.35)))
    fig.set_facecolor("#F8F9FC")
    ax.barh(df_imp["feature"], df_imp["importance"], color="#4F5BD5", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Top features", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _feature_impact_par_classe(model, feature_names: list, y_test):
    """Affiche l'impact positif/négatif des features par classe (dropdown)."""

    # Récupérer les coefficients par classe
    if hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim == 1:
            coefs = coefs.reshape(1, -1)
        classes = model.classes_ if hasattr(model, "classes_") else list(range(coefs.shape[0]))
    else:
        st.info("Ce graphique nécessite un modèle avec des coefficients "
                "(Régression Logistique, SVM linéaire…). "
                "Les modèles à base d'arbres n'ont pas d'impact signé par classe.")
        return

    n_features = min(len(feature_names), coefs.shape[1])
    classes_labels = [str(c) for c in classes]

    selected_class = st.selectbox("Sélectionnez une classe :", classes_labels,
                                   key="impact_class_select")
    class_idx = classes_labels.index(selected_class)

    # Coefficients de la classe sélectionnée
    class_coefs = coefs[class_idx][:n_features]
    df_impact = pd.DataFrame({
        "feature": feature_names[:n_features],
        "coefficient": class_coefs,
    }).sort_values("coefficient")

    # Top 15 (positif + négatif)
    top_neg = df_impact.head(8)
    top_pos = df_impact.tail(8)
    df_show = pd.concat([top_neg, top_pos]).drop_duplicates().sort_values("coefficient")

    # Couleurs : vert = positif, rouge = négatif
    colors = ["#DC2626" if v < 0 else "#059669" for v in df_show["coefficient"]]

    fig, ax = plt.subplots(figsize=(5, max(3, len(df_show) * 0.32)))
    fig.set_facecolor("#F8F9FC")
    ax.barh(df_show["feature"], df_show["coefficient"], color=colors, edgecolor="white")
    ax.axvline(x=0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.set_xlabel("Coefficient (impact)")
    ax.set_title(f"Impact des features → classe « {selected_class} »",
                 fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("🟢 **Positif** = augmente la probabilité de cette classe | "
               "🔴 **Négatif** = diminue la probabilité")


def _permutation_importance_chart(model, X_test, y_test, feature_names: list):
    """Importance par permutation (model-agnostic, fonctionne pour tous les modèles)."""
    from sklearn.inspection import permutation_importance

    with st.spinner("Calcul de l'importance par permutation…"):
        result = permutation_importance(model, X_test, y_test,
                                        n_repeats=10, random_state=42, n_jobs=-1)

    n_features = min(len(feature_names), len(result.importances_mean))
    df_imp = pd.DataFrame({
        "feature": feature_names[:n_features],
        "importance": result.importances_mean[:n_features],
        "std": result.importances_std[:n_features],
    }).sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(5, max(3, len(df_imp) * 0.32)))
    fig.set_facecolor("#F8F9FC")
    colors = ["#4F5BD5" if v > 0 else "#D1D5DB" for v in df_imp["importance"]]
    ax.barh(df_imp["feature"], df_imp["importance"],
            xerr=df_imp["std"], color=colors, edgecolor="white", capsize=3)
    ax.axvline(x=0, color="#9CA3AF", linewidth=1, linestyle="--")
    ax.set_xlabel("Baisse du score quand la feature est mélangée")
    ax.set_title("Permutation importance", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("🟦 Barres hautes = la feature est **cruciale** pour le modèle. "
               "Barres proches de 0 = la feature n'apporte rien.")


def _learning_curve_chart(model, X_train, y_train, problem_type: str):
    """Courbe d'apprentissage : le modèle a-t-il besoin de plus de données ?"""
    from sklearn.model_selection import learning_curve

    scoring = "r2" if problem_type == "Régression" else "accuracy"

    with st.spinner("Calcul de la courbe d'apprentissage…"):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=5, scoring=scoring, n_jobs=-1,
        )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.set_facecolor("#F8F9FC")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#4F5BD5")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.15, color="#059669")
    ax.plot(train_sizes, train_mean, "o-", color="#4F5BD5", linewidth=2,
            label="Train", markersize=4)
    ax.plot(train_sizes, test_mean, "o-", color="#059669", linewidth=2,
            label="Validation", markersize=4)
    ax.set_xlabel("Taille du jeu d'entraînement")
    ax.set_ylabel(scoring.upper())
    ax.set_title("Courbe d'apprentissage", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Diagnostic automatique
    gap = train_mean[-1] - test_mean[-1]
    if gap > 0.1:
        st.warning("⚠️ **Sur-apprentissage** : l'écart train/validation est grand. "
                   "Plus de données ou un modèle plus simple pourrait aider.")
    elif test_mean[-1] < 0.6:
        st.warning("⚠️ **Sous-apprentissage** : les scores sont bas. "
                   "Essayez un modèle plus complexe ou de meilleures features.")
    else:
        st.success("✅ Bonne convergence. Ajouter des données n'améliorerait probablement pas beaucoup.")


def _calibration_curve_chart(model, X_test, y_test):
    """Courbe de calibration : les probabilités prédites sont-elles fiables ?"""
    if not hasattr(model, "predict_proba"):
        st.info("Ce modèle ne supporte pas predict_proba.")
        return

    from sklearn.calibration import calibration_curve

    y_proba = model.predict_proba(X_test)
    if y_proba.shape[1] != 2:
        st.info("Courbe de calibration disponible uniquement en classification binaire.")
        return

    prob_true, prob_pred = calibration_curve(y_test, y_proba[:, 1], n_bins=10)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.set_facecolor("#F8F9FC")
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.5, label="Parfaitement calibré")
    ax.plot(prob_pred, prob_true, "o-", color="#4F5BD5", linewidth=2,
            markersize=5, label=model.__class__.__name__)
    ax.set_xlabel("Probabilité prédite moyenne")
    ax.set_ylabel("Fréquence réelle")
    ax.set_title("Courbe de calibration", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Si la courbe suit la diagonale, les probabilités sont **fiables**. "
               "Au-dessus = le modèle sous-estime. En-dessous = il sur-estime.")


def _qq_plot_residuals(y_test, y_pred):
    """QQ-plot des résidus : vérifier la normalité."""
    from scipy import stats

    residuals = np.array(y_test) - np.array(y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.set_facecolor("#F8F9FC")
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set(color="#4F5BD5", markersize=4, alpha=0.7)
    ax.get_lines()[1].set(color="#DC2626", linewidth=2)
    ax.set_title("QQ-plot des résidus", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Test de Shapiro
    n = min(len(residuals), 5000)
    stat, p_value = stats.shapiro(residuals[:n])
    if p_value > 0.05:
        st.success(f"✅ Résidus normaux (Shapiro p={p_value:.3f}). "
                   "Les hypothèses du modèle linéaire sont respectées.")
    else:
        st.warning(f"⚠️ Résidus non normaux (Shapiro p={p_value:.3f}). "
                   "Le modèle rate peut-être un pattern non-linéaire.")


def afficher_evaluation():
    """Étape 8 — Évaluation détaillée du modèle."""
    best = st.session_state.get("meilleur_modele")
    if not best or not st.session_state.get("entrainement_done"):
        st.info("🔒 **Verrouillé** — Entraînez d'abord un modèle (étape 7).")
        return

    model = best.get("model")
    model_name = best.get("name", "?")
    problem_type = st.session_state.get("problem_type", "Régression")

    with st.expander("🎓 Comment lire l'évaluation ?", expanded=False):
        if problem_type == "Régression":
            st.markdown("""
L'évaluation vérifie **où et comment** le modèle se trompe.

| Graphique | Ce qu'il montre | Ce qu'on cherche |
|---|---|---|
| **Réel vs Prédit** | Points réels vs prédits | Points alignés sur la diagonale = bon |
| **Résidus** | Les erreurs (réel − prédit) | Nuage aléatoire autour de 0 = bon |
| **Distribution** | Comment les erreurs se répartissent | Forme de cloche centrée = bon |
| **Top erreurs** | Les pires prédictions | Identifier les cas problématiques |
| **Importance** | Quelles variables comptent le plus | Vérifier que c'est logique |

> **⚠️ Score Train très supérieur au Score Test ?** C'est du **sur-apprentissage**
> (le modèle a "appris par cœur" au lieu de comprendre). L'écart devrait être < 10%.
""")
        else:
            st.markdown("""
L'évaluation vérifie la qualité des prédictions du modèle de classification.

| Graphique | Ce qu'il montre | Ce qu'on cherche |
|---|---|---|
| **Matrice confusion** | Tableau des bonnes/mauvaises prédictions par classe | Diagonale foncée = bon |
| **Courbe ROC** | Capacité à distinguer les classes | Courbe au-dessus de la diagonale |
| **Precision-Recall** | Équilibre entre "trouver tous les cas" et "ne pas se tromper" | Courbe haute = bon |
| **Importance** | Variables les plus influentes | Vérifier que c'est logique |

> **💡 AUC-ROC** : 1.0 = parfait, 0.5 = au hasard. En dessous de 0.7, le modèle a du mal.
""")

    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    if model is None or X_test is None or y_test is None:
        st.warning("⚠️ Données d'évaluation manquantes.")
        return

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Noms de features (X_test peut être un ndarray après scaling)
    _feature_names = st.session_state.get("feature_cols_used") or st.session_state.get("feature_cols")
    if _feature_names is None:
        if hasattr(X_test, "columns"):
            _feature_names = X_test.columns.tolist()
        else:
            _feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    st.markdown(f"**Modèle évalué :** `{model_name}`")

    # Modèle TS statistique (ARIMA/SARIMA) vs mode horizon (régression supervisée).
    is_ts_stat_model = (
        problem_type == "Série temporelle"
        and not st.session_state.get("ts_horizon_mode")
        and (
            best.get("order") is not None
            or str(model_name).upper().startswith("ARIMA")
            or str(model_name).upper().startswith("SARIMA")
        )
    )

    # ═══════════════════════════════════════
    # Parcours Série temporelle
    # ═══════════════════════════════════════
    if is_ts_stat_model:
        _afficher_evaluation_ts(best, model_name)
        return

    if problem_type == "Détection d'anomalies":
        _afficher_evaluation_anomaly(best, model_name, X_test)
        return

    # Métriques résumées
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Score Train", f"{best.get('train_score', 0):.4f}")
    with c2:
        st.metric("Score Test", f"{best.get('test_score', 0):.4f}")
    with c3:
        train_s = best.get("train_score", 0)
        test_s = best.get("test_score", 0)
        gap = (train_s - test_s) / max(train_s, 0.0001) * 100 if train_s > 0 else 0
        st.metric("Écart Train/Test", f"{gap:.1f}%",
                   delta=f"{'⚠️ Overfitting' if gap > 10 else '✅ OK'}")

    st.divider()

    # ═══════════════════════════════════════
    # Onglets par type de problème
    # ═══════════════════════════════════════
    if problem_type == "Régression":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Réel vs Prédit", "📊 Résidus", "📉 Distribution",
            "🔍 Top erreurs", "🏆 Importance", "🌀 Permutation",
            "📐 Diagnostic"
        ])

        with tab1:
            afficher_aide_graphique("reel_vs_predit")
            fig_plotly = plot_real_vs_pred_interactive(y_test, y_pred)
            st.plotly_chart(fig_plotly, use_container_width=True)

        with tab2:
            afficher_aide_graphique("residus")
            fig = plot_residuals(y_test, y_pred)
            st.pyplot(fig)
            plt.close()

            comment = auto_comment_residuals(y_test, y_pred)
            st.markdown(f"💡 **Analyse :** {comment}")

        with tab3:
            afficher_aide_graphique("distribution_residus")
            fig = plot_residual_distribution(y_test, y_pred)
            st.pyplot(fig)
            plt.close()

        with tab4:
            afficher_aide_graphique("top_erreurs")
            top_err = get_top_errors(y_test, y_pred, n=10)
            st.dataframe(top_err, use_container_width=True)

        with tab5:
            afficher_aide_graphique("importance_features")
            _feature_importance_chart(model, _feature_names)

        with tab6:
            afficher_aide_graphique("permutation_importance")
            _permutation_importance_chart(model, X_test, y_test, _feature_names)

        with tab7:
            sub1, sub2 = st.tabs(["📐 QQ-plot résidus", "📈 Courbe d'apprentissage"])
            with sub1:
                afficher_aide_graphique("qq_plot")
                _qq_plot_residuals(y_test, y_pred)
            with sub2:
                afficher_aide_graphique("learning_curve")
                _learning_curve_chart(model, X_train, y_train, problem_type)

        # Rapport de régression
        st.divider()
        with st.expander("📋 Rapport détaillé de régression"):
            report = get_regression_report(
                model, _feature_names, y_test, y_pred,
                scaler=st.session_state.get("scaler"),
                scaled_columns=st.session_state.get("scaled_columns"),
            )
            if "coefficients" in report and report["coefficients"] is not None:
                st.markdown("**Coefficients :**")
                st.dataframe(report["coefficients"], use_container_width=True)
            for key in ["r2", "rmse", "mae", "mape"]:
                if key in report:
                    st.write(f"**{key.upper()} :** {report[key]:.4f}")

    else:  # Classification
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🎯 Matrice confusion", "📈 Courbe ROC",
            "📊 Precision-Recall", "🔍 Erreurs", "🏆 Importance",
            "🔬 Impact par classe", "🌀 Permutation",
            "📐 Diagnostic"
        ])

        with tab1:
            afficher_aide_graphique("matrice_confusion")
            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                st.markdown("**Comptages**")
                fig = plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig)
                plt.close()
            with col_cm2:
                st.markdown("**Proportions (0→1)**")
                fig_norm = plot_confusion_matrix(y_test, y_pred, normalize=True)
                st.pyplot(fig_norm)
                plt.close()

        with tab2:
            afficher_aide_graphique("roc_curve")
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    fig = plot_roc_curve(y_test, y_proba[:, 1])
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("ROC multi-classes non supporté ici.")
            else:
                st.info("Ce modèle ne supporte pas predict_proba.")

        with tab3:
            afficher_aide_graphique("precision_recall")
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    fig = plot_precision_recall_curve(y_test, y_proba[:, 1])
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Precision-Recall multi-classes non supporté ici.")
            else:
                st.info("Ce modèle ne supporte pas predict_proba.")

        with tab4:
            afficher_aide_graphique("erreurs_classification")
            mis = get_misclassified(y_test, y_pred, X=X_test, n=10)
            st.dataframe(mis, use_container_width=True)

        with tab5:
            afficher_aide_graphique("importance_features")
            _feature_importance_chart(model, _feature_names)

        with tab6:
            afficher_aide_graphique("impact_classe")
            _feature_impact_par_classe(model, _feature_names, y_test)

        with tab7:
            afficher_aide_graphique("permutation_importance")
            _permutation_importance_chart(model, X_test, y_test, _feature_names)

        with tab8:
            sub1, sub2 = st.tabs(["📈 Courbe d'apprentissage", "🎯 Calibration"])
            with sub1:
                afficher_aide_graphique("learning_curve")
                _learning_curve_chart(model, X_train, y_train, problem_type)
            with sub2:
                afficher_aide_graphique("calibration_curve")
                _calibration_curve_chart(model, X_test, y_test)

        # Rapport classification
        st.divider()
        with st.expander("📋 Rapport par classe"):
            report_df = get_classification_report(y_test, y_pred)
            st.dataframe(report_df, use_container_width=True)

            fig = plot_classification_metrics_bar(y_test, y_pred)
            st.pyplot(fig)
            plt.close()

    # Interprétation automatique
    st.divider()
    results = st.session_state.get("resultats_modeles", [])
    if results:
        interpretation = interpret_model_results(results, problem_type)
        st.markdown(f"### 💡 Résumé\n{interpretation}")

    # Analyse et propositions
    with st.expander("💡 Analyse et recommandations pour la suite", expanded=True):
        score = best.get("test_score", 0)
        train_s = best.get("train_score", 0)
        gap = abs(train_s - score) * 100 if train_s > 0 else 0

        st.markdown("**Propositions :**")
        props = []
        if problem_type == "Régression":
            if score < 0.7:
                props.append("- Le modèle explique peu les données → ajoutez des variables ou transformez-les")
            if gap > 10:
                props.append("- Sur-apprentissage → essayez une régularisation (Ridge, Lasso)")
            props.append("- Vérifiez les résidus : s'ils ne sont pas aléatoires, le modèle rate un pattern")
        else:  # Classification
            if score < 0.8:
                props.append("- Précision faible → vérifiez l'équilibre des classes et la pondération")
            if gap > 10:
                props.append("- Sur-apprentissage → réduisez la complexité ou augmentez les données")
            props.append("- Analysez la matrice de confusion : quelles classes sont le plus confondues ?")
            props.append("- Si une classe rare est mal prédite, activez class_weight='balanced'")
        props.append("- Passez à l'étape 9 pour optimiser ou prédire sur de nouvelles données")
        for p in props:
            st.markdown(p)

    # Validation
    st.divider()
    if st.button("✅ Valider l'évaluation", type="primary", key="validate_eval"):
        st.session_state["evaluation_done"] = True

        rapport = st.session_state.get("rapport", {})
        if rapport:
            rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 9)
            ajouter_historique(rapport, f"Évaluation de {model_name} validée")
            sauvegarder_rapport(rapport)

        st.session_state["_pending_step"] = 9
        st.rerun()


def _afficher_evaluation_ts(best: dict, model_name: str):
    """Évaluation spécifique aux séries temporelles."""
    ts_result = st.session_state.get("ts_arima_result")
    ts_series = st.session_state.get("ts_series")
    order = best.get("order")

    if ts_result is None or "error" in ts_result:
        st.warning("⚠️ Aucun résultat ARIMA disponible.")
        return

    # Métriques du modèle
    st.subheader("📊 Performance du modèle")

    # Calcul MASE si train dispo
    train_data = ts_result.get("train")
    test_data = ts_result.get("test")
    forecast_data = ts_result.get("forecast")
    mase_val = None
    if train_data is not None and test_data is not None and forecast_data is not None:
        mase_val = compute_mase(test_data, forecast_data, train_data)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{ts_result['mae']:.4f}")
    c2.metric("RMSE", f"{ts_result['rmse']:.4f}")
    c3.metric("MAPE", f"{ts_result['mape']:.1f}%")
    c4.metric("AIC", f"{ts_result['aic']:.0f}")
    if mase_val is not None and np.isfinite(mase_val):
        c5.metric("MASE", f"{mase_val:.3f}",
                  help="< 1 = meilleur que le naïf, > 1 = pire que le naïf")
    else:
        c5.metric("MASE", "N/A")

    # Graphique train/test/prévision
    st.pyplot(ts_result["figure"])
    plt.close()

    # Walk-forward validation
    st.divider()
    st.subheader("🔄 Validation Walk-Forward")
    st.markdown("*Re-entraîne le modèle à chaque pas pour simuler un usage réel.*")

    n_splits = st.slider("Nombre de folds", 3, 10, 5, key="wf_splits",
                          help="Plus de folds = évaluation plus robuste mais plus lente")
    gap = st.slider("Gap (points exclus entre train/test)", 0, 10, 0, key="wf_gap",
                     help="Exclut N points entre train et test pour éviter les fuites de données")

    if st.button("🚀 Lancer la validation walk-forward", key="wf_btn"):
        if ts_series is not None and order is not None:
            with st.spinner("Walk-forward en cours…"):
                wf_result = walk_forward_validation(ts_series, order,
                                                     n_splits=n_splits,
                                                     gap=gap)

            if "error" in wf_result:
                st.warning(f"⚠️ {wf_result['error']}")
            else:
                st.session_state["ts_wf_result"] = wf_result
                st.success("✅ Validation terminée !")
        else:
            st.warning("Série ou ordre ARIMA manquant.")

    wf_result = st.session_state.get("ts_wf_result")
    if wf_result and "error" not in wf_result:
        c1, c2 = st.columns(2)
        c1.metric("MAE moyenne", f"{wf_result['mean_mae']:.4f}")
        c2.metric("Écart-type MAE", f"{wf_result['std_mae']:.4f}")

        st.pyplot(wf_result["figure"])
        plt.close()

        # Détails par fold
        with st.expander("📋 Détails par fold"):
            df_folds = pd.DataFrame(wf_result["folds"])
            st.dataframe(df_folds, use_container_width=True)

    # Validation
    st.divider()
    if st.button("✅ Valider l'évaluation", type="primary", key="validate_eval_ts"):
        st.session_state["evaluation_done"] = True

        rapport = st.session_state.get("rapport", {})
        if rapport:
            rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 9)
            rapport["evaluation_ts"] = {
                "mae": ts_result["mae"],
                "rmse": ts_result["rmse"],
                "mape": ts_result["mape"],
                "mase": round(mase_val, 4) if mase_val is not None and np.isfinite(mase_val) else None,
            }
            if wf_result and "error" not in wf_result:
                rapport["evaluation_ts"]["wf_mean_mae"] = wf_result["mean_mae"]
            ajouter_historique(rapport, f"Évaluation {model_name} validée")
            sauvegarder_rapport(rapport)

        st.session_state["_pending_step"] = 9
        st.rerun()


def _afficher_evaluation_anomaly(best: dict, model_name: str, X_test):
    """Évaluation dédiée au parcours Détection d'anomalies."""
    model = best.get("model")
    if model is None or X_test is None:
        st.warning("⚠️ Données d'évaluation anomalies manquantes.")
        return

    preds = model.predict(X_test)
    anomaly_mask = preds == -1
    anomaly_rate = float(np.mean(anomaly_mask)) if len(preds) else 0.0
    normal_rate = 1.0 - anomaly_rate

    c1, c2, c3 = st.columns(3)
    c1.metric("Cas analysés", f"{len(preds)}")
    c2.metric("Taux anomalies", f"{anomaly_rate * 100:.2f}%")
    c3.metric("Taux normaux", f"{normal_rate * 100:.2f}%")

    if anomaly_rate > 0.2:
        st.warning("⚠️ Taux d'anomalies élevé (>20%) : possible dérive ou contamination forte.")
    elif anomaly_rate < 0.005:
        st.info("ℹ️ Très peu d'anomalies détectées (<0.5%). Vérifiez le paramètre contamination.")
    else:
        st.success("✅ Taux d'anomalies plausible pour un premier modèle.")

    st.divider()
    st.subheader("Distribution des scores d'anomalie")
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(scores, bins=40, color="#4F5BD5", alpha=0.85, edgecolor="white")
        ax.set_title("Scores de décision (plus bas = plus anormal)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Fréquence")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("**Recommandations :**")
    st.markdown("- Vérifiez manuellement un échantillon des lignes marquées anomalies")
    st.markdown("- Ajustez `contamination` si le taux détecté est trop haut ou trop bas")
    st.markdown("- Surveillez ce taux en production pour détecter la dérive")

    st.divider()
    if st.button("✅ Valider l'évaluation", type="primary", key="validate_eval_anomaly"):
        st.session_state["evaluation_done"] = True

        rapport = st.session_state.get("rapport", {})
        if rapport:
            rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 9)
            rapport["evaluation_anomaly"] = {
                "anomaly_rate": round(anomaly_rate, 4),
                "n_samples": int(len(preds)),
            }
            ajouter_historique(rapport, f"Évaluation anomalies de {model_name} validée")
            sauvegarder_rapport(rapport)

        st.session_state["_pending_step"] = 9
        st.rerun()
