# -*- coding: utf-8 -*-
"""
m2_diagnostic.py — Module 2 : Diagnostic complet + Cible & Variables.

Étape 3 : Diagnostic complet (EDA approfondi + recommandations)
Étape 4 : Cible & Variables (choisir quoi prédire et avec quoi)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import CORRELATION_THRESHOLD, MAX_ONEHOT_CARDINALITY
from src.audit import (
    quality_table, descriptive_stats, detect_anomalies, get_anomaly_actions,
    correlation_matrix, high_correlations, plot_correlation_heatmap,
    compute_quality_score, check_target_imbalance,
)
from src.guide import recommend_features
from src.timeseries import (
    detect_datetime_column, prepare_timeseries, detect_frequency,
    test_stationarity, decompose_series, plot_acf_pacf,
    plot_moving_averages, plot_trend_analysis, plot_timeseries,
    plot_timeseries_interactive,
    plot_seasonal_boxplot, auto_summary, suggest_arima_order,
)
from utils.data_utils import recommend_models, recommend_preprocessing
from utils.projet_manager import sauvegarder_rapport, ajouter_historique

CHART_COLORS = {
    "primary": "#4F5BD5", "secondary": "#818CF8", "accent": "#A5B4FC",
    "success": "#059669", "warning": "#D97706", "danger": "#DC2626",
    "palette": ["#4F5BD5", "#059669", "#D97706", "#DC2626", "#0891B2",
                "#7C3AED", "#EA580C", "#2563EB", "#14B8A6", "#E11D48"],
}


def _grille_distributions(df: pd.DataFrame):
    """Affiche une grille d'histogrammes pour TOUTES les colonnes."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    all_cols = num_cols + cat_cols
    if not all_cols:
        st.info("Aucune colonne à afficher.")
        return

    n_cols_display = min(2, len(all_cols))
    n_rows_display = (len(all_cols) + n_cols_display - 1) // n_cols_display

    fig, axes = plt.subplots(n_rows_display, n_cols_display,
                              figsize=(4 * n_cols_display, 2.5 * n_rows_display))
    fig.set_facecolor("#F8F9FC")

    if n_rows_display == 1 and n_cols_display == 1:
        axes = np.array([[axes]])
    elif n_rows_display == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_display == 1:
        axes = axes.reshape(-1, 1)

    for idx, col in enumerate(all_cols):
        row_i = idx // n_cols_display
        col_i = idx % n_cols_display
        ax = axes[row_i, col_i]
        ax.set_facecolor("#FFFFFF")

        if col in num_cols:
            data = df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=min(30, len(data.unique())),
                        color=CHART_COLORS["primary"], edgecolor="white",
                        alpha=0.85, linewidth=0.5)
                # Annoter la distribution
                skew = data.skew()
                label = "Normal" if abs(skew) < 0.5 else "Asymétrique" if abs(skew) < 1.5 else "Très asymétrique"
                ax.text(0.95, 0.95, f"{label}\nskew={skew:.2f}\n(cible: -0.5 à 0.5)",
                        transform=ax.transAxes, fontsize=7,
                        ha="right", va="top", color=CHART_COLORS["warning"] if abs(skew) > 1 else "#6B7280",
                        fontweight="bold" if abs(skew) > 1 else "normal")
        else:
            # Catégorielle : barplot des top 8
            counts = df[col].value_counts().head(8)
            if len(counts) > 0:
                bars = ax.barh(range(len(counts)), counts.values,
                               color=CHART_COLORS["palette"][:len(counts)],
                               edgecolor="white", linewidth=0.5)
                ax.set_yticks(range(len(counts)))
                ax.set_yticklabels([str(v)[:15] for v in counts.index], fontsize=6)
                for j, v in enumerate(counts.values):
                    pct = v / len(df) * 100
                    ax.text(v + 0.5, j, f"{pct:.0f}%", va="center", fontsize=6)

        ax.set_title(str(col)[:20], fontsize=8, fontweight="bold", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E5E7EB")
        ax.spines["bottom"].set_color("#E5E7EB")
        ax.tick_params(labelsize=6, colors="#6B7280")
        ax.grid(axis="y" if col in num_cols else "x", alpha=0.3, color="#F3F4F6")

    # Masquer les axes vides
    for idx in range(len(all_cols), n_rows_display * n_cols_display):
        row_i = idx // n_cols_display
        col_i = idx % n_cols_display
        axes[row_i, col_i].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _heatmap_manquantes(df: pd.DataFrame):
    """Carte thermique des valeurs manquantes."""
    na_matrix = df.isna().astype(int)
    if na_matrix.sum().sum() == 0:
        st.success("✅ Aucune valeur manquante !")
        return

    fig, ax = plt.subplots(figsize=(min(12, len(df.columns) * 0.5), 4))
    fig.set_facecolor("#F8F9FC")

    # Afficher un échantillon si trop de lignes
    sample = na_matrix if len(na_matrix) <= 200 else na_matrix.sample(200, random_state=42).sort_index()

    sns.heatmap(sample.T, cmap=["#FFFFFF", "#DC2626"], cbar_kws={"label": "Manquant"},
                ax=ax, yticklabels=True, xticklabels=False)
    ax.set_title("Carte des valeurs manquantes (rouge = manquant)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Résumé chiffré
    na_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    na_pct = na_pct[na_pct > 0]
    if not na_pct.empty:
        st.markdown("**Colonnes avec des trous :**")
        for col, pct in na_pct.items():
            icon = "🔴" if pct > 50 else "🟠" if pct > 20 else "🟢"
            st.write(f"  {icon} **{col}** : {pct:.1f}% manquant ({int(df[col].isna().sum())} valeurs)")


def _section_anomalies(df: pd.DataFrame):
    """Affiche les anomalies détectées."""
    anomalies = detect_anomalies(df)

    found = False
    if anomalies.get("constant_cols"):
        found = True
        st.error(f"🔴 **Colonnes constantes** (toujours la même valeur) : "
                 f"{', '.join(anomalies['constant_cols'])}")
        st.caption("→ À supprimer car elles n'apportent aucune information.")

    if anomalies.get("quasi_constant_cols"):
        found = True
        st.warning(f"🟠 **Colonnes quasi-constantes** : "
                   f"{', '.join(anomalies['quasi_constant_cols'])}")

    if anomalies.get("high_cardinality_cols"):
        found = True
        st.warning(f"🟠 **Haute cardinalité** (identifiants ?) : "
                   f"{', '.join(anomalies['high_cardinality_cols'])}")

    if anomalies.get("outlier_counts"):
        found = True
        st.info("📌 **Outliers détectés :**")
        for col, count in anomalies["outlier_counts"].items():
            pct = round(count / len(df) * 100, 1)
            st.write(f"  • **{col}** : {count} outlier(s) ({pct}%)")

    if not found:
        st.success("✅ Aucune anomalie détectée.")

    return anomalies


def afficher_diagnostic():
    """Étape 3 — Diagnostic complet des données."""
    st.caption("ÉTAPE 3")

    df = st.session_state.get("df_courant")
    if df is None:
        st.warning("⚠️ Complétez d'abord le chargement et la consolidation.")
        return

    with st.expander("🎓 Pourquoi cette étape ?", expanded=False):
        st.markdown("""
Avant de construire un modèle, il faut **examiner ses données** comme un médecin
examine un patient avant de prescrire un traitement.

Cette étape analyse automatiquement :

| Onglet | Ce qu'on cherche | Pourquoi c'est important |
|---|---|---|
| 📊 Distributions | Comment chaque colonne est répartie | Détecter les déséquilibres, biais |
| 🕳️ Manquantes | Les trous dans les données | Des trous = des calculs faussés |
| 🔗 Corrélations | Les liens entre colonnes | Colonnes redondantes nuisent au modèle |
| ⚠️ Anomalies | Valeurs aberrantes, quasi-constantes | Données suspectes qui faussent tout |
| 🎯 Recommandations | Suggestions de traitement | Actions concrètes pour la suite |

> **💡 Les indicateurs rouges et oranges** sont les points à traiter en priorité.
""")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distributions", "🕳️ Valeurs manquantes",
        "🔗 Corrélations", "⚠️ Anomalies", "🎯 Recommandations"
    ])

    problem_type = st.session_state.get("problem_type", "Régression")
    is_ts = problem_type == "Série temporelle"

    # ── Tab 1 : Distributions de TOUTES les colonnes ──
    with tab1:
        st.subheader("Distribution de toutes les colonnes")
        st.markdown("*Vue globale de la répartition de vos données. "
                    "Les colonnes numériques montrent un histogramme, "
                    "les catégorielles montrent les fréquences.*")
        _grille_distributions(df)

        # Boxplots des numériques
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols and len(num_cols) <= 20:
            st.divider()
            st.subheader("📦 Boxplots (dispersion des valeurs)")
            fig, ax = plt.subplots(figsize=(6, max(2, len(num_cols) * 0.35)))
            fig.set_facecolor("#F8F9FC")
            df[num_cols].boxplot(ax=ax, vert=False, patch_artist=True, return_type="dict",
                                 boxprops=dict(facecolor=CHART_COLORS["accent"],
                                               edgecolor=CHART_COLORS["primary"], linewidth=1.2),
                                 medianprops=dict(color=CHART_COLORS["danger"], linewidth=2),
                                 flierprops=dict(marker="o", markerfacecolor=CHART_COLORS["warning"],
                                                 markeredgecolor="white", markersize=5, alpha=0.7))
            ax.set_facecolor("#FFFFFF")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title("Dispersion (points isolés = outliers)", fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 2 : Carte thermique des valeurs manquantes ──
    with tab2:
        st.subheader("Carte des valeurs manquantes")
        _heatmap_manquantes(df)

    # ── Tab 3 : Corrélations ──
    with tab3:
        st.subheader("Corrélations entre variables")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr = correlation_matrix(df)
            fig = plot_correlation_heatmap(corr)
            st.pyplot(fig)
            plt.close()

            # Top paires corrélées
            high_corr = high_correlations(corr, threshold=CORRELATION_THRESHOLD)
            if not high_corr.empty:
                st.warning("⚠️ Variables très corrélées (> 80%) — risque de multicolinéarité :")
                st.dataframe(high_corr, use_container_width=True)
            else:
                st.success("✅ Pas de multicolinéarité excessive.")

            # Top 10 paires les plus corrélées
            st.divider()
            st.markdown("**Top corrélations (valeur absolue) :**")
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            corr_flat = corr.where(mask).stack().abs().sort_values(ascending=False)
            top_pairs = corr_flat.head(10)
            for (c1, c2), val in top_pairs.items():
                icon = "🟢" if val > 0.7 else "🟡" if val > 0.4 else "⚪"
                st.write(f"  {icon} **{c1}** ↔ **{c2}** : {val:.3f}")
        else:
            st.info("Pas assez de colonnes numériques pour les corrélations.")

    # ── Tab 4 : Anomalies ──
    with tab4:
        st.subheader("Anomalies détectées")
        anomalies = _section_anomalies(df)
        st.session_state["anomalies"] = anomalies

    # ── Tab 5 : Recommandations (LA GROSSE NOUVEAUTÉ) ──
    with tab5:
        st.subheader("🎯 Recommandations basées sur vos données")
        st.markdown("*L'application a analysé vos données et propose des conseils personnalisés.*")

        problem_type = st.session_state.get("problem_type", "Régression")
        target_col = st.session_state.get("target_col")

        # Preprocessing recommendations
        preproc = recommend_preprocessing(df, target_col)

        # Scaling
        st.markdown("---")
        if preproc["scaling"]["needed"]:
            st.markdown(f"📐 **Scaling : OUI**")
            st.markdown(f"  → {preproc['scaling']['reason']}")
        else:
            st.markdown("📐 **Scaling : optionnel**")
            st.caption("Les échelles de vos colonnes numériques sont similaires.")

        # Encoding
        if preproc["encoding"]:
            st.markdown("---")
            st.markdown(f"🏷️ **Encoding : {len(preproc['encoding'])} colonne(s) texte à convertir**")
            for enc in preproc["encoding"]:
                st.write(f"  → **{enc['colonne']}** ({enc['n_valeurs']} valeurs) → {enc['methode']}")
                st.caption(f"    {enc['raison']}")

        # Model recommendations (si cible définie)
        if target_col and target_col in df.columns:
            st.markdown("---")
            reco = recommend_models(df, target_col, problem_type)

            st.markdown("🤖 **Modèles suggérés :**")
            for i, m in enumerate(reco["modeles"], 1):
                st.write(f"  {i}. **{m['nom']}** — {m['raison']}")

            if reco["alertes"]:
                st.markdown("---")
                for alerte in reco["alertes"]:
                    st.warning(alerte)

            # Sauvegarder les recommandations
            st.session_state["modeles_recommandes"] = [m["nom"] for m in reco["modeles"]]
        else:
            st.markdown("---")
            st.info("💡 Définissez la cible (étape 4) pour obtenir des recommandations de modèles.")

        # Alertes preprocessing
        if preproc["alertes"]:
            st.markdown("---")
            st.markdown("⚠️ **Points d'attention :**")
            for alerte in preproc["alertes"]:
                st.write(f"  - {alerte}")

        # Score qualité global
        st.markdown("---")
        anomalies = st.session_state.get("anomalies", detect_anomalies(df))
        quality = compute_quality_score(df, anomalies)
        score = quality["score"]

        badge = "quality-good" if score >= 70 else "quality-medium" if score >= 40 else "quality-bad"
        st.markdown(f"### Score qualité : **{score}/100** {'🟢' if score >= 70 else '🟠' if score >= 40 else '🔴'}")

        if quality.get("issues"):
            for issue in quality["issues"]:
                st.write(f"  → {issue}")

    # ── Section Série temporelle (conditionnelle) ──
    if is_ts:
        st.divider()
        st.subheader("📈 Diagnostic Série temporelle")

        dt_col = detect_datetime_column(df)
        if dt_col is None:
            st.warning("⚠️ Aucune colonne datetime détectée. "
                       "Vérifiez que vos données contiennent une colonne de dates.")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.warning("⚠️ Aucune colonne numérique pour la série temporelle.")
            else:
                ts_value_col = st.selectbox(
                    "Colonne de valeurs à analyser", num_cols,
                    key="ts_diag_value_col")

                try:
                    ts_series = prepare_timeseries(df, dt_col, ts_value_col)
                except Exception as e:
                    st.error(f"❌ Impossible de préparer la série : {e}")
                    ts_series = None

                if ts_series is not None and len(ts_series) >= 10:
                    # Résumé automatique
                    summary = auto_summary(ts_series)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Points", summary["n_points"])
                    c2.metric("Fréquence", summary["frequence"])
                    c3.metric("Moyenne", f"{summary['moyenne']:.2f}")
                    c4.metric("Stationnaire",
                              "✅ Oui" if summary["stationnaire"] else "❌ Non")

                    ts_tab1, ts_tab2, ts_tab3, ts_tab4 = st.tabs([
                        "📈 Série", "📊 Décomposition",
                        "📉 ACF / PACF", "📋 Stationnarité",
                    ])

                    with ts_tab1:
                        fig_plotly = plot_timeseries_interactive(
                            ts_series, title=f"{ts_value_col} au cours du temps")
                        st.plotly_chart(fig_plotly, use_container_width=True)

                        _gc1, _gc2 = st.columns(2)
                        with _gc1:
                            fig = plot_moving_averages(ts_series)
                            st.pyplot(fig)
                            plt.close()
                        with _gc2:
                            fig = plot_trend_analysis(ts_series)
                            st.pyplot(fig)
                            plt.close()

                    with ts_tab2:
                        _gd1, _gd2 = st.columns(2)
                        with _gd1:
                            try:
                                decomp = decompose_series(ts_series)
                                st.pyplot(decomp["figure"])
                                plt.close()
                                st.caption(f"Période détectée : {decomp['period']}")
                            except Exception as e:
                                st.warning(f"Décomposition impossible : {e}")
                        with _gd2:
                            try:
                                fig = plot_seasonal_boxplot(ts_series, period="month")
                                st.pyplot(fig)
                                plt.close()
                            except Exception:
                                pass

                    with ts_tab3:
                        fig = plot_acf_pacf(ts_series)
                        st.pyplot(fig)
                        plt.close()

                        order = suggest_arima_order(ts_series)
                        st.info(f"💡 Ordre ARIMA suggéré : **{order}**")
                        st.session_state["ts_suggested_order"] = order

                    with ts_tab4:
                        stationarity = test_stationarity(ts_series)
                        st.markdown(f"**Conclusion :** {stationarity['conclusion']}")

                        st.markdown("**Test ADF** (H₀ = non stationnaire)")
                        st.write(f"  Statistique : {stationarity['adf_statistic']}, "
                                 f"p-value : {stationarity['adf_pvalue']} → "
                                 f"{'Stationnaire ✅' if stationarity['adf_stationary'] else 'Non stationnaire ❌'}")

                        if stationarity["kpss_statistic"] is not None:
                            st.markdown("**Test KPSS** (H₀ = stationnaire)")
                            st.write(f"  Statistique : {stationarity['kpss_statistic']}, "
                                     f"p-value : {stationarity['kpss_pvalue']} → "
                                     f"{'Stationnaire ✅' if stationarity['kpss_stationary'] else 'Non stationnaire ❌'}")

                    # Sauvegarder en session pour les modules suivants
                    st.session_state["ts_datetime_col"] = dt_col
                    st.session_state["ts_value_col"] = ts_value_col
                    st.session_state["ts_series"] = ts_series

                elif ts_series is not None:
                    st.warning("⚠️ Pas assez de points (minimum 10) pour l'analyse temporelle.")

    st.session_state["diagnostic_done"] = True
    rapport = st.session_state.get("rapport", {})
    if rapport:
        rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 4)
        rapport["diagnostic"] = {
            "score_qualite": quality["score"] if "quality" in dir() else None,
            "n_anomalies": len(anomalies.get("outlier_counts", {})) if "anomalies" in dir() else 0,
        }
        ajouter_historique(rapport, "Diagnostic complet")
        sauvegarder_rapport(rapport)


def afficher_cible_variables():
    """Étape 4 — Choix de la cible et des variables."""
    st.caption("ÉTAPE 4")

    df = st.session_state.get("df_courant")
    if df is None:
        st.warning("⚠️ Complétez d'abord le diagnostic (étape 3).")
        return

    with st.expander("🎓 Cible et variables : c'est quoi ?", expanded=False):
        st.markdown("""
**La cible** (ou *target*) = ce que vous voulez **prédire**.
Exemple : le prix d'un bien, la catégorie d'un client, le niveau d'un barrage.

**Les variables explicatives** (ou *features*) = les informations à partir desquelles
le modèle va apprendre à prédire.

```
┌────────────────────────────────────────────┐
│  Données                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Surface  │  │ Quartier │  │  PRIX  🎯 │ │
│  │ (feature)│  │ (feature)│  │ (cible)   │ │
│  └──────────┘  └──────────┘  └───────────┘ │
└────────────────────────────────────────────┘
     Le modèle apprend :  surface + quartier → prix
```

> **⚠️ Piège à éviter** : n'incluez pas de colonne qui "triche" !
> Par exemple, si vous prédisez le prix TTC, n'incluez pas le prix HT
> (le modèle aura un score parfait mais sera inutile).
""")

    problem_type = st.session_state.get("problem_type", "Régression")
    all_cols = df.columns.tolist()

    # ═══════════════════════════════════════
    # Parcours Série temporelle
    # ═══════════════════════════════════════
    if problem_type == "Série temporelle":
        st.subheader("Définir la série temporelle")

        dt_col_saved = st.session_state.get("ts_datetime_col")
        dt_col_default = detect_datetime_column(df)
        dt_candidates = all_cols
        dt_idx = 0
        if dt_col_saved and dt_col_saved in dt_candidates:
            dt_idx = dt_candidates.index(dt_col_saved)
        elif dt_col_default and dt_col_default in dt_candidates:
            dt_idx = dt_candidates.index(dt_col_default)

        datetime_col = st.selectbox("Colonne de dates (index temporel)",
                                     dt_candidates, index=dt_idx,
                                     key="ts_dt_select_m2",
                                     help="Colonne contenant les dates/timestamps pour l'analyse temporelle")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.warning("⚠️ Aucune colonne numérique disponible.")
            return

        val_saved = st.session_state.get("ts_value_col")
        val_idx = num_cols.index(val_saved) if val_saved and val_saved in num_cols else 0
        value_col = st.selectbox("Colonne de valeurs à prédire",
                                  num_cols, index=val_idx,
                                  key="ts_val_select_m2")

        # Aperçu de la série
        try:
            ts_preview = prepare_timeseries(df, datetime_col, value_col)
            freq_info = detect_frequency(ts_preview)
            c1, c2, c3 = st.columns(3)
            c1.metric("Points", len(ts_preview))
            c2.metric("Fréquence détectée", freq_info["label"])
            c3.metric("Plage", f"{ts_preview.index.min().date()} → {ts_preview.index.max().date()}")

            fig_plotly = plot_timeseries_interactive(
                ts_preview, title=f"{value_col} au cours du temps")
            st.plotly_chart(fig_plotly, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Impossible de préparer la série : {e}")

        # Guide choix univarié / multivarié
        other_num = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c != value_col]
        if other_num:
            st.info(
                f"💡 Vous avez **{len(other_num)} autre(s) variable(s) numérique(s)** "
                f"({', '.join(other_num[:5])}{'…' if len(other_num) > 5 else ''}).\n\n"
                f"- **ARIMA** (défaut) : prédit `{value_col}` **uniquement** à partir "
                f"de son historique — les autres variables seront ignorées.\n"
                f"- **Prédiction horizon** (étape 6) : utilise toutes les variables "
                f"pour prédire à N jours. Activez-le à l'étape Transformation."
            )

        if st.button("✅ Valider la série temporelle", type="primary",
                      key="validate_ts_target"):
            st.session_state["target_col"] = value_col
            st.session_state["ts_datetime_col"] = datetime_col
            st.session_state["ts_value_col"] = value_col
            st.session_state["feature_cols"] = [datetime_col, value_col]
            st.session_state["cible_done"] = True

            try:
                ts_series = prepare_timeseries(df, datetime_col, value_col)
                st.session_state["ts_series"] = ts_series
            except Exception:
                pass

            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["colonne_cible"] = value_col
                rapport["ts_datetime_col"] = datetime_col
                rapport["colonnes_features"] = [datetime_col, value_col]
                rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 5)
                ajouter_historique(rapport,
                                   f"Série temporelle : date={datetime_col}, valeur={value_col}")
                sauvegarder_rapport(rapport)

            st.session_state["_pending_step"] = 5
            st.rerun()
        return

    # ═══════════════════════════════════════
    # Parcours ML classique (Régression / Classification)
    # ═══════════════════════════════════════
    st.subheader("Que voulez-vous prédire ?")

    saved_target = st.session_state.get("target_col")
    target_idx = all_cols.index(saved_target) if saved_target and saved_target in all_cols else 0
    target = st.selectbox("Colonne cible", all_cols, index=target_idx,
                          key="target_select_m2")

    if target:
        # Distribution de la cible
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("**Distribution de la cible :**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.set_facecolor("#F8F9FC")
            ax.set_facecolor("#FFFFFF")

            if pd.api.types.is_numeric_dtype(df[target]):
                df[target].dropna().hist(ax=ax, bins=30, color=CHART_COLORS["primary"],
                                         edgecolor="white", alpha=0.85)
                skew = df[target].skew()
                skew_info = ("✅ symétrique" if abs(skew) < 0.5
                             else "🟠 asymétrique" if abs(skew) < 1.5
                             else "🔴 très asymétrique")
                ax.set_title(f"Distribution de {target} (skew={skew:.2f}, {skew_info})",
                             fontsize=10, fontweight="bold")
            else:
                counts = df[target].value_counts().head(15)
                counts.plot.barh(ax=ax, color=CHART_COLORS["palette"][:len(counts)],
                                 edgecolor="white")
                ax.set_title(f"Fréquences de {target}", fontsize=10, fontweight="bold")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.metric("Type", "Numérique" if pd.api.types.is_numeric_dtype(df[target]) else "Catégorielle")
            st.metric("Valeurs uniques", df[target].nunique())
            st.metric("Manquantes", f"{df[target].isna().sum()} ({df[target].isna().mean()*100:.1f}%)")

            if problem_type == "Classification":
                is_imbalanced = check_target_imbalance(df[target])
                if is_imbalanced:
                    st.warning("⚠️ Classes déséquilibrées")
                else:
                    st.success("✅ Classes équilibrées")

        # Corrélations avec la cible
        if pd.api.types.is_numeric_dtype(df[target]):
            other_num = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
            if other_num:
                st.divider()
                st.subheader("🔗 Corrélation avec la cible")
                corr_target = df[other_num + [target]].corr()[target].drop(target).abs().sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(5, max(2, len(corr_target) * 0.28)))
                fig.set_facecolor("#F8F9FC")
                colors = [CHART_COLORS["success"] if v > 0.3
                          else CHART_COLORS["warning"] if v > 0.1
                          else "#D1D5DB" for v in corr_target.values]
                corr_target.plot.barh(ax=ax, color=colors, edgecolor="white")
                ax.set_facecolor("#FFFFFF")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_title("Force du lien avec la cible", fontsize=10, fontweight="bold")
                ax.tick_params(labelsize=8)
                for i, v in enumerate(corr_target.values):
                    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption("🟢 Fort (>0.3) | 🟠 Moyen (>0.1) | ⚪ Faible")

        # ── Sélection des features ──
        st.divider()
        st.subheader("📋 Variables explicatives")

        other_cols = [c for c in all_cols if c != target]

        # Auto-recommendation
        recs = recommend_features(df, target, problem_type)
        recommended_cols = []
        if recs:
            st.markdown("**Classement automatique :**")
            for rec in recs:
                col_name = rec["col"]
                recommendation = rec.get("recommendation", "")
                reason = rec.get("reason", "")
                c1, c2 = st.columns([3, 5])
                with c1:
                    icon = recommendation.split(" ")[0] if recommendation else ""
                    st.markdown(f"{icon} **{col_name}**")
                with c2:
                    st.caption(reason)
                if "Très utile" in recommendation or "Utile" in recommendation:
                    recommended_cols.append(col_name)

        saved_features = st.session_state.get("feature_cols")
        if saved_features and saved_target == target:
            default = [c for c in saved_features if c in other_cols]
        else:
            default = recommended_cols if recommended_cols else other_cols

        features = st.multiselect("Variables sélectionnées", other_cols,
                                   default=default, key="feature_select_m2")

        if st.button("✅ Valider cible et variables", type="primary"):
            if not features:
                st.error("❌ Sélectionnez au moins **une variable explicative**.")
                st.stop()
            st.session_state["target_col"] = target
            st.session_state["feature_cols"] = features
            st.session_state["cible_done"] = True

            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["colonne_cible"] = target
                rapport["colonnes_features"] = features
                rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 5)
                ajouter_historique(rapport, f"Cible={target}, {len(features)} features")
                sauvegarder_rapport(rapport)

            # Re-calculer recommandations modèles avec la cible
            reco = recommend_models(df, target, problem_type)
            st.session_state["modeles_recommandes"] = [m["nom"] for m in reco["modeles"]]

            st.session_state["_pending_step"] = 5
            st.rerun()
