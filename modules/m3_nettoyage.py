# -*- coding: utf-8 -*-
"""
m3_nettoyage.py — Module 3 : Nettoyage + Transformation.

Étape 5 : Nettoyage (valeurs manquantes → outliers, ordre imposé)
Étape 6 : Transformation (encoding → scaling → feature engineering, ordre imposé)

Chaque sous-étape est verrouillée tant que la précédente n'est pas validée.
Chaque opération montre un avant/après côte à côte.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MAX_ONEHOT_CARDINALITY
from src.preprocessing import (
    handle_missing, detect_outliers_iqr, handle_outliers,
    normalize_columns, encode_categorical,
    get_categorical_columns, get_numeric_columns,
)
from src.feature_engineering import (
    combine_columns, transform_column, discretize_column,
    rename_column, drop_column, detect_datetime_columns,
    extract_datetime_features, create_lag_features, create_rolling_features,
    create_horizon_target, create_lead_features,
)
from src.guide import (
    recommend_missing_strategy, recommend_outlier_strategy,
    recommend_encoding, recommend_normalization,
)
from src.timeseries import (
    analyze_ts_continuity, detect_seasonality, recommend_ts_transforms,
    detect_frequency, prepare_timeseries, reindex_ts,
)
from utils.data_utils import apercu_avant_apres
from utils.projet_manager import sauvegarder_rapport, sauvegarder_csv, ajouter_historique, sauvegarder_objet

CHART_COLORS = {
    "primary": "#4F5BD5", "accent": "#A5B4FC",
    "success": "#059669", "warning": "#D97706", "danger": "#DC2626",
}


def _skew_label(skew_val: float) -> str:
    """Renvoie un libellé interprété pour le skewness."""
    v = abs(skew_val)
    if v < 0.5:
        return f"skew = {skew_val:.2f} (✅ symétrique)"
    elif v < 1.0:
        return f"skew = {skew_val:.2f} (🟠 légèrement asymétrique)"
    elif v < 1.5:
        return f"skew = {skew_val:.2f} (🟠 asymétrique)"
    else:
        return f"skew = {skew_val:.2f} (🔴 très asymétrique)"


def _afficher_avant_apres(avant: pd.DataFrame, apres: pd.DataFrame, titre: str = ""):
    """Affiche un comparatif avant/après côte à côte."""
    comp = apercu_avant_apres(avant, apres)

    if titre:
        st.markdown(f"### {titre}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**AVANT** (5 lignes)")
        st.dataframe(comp["avant"], use_container_width=True, height=200)
    with c2:
        st.markdown("**APRÈS** (5 lignes)")
        st.dataframe(comp["apres"], use_container_width=True, height=200)

    # Résumé des modifications
    details = []
    if comp["colonnes_modifiees"]:
        details.append(f"Colonnes modifiées : {', '.join(comp['colonnes_modifiees'])}")
    if comp["colonnes_ajoutees"]:
        details.append(f"Colonnes ajoutées : {', '.join(comp['colonnes_ajoutees'])}")
    if comp["colonnes_supprimees"]:
        details.append(f"Colonnes supprimées : {', '.join(comp['colonnes_supprimees'])}")
    details.append(f"Lignes : {comp['lignes_avant']} → {comp['lignes_apres']}"
                   f" ({comp['lignes_diff']} supprimées)" if comp["lignes_diff"] > 0
                   else f"Lignes : {comp['lignes_avant']} → {comp['lignes_apres']} (aucune supprimée)")

    for d in details:
        st.caption(d)


def _afficher_nettoyage_ts(df: pd.DataFrame):
    """Nettoyage spécifique aux séries temporelles : doublons, continuité, gaps, interpolation."""
    dt_col = st.session_state.get("ts_datetime_col")
    val_col = st.session_state.get("ts_value_col")
    if not dt_col:
        st.warning("⚠️ Colonne datetime non définie. Retournez au diagnostic.")
        return

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != val_col]
    value_cols = [val_col] + num_cols if val_col else num_cols

    tab_dup, tab_cont, tab_interp, tab_valid = st.tabs([
        "👯 5a. Doublons de date",
        "📊 5b. Continuité & Gaps",
        "🔧 5c. Interpolation",
        "✅ 5d. Validation",
    ])

    # ═══════════════════════════════════════
    # 5a. Doublons de date
    # ═══════════════════════════════════════
    with tab_dup:
        st.subheader("Détection des doublons de date")
        st.markdown(
            "**Objectif :** Vérifier qu'il n'y a pas de **dates en double** dans la série. "
            "Deux cas possibles :\n"
            "- **Lignes identiques** : même date, mêmes valeurs → on supprime le doublon\n"
            "- **Valeurs conflictuelles** : même date, valeurs différentes → "
            "on choisit quelle valeur garder (1ère, dernière, ou moyenne)"
        )

        if st.session_state.get("_ts_dedup_success"):
            st.success(st.session_state.pop("_ts_dedup_success"))

        df_ts = df.copy()
        df_ts[dt_col] = pd.to_datetime(df_ts[dt_col])

        # Doublons de date (pas doublons de ligne entière)
        date_counts = df_ts[dt_col].value_counts()
        dup_dates = date_counts[date_counts > 1]
        n_dup_dates = len(dup_dates)

        if n_dup_dates == 0:
            st.success("✅ Aucune date en double — chaque date est unique.")
            st.session_state["ts_dedup_done"] = True
        else:
            n_dup_rows = int(dup_dates.sum() - n_dup_dates)
            st.warning(f"⚠️ **{n_dup_dates} date(s)** apparaissent plus d'une fois "
                       f"(**{n_dup_rows} ligne(s) surnuméraires**).")

            # Classer en identiques vs conflictuels
            identical_dates = []
            conflict_dates = []
            for dt_val in dup_dates.index:
                rows = df_ts[df_ts[dt_col] == dt_val]
                vals_only = rows.drop(columns=[dt_col])
                if vals_only.duplicated(keep=False).all():
                    identical_dates.append(dt_val)
                else:
                    conflict_dates.append(dt_val)

            if identical_dates:
                st.markdown(f"🟢 **{len(identical_dates)} date(s)** avec lignes identiques "
                            "(suppression sans perte)")
            if conflict_dates:
                st.markdown(f"🔴 **{len(conflict_dates)} date(s)** avec valeurs différentes "
                            "(choix nécessaire)")

            # Tableau des doublons
            with st.expander(f"👁️ Voir les {min(n_dup_dates, 20)} premières dates en double",
                             expanded=(n_dup_dates <= 10)):
                dup_rows = []
                for dt_val in list(dup_dates.index)[:20]:
                    rows = df_ts[df_ts[dt_col] == dt_val]
                    for _, row in rows.iterrows():
                        row_dict = {"Date": str(dt_val)[:19]}
                        for vc in value_cols[:5]:
                            if vc in row.index:
                                row_dict[vc] = row[vc]
                        row_dict["Type"] = "Identique" if dt_val in identical_dates else "Conflit"
                        dup_rows.append(row_dict)
                st.dataframe(pd.DataFrame(dup_rows), use_container_width=True)

            # Stratégie de résolution
            if conflict_dates:
                strategy = st.selectbox("Stratégie pour les valeurs conflictuelles :", [
                    "first (garder la 1ère occurrence)",
                    "last (garder la dernière occurrence)",
                    "mean (moyenne des valeurs)",
                ], key="ts_dedup_strategy",
                   help="Comment résoudre les dates en double avec des valeurs différentes")
                strategy_key = strategy.split(" ")[0]
            else:
                strategy_key = "first"

            if st.button("🗑️ Supprimer les doublons de date", type="primary",
                         key="ts_dedup_apply"):
                df_new = df_ts.copy()
                df_new = df_new.sort_values(dt_col)

                if strategy_key == "mean":
                    # Agréger numériques par moyenne, garder 1er pour le reste
                    num_agg = {c: "mean" for c in df_new.select_dtypes(include="number").columns}
                    cat_agg = {c: "first" for c in df_new.columns
                               if c != dt_col and c not in num_agg}
                    agg_dict = {**num_agg, **cat_agg}
                    df_new = df_new.groupby(dt_col, as_index=False).agg(agg_dict)
                else:
                    keep = "first" if strategy_key == "first" else "last"
                    df_new = df_new.drop_duplicates(subset=dt_col, keep=keep)

                df_new = df_new.sort_values(dt_col).reset_index(drop=True)
                n_removed = len(df) - len(df_new)
                st.session_state["df_courant"] = df_new
                st.session_state["ts_dedup_done"] = True
                st.session_state["_ts_dedup_success"] = (
                    f"✅ {n_removed} doublon(s) supprimé(s) (stratégie : {strategy_key}). "
                    f"{len(df_new)} lignes restantes.")

                rapport = st.session_state.get("rapport", {})
                if rapport:
                    ajouter_historique(rapport,
                                       f"Doublons de date : {n_removed} supprimé(s) ({strategy_key})")
                    sauvegarder_rapport(rapport)

                st.rerun()

            if st.button("⏭️ Conserver les doublons", key="ts_skip_dedup"):
                st.session_state["ts_dedup_done"] = True
                st.rerun()

    # ═══════════════════════════════════════
    # 5b. Analyse de continuité
    # ═══════════════════════════════════════
    with tab_cont:
        st.subheader("Analyse de la continuité temporelle")
        st.markdown(
            "**Objectif :** Vérifier que la série de dates est régulière et "
            "identifier les **trous** (gaps) et les **valeurs manquantes** "
            "colonne par colonne. Cette étape est un diagnostic — "
            "les corrections se font dans l'onglet suivant (5c)."
        )

        # Recharger df courant (peut avoir changé après dédup)
        df = st.session_state.get("df_courant", df)

        with st.spinner("Analyse en cours…"):
            analysis = analyze_ts_continuity(df, dt_col, value_cols)

        # Résumé global
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📏 Points totaux", f"{analysis['total_points']}")
        freq_label = str(analysis["median_delta"])
        if hasattr(analysis["median_delta"], 'days'):
            d = analysis["median_delta"]
            if d.days >= 28:
                freq_label = f"~{d.days} jours (~mensuel)"
            elif d.days >= 7:
                freq_label = f"{d.days} jours (~hebdo)"
            elif d.days >= 1:
                freq_label = f"{d.days} jour(s)"
            else:
                total_sec = d.total_seconds()
                if total_sec >= 3600:
                    freq_label = f"{total_sec / 3600:.0f}h"
                else:
                    freq_label = f"{total_sec / 60:.0f}min"
        c2.metric("⏱️ Fréquence médiane", freq_label)
        c3.metric("🕳️ Gaps détectés", f"{analysis['n_gaps']}")
        c4.metric("👯 Doublons date", f"{analysis['n_duplicates']}")

        # ── Graphique de la série avec trous surlignés ──
        if val_col and val_col in df.columns:
            st.markdown("### 📈 Série temporelle avec trous surlignés")
            df_plot = df.copy()
            df_plot[dt_col] = pd.to_datetime(df_plot[dt_col])
            df_plot = df_plot.sort_values(dt_col).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(12, 4))
            fig.set_facecolor("#F8F9FC")
            ax.set_facecolor("#FFFFFF")
            ax.plot(df_plot[dt_col], df_plot[val_col], linewidth=0.8,
                    color="#4F5BD5", label=val_col, zorder=2)

            # Surligner chaque gap en rouge
            for g in analysis["gaps"]:
                ax.axvspan(g["debut"], g["fin"], alpha=0.25, color="#DC2626",
                           zorder=1)

            # Marquer les NaN existants (points orange)
            na_mask = df_plot[val_col].isna()
            if na_mask.any():
                na_dates = df_plot.loc[na_mask, dt_col]
                ax.scatter(na_dates, [ax.get_ylim()[0]] * len(na_dates),
                           color="#D97706", s=8, zorder=3, label="NaN existants")

            ax.set_title(f"{val_col} — zones rouges = gaps temporels", fontsize=10,
                         fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Carte des trous : vue empilée de toutes les colonnes ──
        st.markdown("### 🗺️ Carte des trous par colonne")
        st.caption("Chaque ligne = une colonne. "
                   "**Bleu** = donnée présente, **Rouge** = valeur manquante.")

        df_ts = df.copy()
        df_ts[dt_col] = pd.to_datetime(df_ts[dt_col])
        df_ts = df_ts.sort_values(dt_col).reset_index(drop=True)
        dates = df_ts[dt_col]

        display_cols = [c for c in value_cols if c in df_ts.columns]
        n_cols = len(display_cols)

        if n_cols > 0:
            fig_h = max(1.5, 0.5 * n_cols + 0.8)
            fig, ax = plt.subplots(figsize=(12, fig_h))
            fig.set_facecolor("#F8F9FC")
            ax.set_facecolor("#FFFFFF")

            for i, col in enumerate(display_cols):
                present = ~df_ts[col].isna()
                y = n_cols - 1 - i

                prev_state = None
                seg_start = 0
                for j in range(len(present)):
                    state = present.iloc[j]
                    if state != prev_state and prev_state is not None:
                        color = "#4F5BD5" if prev_state else "#DC2626"
                        alpha = 0.7 if prev_state else 0.85
                        ax.barh(y, dates.iloc[j] - dates.iloc[seg_start],
                                left=dates.iloc[seg_start],
                                height=0.7, color=color, alpha=alpha,
                                edgecolor="none")
                        seg_start = j
                    prev_state = state
                if prev_state is not None:
                    color = "#4F5BD5" if prev_state else "#DC2626"
                    alpha = 0.7 if prev_state else 0.85
                    ax.barh(y, dates.iloc[-1] - dates.iloc[seg_start],
                            left=dates.iloc[seg_start],
                            height=0.7, color=color, alpha=alpha,
                            edgecolor="none")

            ax.set_yticks(range(n_cols))
            ax.set_yticklabels(list(reversed(display_cols)), fontsize=8)
            ax.set_title("Carte des trous — toutes colonnes", fontsize=10,
                         fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("🟦 Données présentes  🟥 Valeurs manquantes")

        # ── Tableau détaillé des gaps ──
        if analysis["gaps"]:
            total_missing = sum(g["periodes_manquees"] for g in analysis["gaps"])
            st.markdown(f"### 📋 Détail des {analysis['n_gaps']} gap(s) "
                        f"({total_missing} périodes manquées)")

            gap_rows = []
            for i, g in enumerate(analysis["gaps"]):
                row = {
                    "#": i + 1,
                    "Début": str(g["debut"])[:10],
                    "Fin": str(g["fin"])[:10],
                    "Périodes manquées": g["periodes_manquees"],
                    "Durée": str(g["duree"]),
                }
                # Valeurs avant/après pour la colonne cible
                if val_col and "val_before" in g:
                    vb = g["val_before"].get(val_col)
                    va = g["val_after"].get(val_col)
                    row["Valeur avant"] = f"{vb:.4g}" if vb is not None else "—"
                    row["Valeur après"] = f"{va:.4g}" if va is not None else "—"
                    if vb is not None and va is not None:
                        row["Écart"] = f"{abs(va - vb):.4g}"
                    else:
                        row["Écart"] = "—"
                # Catégorie
                if g["periodes_manquees"] <= 5:
                    row["Catégorie"] = "🟢 Petit"
                elif g["periodes_manquees"] <= 30:
                    row["Catégorie"] = "🟠 Moyen"
                else:
                    row["Catégorie"] = "🔴 Gros"
                gap_rows.append(row)

            st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Aucun gap détecté — la série est continue.")

        # Stats par colonne
        if len(value_cols) > 1:
            st.markdown("### Statistiques par colonne")
            col_stats_rows = []
            for col, stats in analysis["column_stats"].items():
                col_stats_rows.append({
                    "Colonne": col,
                    "Points": stats["n_total"],
                    "Manquants": stats["n_missing"],
                    "% manquant": f"{stats['pct_missing']}%",
                    "Plus long trou": stats["longest_na_streak"],
                    "1ère valide": stats["first_valid"] or "—",
                    "Dernière valide": stats["last_valid"] or "—",
                })
            st.dataframe(pd.DataFrame(col_stats_rows), use_container_width=True)

        # Recommandations
        if analysis["recommendations"]:
            st.markdown("### 💡 Recommandations")
            for rec in analysis["recommendations"]:
                st.markdown(rec["message"])

        st.session_state["ts_continuity_analysis"] = analysis

    # ═══════════════════════════════════════
    # 5c. Interpolation / Resampling
    # ═══════════════════════════════════════
    with tab_interp:
        st.subheader("Traitement des trous")
        st.markdown(
            "**Objectif :** Combler les **dates absentes** et les **valeurs manquantes** "
            "pour obtenir une série **continue et complète**.\n\n"
            "Le processus se fait en **3 étapes** :\n"
            "1. **✂️ Couper** (optionnel) : réduire la plage pour éviter les gros trous\n"
            "2. **📅 Matérialiser** : créer les lignes pour les dates absentes (reindex)\n"
            "3. **🩹 Interpoler** : combler les NaN par estimation"
        )

        # Afficher le message de succès persistant après rerun
        if st.session_state.get("_ts_cut_success"):
            st.success(st.session_state.pop("_ts_cut_success"))
        if st.session_state.get("_ts_reindex_success"):
            st.success(st.session_state.pop("_ts_reindex_success"))
        if st.session_state.get("_ts_interp_success"):
            st.success(st.session_state.pop("_ts_interp_success"))

        # Recharger df courant
        df = st.session_state.get("df_courant", df)
        analysis = st.session_state.get("ts_continuity_analysis")

        if not analysis:
            st.info("Consultez d'abord l'onglet 5b pour lancer l'analyse de continuité.")
        else:
            # ── Étape 1 : Couper la série ──
            st.markdown("#### ✂️ Étape 1 : Couper la série (optionnel)")
            st.caption("Gardez uniquement un segment continu (utile si de gros trous existent).")

            df_ts = df.copy()
            df_ts[dt_col] = pd.to_datetime(df_ts[dt_col])
            df_ts = df_ts.sort_values(dt_col)
            date_min = df_ts[dt_col].min()
            date_max = df_ts[dt_col].max()

            segments = analysis.get("segments", [])
            if segments and len(segments) > 1:
                st.markdown("**💡 Segments contigus détectés** (classés par taille) :")
                sorted_segs = sorted(segments, key=lambda s: s["n_points"],
                                     reverse=True)
                for i, seg in enumerate(sorted_segs[:5]):
                    start_s = str(seg['start'])[:10]
                    end_s = str(seg['end'])[:10]
                    st.caption(
                        f"➔ Segment {i+1} : **{start_s}** → **{end_s}** "
                        f"({seg['n_points']} points, "
                        f"gap max interne : {seg['max_internal_gap']} périodes)")
                best = sorted_segs[0]
                default_start = best["start"].date() if hasattr(best["start"], 'date') else date_min.date()
                default_end = best["end"].date() if hasattr(best["end"], 'date') else date_max.date()
            else:
                default_start = date_min.date()
                default_end = date_max.date()

            gap_tolerance = st.slider(
                "🎚️ Taille max de trou acceptable (en périodes)",
                min_value=1, max_value=100, value=5,
                key="ts_gap_tolerance",
                help="Les trous plus petits seront comblés par interpolation. "
                     "Les plus grands délimitent les segments.")

            filtered_gaps = [g for g in analysis.get("gaps", [])
                             if g["periodes_manquees"] > gap_tolerance]
            if filtered_gaps:
                boundaries = ([date_min] +
                              [g["debut"] for g in filtered_gaps] +
                              [date_max])
                best_len = 0
                best_pair = (date_min, date_max)
                for i in range(len(boundaries) - 1):
                    seg_start = boundaries[i]
                    seg_end = boundaries[i + 1]
                    seg_df = df_ts[(df_ts[dt_col] >= seg_start) &
                                   (df_ts[dt_col] <= seg_end)]
                    if len(seg_df) > best_len:
                        best_len = len(seg_df)
                        best_pair = (seg_start, seg_end)
                default_start = best_pair[0].date() if hasattr(best_pair[0], 'date') else default_start
                default_end = best_pair[1].date() if hasattr(best_pair[1], 'date') else default_end
                st.info(f"💡 Avec un seuil de **{gap_tolerance} périodes**, "
                        f"**{len(filtered_gaps)}** gros trou(s) restent. "
                        f"Meilleur segment : **{default_start}** → "
                        f"**{default_end}** ({best_len} points).")
            else:
                st.success(f"✅ Avec un seuil de {gap_tolerance}, "
                           "tous les trous sont acceptables.")

            c1, c2 = st.columns(2)
            with c1:
                cut_start = st.date_input("Date de début", value=default_start,
                                           min_value=date_min.date(),
                                           max_value=date_max.date(),
                                           key="ts_cut_start")
            with c2:
                cut_end = st.date_input("Date de fin", value=default_end,
                                         min_value=date_min.date(),
                                         max_value=date_max.date(),
                                         key="ts_cut_end")

            if st.button("✂️ Couper", key="ts_cut_btn"):
                mask = (df_ts[dt_col] >= pd.Timestamp(cut_start)) & \
                       (df_ts[dt_col] <= pd.Timestamp(cut_end))
                df_cut = df_ts[mask].reset_index(drop=True)
                st.session_state["df_courant"] = df_cut
                st.session_state["_ts_cut_success"] = (
                    f"✅ Série coupée : {len(df_cut)} lignes "
                    f"(du {cut_start} au {cut_end}).")

                rapport = st.session_state.get("rapport", {})
                if rapport:
                    ajouter_historique(rapport,
                                       f"Série coupée : {cut_start} → {cut_end} ({len(df_cut)} lignes)")
                    sauvegarder_rapport(rapport)

                st.rerun()

            st.divider()

            # ── Étape 2 : Matérialiser les dates manquantes ──
            st.markdown("#### 📅 Étape 2 : Matérialiser les dates absentes")
            st.markdown(
                "Cette étape crée une **ligne pour chaque date absente** (avec des NaN). "
                "Sans cette étape, l'interpolation ne peut pas combler les trous "
                "car les lignes n'existent tout simplement pas dans le tableau."
            )

            # Compter les dates absentes
            df_cur = st.session_state.get("df_courant", df)
            df_cur_copy = df_cur.copy()
            df_cur_copy[dt_col] = pd.to_datetime(df_cur_copy[dt_col])
            df_cur_copy = df_cur_copy.sort_values(dt_col).reset_index(drop=True)
            n_before = len(df_cur_copy)

            # Estimer le nombre de lignes après reindex
            deltas = df_cur_copy[dt_col].diff().dropna()
            if len(deltas) > 0:
                median_d = deltas.median()
                date_range_total = df_cur_copy[dt_col].max() - df_cur_copy[dt_col].min()
                n_expected = int(date_range_total / median_d) + 1 if median_d.total_seconds() > 0 else n_before
                n_to_create = max(0, n_expected - n_before)
            else:
                n_to_create = 0

            if n_to_create == 0:
                st.success("✅ Toutes les dates sont déjà présentes — pas de reindex nécessaire.")
            else:
                st.info(f"📊 **{n_before}** lignes actuelles → **~{n_expected}** attendues "
                        f"→ **~{n_to_create} date(s) à créer** (remplies de NaN).")

                if st.button("📅 Matérialiser les dates manquantes", type="primary",
                             key="ts_reindex_apply"):
                    df_reindexed = reindex_ts(df_cur, dt_col)
                    n_after = len(df_reindexed)
                    n_created = n_after - n_before
                    st.session_state["df_courant"] = df_reindexed
                    st.session_state["_ts_reindex_success"] = (
                        f"✅ {n_created} date(s) matérialisée(s). "
                        f"{n_after} lignes au total.")

                    rapport = st.session_state.get("rapport", {})
                    if rapport:
                        ajouter_historique(rapport,
                                           f"Reindex TS : {n_created} dates créées ({n_after} lignes)")
                        sauvegarder_rapport(rapport)

                    st.rerun()

            st.divider()

            # ── Étape 3 : Interpolation ──
            st.markdown("#### 🩹 Étape 3 : Interpoler les valeurs manquantes")

            # Recharger df courant (peut avoir changé après reindex)
            df_cur = st.session_state.get("df_courant", df)
            cols_with_na = [c for c in value_cols if c in df_cur.columns
                            and df_cur[c].isna().sum() > 0]

            if not cols_with_na:
                st.success("✅ Aucune valeur manquante à interpoler.")
            else:
                total_na = sum(int(df_cur[c].isna().sum()) for c in cols_with_na)
                st.info(f"📊 **{total_na} valeur(s) manquante(s)** dans "
                        f"**{len(cols_with_na)} colonne(s)**.")

                method = st.selectbox("Méthode d'interpolation", [
                    "linear (interpolation linéaire)",
                    "ffill (propager la dernière valeur)",
                    "bfill (propager la valeur suivante)",
                    "time (interpolation proportionnelle au temps)",
                ], key="ts_interp_method",
                   help="Linéaire = régulière, ffill/bfill = répétition, time = au prorata du temps")

                method_key = method.split(" ")[0]
                limit = st.number_input("Limite de trous consécutifs à combler",
                                         min_value=1, max_value=500, value=10,
                                         key="ts_interp_limit",
                                         help="Au-delà de cette limite, les trous "
                                              "ne seront pas comblés (trop risqué).")

                cols_to_interp = st.multiselect("Colonnes à interpoler",
                                                 cols_with_na,
                                                 default=cols_with_na,
                                                 key="ts_interp_cols",
                                                 help="Colonnes dans lesquelles combler les valeurs manquantes")

                # ── Aperçu avant/après ──
                if cols_to_interp:
                    with st.expander("👁️ Aperçu de l'interpolation (avant d'appliquer)",
                                     expanded=False):
                        df_preview = df_cur.copy()
                        if method_key == "time" and dt_col in df_preview.columns:
                            df_preview = df_preview.set_index(dt_col)

                        for col in cols_to_interp:
                            if method_key in ("ffill", "bfill"):
                                df_preview[col] = df_preview[col].fillna(
                                    method=method_key, limit=limit)
                            else:
                                df_preview[col] = df_preview[col].interpolate(
                                    method=method_key, limit=limit)

                        if method_key == "time" and dt_col not in df_preview.columns:
                            df_preview = df_preview.reset_index()

                        # Graphique avant/après pour la colonne cible
                        if val_col and val_col in cols_to_interp:
                            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5),
                                                     sharey=True)
                            fig.set_facecolor("#F8F9FC")

                            df_cur_plot = df_cur.copy()
                            df_cur_plot[dt_col] = pd.to_datetime(df_cur_plot[dt_col])
                            df_cur_plot = df_cur_plot.sort_values(dt_col)

                            df_preview_plot = df_preview.copy()
                            df_preview_plot[dt_col] = pd.to_datetime(df_preview_plot[dt_col])
                            df_preview_plot = df_preview_plot.sort_values(dt_col)

                            axes[0].plot(df_cur_plot[dt_col], df_cur_plot[val_col],
                                         linewidth=0.8, color="#4F5BD5")
                            axes[0].set_title("AVANT interpolation", fontsize=9,
                                              fontweight="bold")
                            axes[0].grid(True, alpha=0.3)

                            axes[1].plot(df_preview_plot[dt_col],
                                         df_preview_plot[val_col],
                                         linewidth=0.8, color="#059669")
                            axes[1].set_title("APRÈS interpolation", fontsize=9,
                                              fontweight="bold")
                            axes[1].grid(True, alpha=0.3)

                            for ax in axes:
                                ax.set_facecolor("#FFFFFF")
                                ax.spines["top"].set_visible(False)
                                ax.spines["right"].set_visible(False)
                                ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        # Stats résumé
                        remaining = sum(int(df_preview[c].isna().sum())
                                        for c in cols_to_interp
                                        if c in df_preview.columns)
                        filled = total_na - remaining
                        st.markdown(f"**Résultat attendu :** {filled} trous comblés, "
                                    f"{remaining} restants")

                    if st.button("🩹 Appliquer l'interpolation", type="primary",
                                 key="ts_interp_apply"):
                        df_new = df_cur.copy()

                        if method_key == "time" and dt_col in df_new.columns:
                            df_new = df_new.set_index(dt_col)

                        for col in cols_to_interp:
                            if method_key in ("ffill", "bfill"):
                                df_new[col] = df_new[col].fillna(
                                    method=method_key, limit=limit)
                            else:
                                df_new[col] = df_new[col].interpolate(
                                    method=method_key, limit=limit)

                        if method_key == "time" and dt_col not in df_new.columns:
                            df_new = df_new.reset_index()

                        n_filled = int(df_cur.isna().sum().sum() - df_new.isna().sum().sum())
                        st.session_state["df_courant"] = df_new
                        st.session_state["_ts_interp_success"] = (
                            f"✅ Interpolation appliquée ({method_key}) — "
                            f"{n_filled} valeur(s) comblée(s) sur "
                            f"{len(cols_to_interp)} colonne(s).")

                        rapport = st.session_state.get("rapport", {})
                        if rapport:
                            rapport.setdefault("nettoyage", {})["ts_interpolation"] = {
                                "method": method_key, "limit": limit,
                                "columns": cols_to_interp}
                            ajouter_historique(rapport,
                                               f"Interpolation {method_key} : "
                                               f"{n_filled} trous comblés")
                            sauvegarder_rapport(rapport)

                        st.rerun()

    # ═══════════════════════════════════════
    # 5d. Validation
    # ═══════════════════════════════════════
    with tab_valid:
        st.subheader("Validation du nettoyage TS")
        st.markdown(
            "**Objectif :** Vérifier que la série est prête (plus de trous, "
            "plus de doublons) puis valider pour **débloquer l'étape suivante**."
        )
        df = st.session_state.get("df_courant")

        # Vérification doublons
        df_check = df.copy()
        df_check[dt_col] = pd.to_datetime(df_check[dt_col])
        n_dup_remaining = int(df_check[dt_col].duplicated().sum())
        if n_dup_remaining > 0:
            st.warning(f"⚠️ Il reste **{n_dup_remaining}** date(s) en double. "
                       "Retournez à l'onglet 5a.")

        # Vérification NaN
        remaining_na = 0
        for col in value_cols:
            if col in df.columns:
                remaining_na += int(df[col].isna().sum())

        if remaining_na > 0:
            st.warning(f"⚠️ Il reste **{remaining_na}** valeurs manquantes. "
                       "L'interpolation est recommandée (onglet 5c).")
        elif n_dup_remaining == 0:
            st.success("✅ Série propre : aucun doublon, aucune valeur manquante.")

        st.markdown(f"**Données actuelles :** {len(df)} lignes × {len(df.columns)} colonnes")

        if val_col and val_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 3))
            try:
                plot_df = df.copy()
                plot_df[dt_col] = pd.to_datetime(plot_df[dt_col])
                plot_df = plot_df.sort_values(dt_col)
                ax.plot(plot_df[dt_col], plot_df[val_col], linewidth=0.8,
                        color="steelblue")
                ax.set_title(f"Aperçu final : {val_col}", fontsize=10,
                             fontweight="bold")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception:
                plt.close()

        if st.button("✅ Valider le nettoyage TS", type="primary", key="validate_ts_clean"):
            st.session_state["manquantes_done"] = True
            st.session_state["doublons_done"] = True
            st.session_state["outliers_done"] = True
            st.session_state["nettoyage_done"] = True

            rapport = st.session_state.get("rapport", {})
            if rapport:
                sauvegarder_csv(rapport, df, "data_cleaned.csv")
                rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 6)
                ajouter_historique(rapport, "Nettoyage TS terminé")
                sauvegarder_rapport(rapport)

            st.session_state["_pending_step"] = 6
            st.rerun()


def afficher_nettoyage():
    """Étape 5 — Nettoyage : valeurs manquantes puis outliers."""
    st.caption("ÉTAPE 5")

    df = st.session_state.get("df_courant")
    if df is None:
        st.warning("⚠️ Complétez d'abord les étapes précédentes.")
        return

    with st.expander("🎓 Pourquoi nettoyer les données ?", expanded=False):
        st.markdown("""
Un modèle prédictif est comme une recette de cuisine : si les ingrédients sont
avariés, le plat sera raté. Le nettoyage suit un **ordre précis** :

```
  1. VALEURS MANQUANTES         2. DOUBLONS          3. OUTLIERS
  ─────────────────────         ──────────           ────────────
  Boucher les trous             Supprimer les        Gérer les valeurs
  (médiane, moyenne,            lignes en double     extrêmes (capping,
   interpolation…)                                   suppression)
       │                              │                    │
       └──────────── → ─── → ─────────┘──── → ── → ───────┘
              Pourquoi cet ordre ?

  On bouche les trous D'ABORD, sinon les calculs de détection
  de doublons et d'outliers seront faussés par les valeurs manquantes.
```

| Méthode de remplissage | Quand l'utiliser |
|---|---|
| **Médiane** | Nombres avec valeurs extrêmes (recommandé par défaut) |
| **Moyenne** | Nombres sans valeurs extrêmes |
| **Mode** | Catégories (texte) |
| **Interpolation** | Séries temporelles (chaque trou est comblé par le contexte temporel) |
| **Supprimer** | Beaucoup de trous (>50%) dans une colonne → la colonne est inutilisable |
""")

    # ═══════════════════════════════════════
    # Parcours TS : analyse de continuité
    # ═══════════════════════════════════════
    problem_type = st.session_state.get("problem_type", "Régression")
    if problem_type == "Série temporelle":
        _afficher_nettoyage_ts(df)
        return

    # Indicateurs de progression des sous-étapes
    manquantes_done = st.session_state.get("manquantes_done", False)
    doublons_done = st.session_state.get("doublons_done", False)
    outliers_done = st.session_state.get("outliers_done", False)

    c1, c2, c3 = st.columns(3)
    with c1:
        icon1 = "🟢" if manquantes_done else "🔵"
        st.markdown(f"{icon1} **5a. Valeurs manquantes**")
    with c2:
        icon2 = "🟢" if doublons_done else ("🔵" if manquantes_done else "🔒")
        st.markdown(f"{icon2} **5b. Doublons**")
    with c3:
        icon3 = "🟢" if outliers_done else ("🔵" if doublons_done else "🔒")
        st.markdown(f"{icon3} **5c. Outliers**")

    tab_a, tab_b, tab_c = st.tabs(["🕳️ 5a. Valeurs manquantes",
                                    "👯 5b. Doublons",
                                    "📊 5c. Outliers"])

    # ═══════════════════════════════════════
    # 5a. Valeurs manquantes
    # ═══════════════════════════════════════
    with tab_a:
        st.subheader("Boucher les trous")

        cols_with_na = [c for c in df.columns if df[c].isna().sum() > 0]

        if not cols_with_na:
            st.success("✅ Aucune valeur manquante !")
            if not manquantes_done:
                st.session_state["manquantes_done"] = True
                st.rerun()
        else:
            st.markdown(f"**{len(cols_with_na)} colonne(s)** contiennent des trous.")

            missing_strategies = {}
            fixed_values = {}

            for col in cols_with_na:
                na_count = df[col].isna().sum()
                na_pct = round(na_count / len(df) * 100, 1)

                with st.expander(f"🕳️ **{col}** — {na_count} trous ({na_pct}%)",
                                 expanded=(na_pct > 20)):
                    rec = recommend_missing_strategy(df[col], col, na_pct)
                    st.markdown(f"💡 **Recommandation :** {rec['label']}")
                    st.caption(rec["reason"])

                    is_num = pd.api.types.is_numeric_dtype(df[col])
                    options = ["Conserver", "Supprimer la colonne", "Supprimer les lignes"]
                    if is_num:
                        options += ["Remplacer par la moyenne", "Remplacer par la médiane"]
                    options += ["Valeur la plus fréquente", "Valeur fixe"]

                    strat_map = {
                        "Conserver": None, "Supprimer la colonne": "drop_column",
                        "Supprimer les lignes": "drop_rows",
                        "Remplacer par la moyenne": "mean",
                        "Remplacer par la médiane": "median",
                        "Valeur la plus fréquente": "mode",
                        "Valeur fixe": "fixed",
                    }

                    choice = st.selectbox(f"Action pour « {col} »", options,
                                          key=f"na_{col}",
                                          help="Stratégie de remplacement des valeurs manquantes")
                    strategy = strat_map.get(choice)
                    if strategy:
                        missing_strategies[col] = strategy
                    if choice == "Valeur fixe":
                        fixed_values[col] = st.text_input(f"Valeur pour {col}",
                                                           key=f"fix_{col}")

            if missing_strategies and st.button("🩹 Appliquer les corrections",
                                                 type="primary", key="apply_na"):
                df_avant = df.copy()
                df = handle_missing(df, missing_strategies, fixed_values)
                st.session_state["df_courant"] = df
                st.session_state["manquantes_done"] = True

                _afficher_avant_apres(df_avant, df, "Résultat du nettoyage")

                rapport = st.session_state.get("rapport", {})
                if rapport:
                    rapport["nettoyage"]["valeurs_manquantes"] = str(missing_strategies)
                    ajouter_historique(rapport, "Valeurs manquantes traitées")
                    sauvegarder_rapport(rapport)

                st.success(f"✅ Corrections appliquées. {len(df)} lignes restantes.")
                st.rerun()

            # Bouton skip si pas de NA à traiter
            if not missing_strategies:
                if st.button("✅ Passer aux outliers", key="skip_na"):
                    st.session_state["manquantes_done"] = True
                    st.rerun()

    # ═══════════════════════════════════════
    # 5b. Doublons
    # ═══════════════════════════════════════
    with tab_b:
        if not manquantes_done:
            st.info("🔒 **Verrouillé** — Traitez d'abord les valeurs manquantes (onglet 5a).")
        else:
            st.subheader("Supprimer les lignes en double")
            n_dup = int(df.duplicated().sum())
            if n_dup == 0:
                st.success("✅ Aucun doublon détecté !")
                if not doublons_done:
                    st.session_state["doublons_done"] = True
                    st.rerun()
            else:
                st.warning(f"⚠️ **{n_dup} ligne(s) en double** détectée(s).")

                # Aperçu des doublons
                with st.expander(f"👁️ Afficher les {min(n_dup, 20)} premiers doublons"):
                    st.dataframe(df[df.duplicated(keep="first")].head(20),
                                 use_container_width=True)

                if st.button("🗑️ Supprimer les doublons", type="primary", key="apply_dup"):
                    df_avant = df.copy()
                    df = df.drop_duplicates().reset_index(drop=True)
                    st.session_state["df_courant"] = df
                    st.session_state["doublons_done"] = True

                    _afficher_avant_apres(df_avant, df, "Résultat de la dédoublonnage")

                    rapport = st.session_state.get("rapport", {})
                    if rapport:
                        rapport["nettoyage"]["doublons_supprimes"] = n_dup
                        ajouter_historique(rapport, f"{n_dup} doublons supprimés")
                        sauvegarder_rapport(rapport)

                    st.success(f"✅ {n_dup} doublons supprimés. {len(df)} lignes restantes.")
                    st.rerun()

                if st.button("⏭️ Conserver les doublons", key="skip_dup"):
                    st.session_state["doublons_done"] = True
                    st.rerun()

    # ═══════════════════════════════════════
    # 5c. Outliers
    # ═══════════════════════════════════════
    with tab_c:
        if not doublons_done:
            st.info("🔒 **Verrouillé** — Traitez d'abord les doublons (onglet 5b).")
            return

        st.subheader("Gérer les valeurs extrêmes")

        num_cols = get_numeric_columns(df)
        outlier_strategies = {}
        found = False

        for col in num_cols:
            info = detect_outliers_iqr(df[col])
            if info["count"] > 0:
                found = True
                with st.expander(f"📊 **{col}** — {info['count']} outlier(s)"):
                    # Mini boxplot
                    fig, ax = plt.subplots(figsize=(4, 0.9))
                    fig.set_facecolor("#F8F9FC")
                    df[col].dropna().plot.box(ax=ax, vert=False, patch_artist=True,
                                              boxprops=dict(facecolor=CHART_COLORS["accent"],
                                                            edgecolor=CHART_COLORS["primary"]),
                                              medianprops=dict(color=CHART_COLORS["danger"], linewidth=2),
                                              flierprops=dict(marker="o", markerfacecolor=CHART_COLORS["warning"],
                                                              markeredgecolor="white", markersize=5))
                    ax.set_facecolor("#FFFFFF")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_title(f"{col}", fontsize=9, fontweight="bold")
                    ax.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    rec = recommend_outlier_strategy(df[col], col, info["count"])
                    st.markdown(f"💡 {rec['label']}")
                    st.caption(rec["reason"])

                    choice = st.selectbox(f"Action pour « {col} »",
                                          ["Conserver", "Supprimer", "Plafonner", "Log transform"],
                                          key=f"out_{col}",
                                          help="Conserver = ignorer, Plafonner = borner aux limites IQR")
                    strat_map = {"Conserver": None, "Supprimer": "drop",
                                 "Plafonner": "cap", "Log transform": "log"}
                    s = strat_map.get(choice)
                    if s:
                        outlier_strategies[col] = s

        if not found and num_cols:
            st.success("✅ Aucun outlier détecté.")

        if outlier_strategies and st.button("🩹 Traiter les outliers", type="primary",
                                             key="apply_outliers"):
            df_avant = df.copy()
            df = handle_outliers(df, outlier_strategies)
            st.session_state["df_courant"] = df
            st.session_state["outliers_done"] = True

            _afficher_avant_apres(df_avant, df, "Résultat du traitement")

            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["nettoyage"]["outliers"] = str(outlier_strategies)
                ajouter_historique(rapport, "Outliers traités")
                sauvegarder_rapport(rapport)

            st.success(f"✅ {len(df)} lignes restantes.")
            st.rerun()

        if not outlier_strategies or not found:
            if st.button("✅ Terminer le nettoyage", key="finish_clean"):
                st.session_state["outliers_done"] = True
                st.session_state["nettoyage_done"] = True
                rapport = st.session_state.get("rapport", {})
                if rapport:
                    sauvegarder_csv(rapport, df, "data_cleaned.csv")
                    rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 6)
                    ajouter_historique(rapport, "Nettoyage terminé")
                    sauvegarder_rapport(rapport)
                st.session_state["_pending_step"] = 6
                st.rerun()

    # Marquer le nettoyage comme terminé si toutes les sous-étapes sont faites
    if manquantes_done and doublons_done and outliers_done and not st.session_state.get("nettoyage_done"):
        st.session_state["nettoyage_done"] = True
        rapport = st.session_state.get("rapport", {})
        if rapport:
            sauvegarder_csv(rapport, df, "data_cleaned.csv")
            rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 6)
            ajouter_historique(rapport, "Nettoyage terminé")
            sauvegarder_rapport(rapport)


            ajouter_historique(rapport, "Nettoyage terminé")
            sauvegarder_rapport(rapport)


def _afficher_transformation_ts(df: pd.DataFrame):
    """Transformations spécifiques aux séries temporelles."""
    val_col = st.session_state.get("ts_value_col")
    dt_col = st.session_state.get("ts_datetime_col")

    if not val_col or val_col not in df.columns:
        st.warning("⚠️ Colonne cible TS non définie. Retournez au diagnostic.")
        return

    tab_reco, tab_transform, tab_scale, tab_horizon, tab_valid = st.tabs([
        "📊 6a. Analyse & Recommandations",
        "🔧 6b. Transformations",
        "📐 6c. Scaling",
        "🎯 6d. Prédiction horizon",
        "✅ 6e. Valider",
    ])

    # ═══════════════════════════════════════
    # 6a. Recommandations automatiques
    # ═══════════════════════════════════════
    with tab_reco:
        st.subheader("Analyse de la série et recommandations")

        # Préparer la série
        try:
            ts_series = prepare_timeseries(df, dt_col, val_col)
        except Exception:
            ts_series = pd.Series(df[val_col].values, dtype=float)

        # Recommandations de transformations
        ts_recos = recommend_ts_transforms(ts_series)

        if ts_recos:
            for reco in ts_recos:
                priority_icon = {"high": "🔴", "medium": "🟠",
                                 "low": "🟢", "info": "💡"}.get(
                    reco["priority"], "💡")
                with st.expander(
                    f"{priority_icon} **{reco['transform'].upper()}** — {reco['reason'][:80]}…"
                    if len(reco['reason']) > 80 else
                    f"{priority_icon} **{reco['transform'].upper()}** — {reco['reason']}",
                    expanded=(reco["priority"] in ("high", "medium"))
                ):
                    st.markdown(reco["reason"])
                    st.markdown(f"**Comment :** {reco['how']}")
        else:
            st.success("✅ La série ne nécessite pas de transformation particulière.")

        # Détection de saisonnalité
        st.divider()
        st.markdown("### 🌊 Analyse de la saisonnalité")
        with st.spinner("Détection de la saisonnalité…"):
            seasonal = detect_seasonality(ts_series)
            st.session_state["ts_seasonality"] = seasonal

        for rec in seasonal.get("recommendations", []):
            st.markdown(rec)

        if seasonal["has_seasonality"]:
            st.info(f"📌 Période détectée : **{seasonal['period']}** | "
                    f"Force : **{seasonal['strength']:.1%}** | "
                    f"Type : **{seasonal['model_type']}**")

        # Visualisation rapide
        st.divider()
        st.markdown("### 📈 Aperçu des distributions")
        st.caption("📊 **Skewness (asymétrie)** — mesure si la distribution est décentrée. "
                   "Cible : entre **-0.5** et **0.5** (symétrique). "
                   "Au-delà de ±1, une transformation (log, √) est recommandée.")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            ts_series.hist(ax=ax, bins=30, color="steelblue", edgecolor="white")
            ax.set_title(f"Distribution de {val_col}", fontsize=10)
            skew = ts_series.skew()
            ax.axvline(ts_series.mean(), color="red", linestyle="--", linewidth=1)
            ax.set_xlabel(_skew_label(skew), fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
        with c2:
            if ts_series.min() > 0:
                fig, ax = plt.subplots(figsize=(5, 3))
                np.log1p(ts_series).hist(ax=ax, bins=30, color="teal", edgecolor="white")
                ax.set_title(f"Distribution après LOG", fontsize=10)
                log_skew = np.log1p(ts_series).skew()
                ax.set_xlabel(_skew_label(log_skew), fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Log non applicable (valeurs ≤ 0).")

    # ═══════════════════════════════════════
    # 6b. Appliquer des transformations
    # ═══════════════════════════════════════
    with tab_transform:
        st.subheader("Transformations disponibles")
        st.caption("**Objectif :** appliquer des transformations mathématiques "
                   "pour stabiliser la variance, réduire l'asymétrie ou retirer "
                   "la tendance. Vous pouvez choisir les colonnes concernées.")
        st.caption("🎯 **Skewness cible : entre -0.5 et 0.5** (distribution symétrique). "
                   "Si le skew est > 1 ou < -1, une transformation est recommandée.")

        # Colonnes numériques disponibles (hors datetime)
        numeric_cols = [c for c in df.select_dtypes(include="number").columns]
        if not numeric_cols:
            st.warning("⚠️ Aucune colonne numérique trouvée.")
            return

        transform_action = st.selectbox("Transformation à appliquer", [
            "Log (stabiliser la variance / asymétrie)",
            "Racine carrée (atténuer l'asymétrie)",
            "Différenciation (retirer la tendance)",
            "Aucune transformation",
        ], key="ts_transform_action",
           help="Choisissez une transformation pour stabiliser la série")

        if transform_action == "Aucune transformation":
            st.info("Aucune transformation sélectionnée.")

        elif transform_action.startswith("Log"):
            cols_log = st.multiselect(
                "Colonnes à transformer", numeric_cols,
                default=[val_col] if val_col in numeric_cols else [],
                key="ts_log_cols",
                help="Colonnes sur lesquelles appliquer la transformation logarithmique")

            if cols_log:
                # ── Aperçu avant/après ──
                df_preview = df.copy()
                warnings_cols = []
                for col in cols_log:
                    min_v = float(df_preview[col].min())
                    if min_v <= 0:
                        df_preview[col] = np.log1p(df_preview[col] - min_v)
                        warnings_cols.append(col)
                    else:
                        df_preview[col] = np.log1p(df_preview[col])

                if warnings_cols:
                    st.warning(f"⚠️ Valeurs ≤ 0 dans : {', '.join(warnings_cols)} "
                               f"→ log1p(x − min) sera utilisé.")

                st.markdown("#### 👁️ Aperçu de l'impact")
                n_show = min(len(cols_log), 4)
                for i in range(0, n_show, 2):
                    row_cols = cols_log[i:i+2]
                    st_cols = st.columns(len(row_cols) * 2)
                    for j, col in enumerate(row_cols):
                        with st_cols[j * 2]:
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            df[col].dropna().hist(ax=ax, bins=30, color="steelblue",
                                                   edgecolor="white", alpha=0.85)
                            skew_before = df[col].dropna().skew()
                            ax.set_title(f"AVANT — {col}", fontsize=9, fontweight="bold")
                            ax.set_xlabel(_skew_label(skew_before), fontsize=8)
                            ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        with st_cols[j * 2 + 1]:
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            df_preview[col].dropna().hist(ax=ax, bins=30, color="teal",
                                                          edgecolor="white", alpha=0.85)
                            skew_after = df_preview[col].dropna().skew()
                            ax.set_title(f"APRÈS LOG — {col}", fontsize=9, fontweight="bold")
                            ax.set_xlabel(_skew_label(skew_after), fontsize=8)
                            ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                if len(cols_log) > n_show:
                    st.caption(f"… et {len(cols_log) - n_show} autre(s) colonne(s).")

                if st.button("📐 Appliquer le Log", type="primary", key="ts_apply_log"):
                    st.session_state["df_courant"] = df_preview
                    st.session_state["ts_log_applied"] = True
                    st.session_state["ts_log_applied_cols"] = cols_log
                    rapport = st.session_state.get("rapport", {})
                    if rapport:
                        rapport.setdefault("nettoyage", {})["ts_transform"] = {
                            "type": "log", "columns": cols_log}
                        ajouter_historique(rapport,
                            f"Log appliqué sur {', '.join(cols_log)}")
                        sauvegarder_rapport(rapport)
                    st.success(f"✅ Log appliqué sur {len(cols_log)} colonne(s). "
                               "N'oubliez pas d'inverser (expm1) sur les prédictions.")
                    st.rerun()

        elif transform_action.startswith("Racine"):
            cols_sqrt = st.multiselect(
                "Colonnes à transformer", numeric_cols,
                default=[val_col] if val_col in numeric_cols else [],
                key="ts_sqrt_cols",
                help="Colonnes sur lesquelles appliquer la racine carrée")

            if cols_sqrt:
                # ── Aperçu avant/après ──
                df_preview = df.copy()
                for col in cols_sqrt:
                    min_v = float(df_preview[col].min())
                    if min_v < 0:
                        df_preview[col] = np.sqrt(df_preview[col] - min_v)
                    else:
                        df_preview[col] = np.sqrt(df_preview[col])

                st.markdown("#### 👁️ Aperçu de l'impact")
                n_show = min(len(cols_sqrt), 4)
                for i in range(0, n_show, 2):
                    row_cols = cols_sqrt[i:i+2]
                    st_cols = st.columns(len(row_cols) * 2)
                    for j, col in enumerate(row_cols):
                        with st_cols[j * 2]:
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            df[col].dropna().hist(ax=ax, bins=30, color="steelblue",
                                                   edgecolor="white", alpha=0.85)
                            skew_before = df[col].dropna().skew()
                            ax.set_title(f"AVANT — {col}", fontsize=9, fontweight="bold")
                            ax.set_xlabel(_skew_label(skew_before), fontsize=8)
                            ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        with st_cols[j * 2 + 1]:
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            df_preview[col].dropna().hist(ax=ax, bins=30, color="darkorange",
                                                          edgecolor="white", alpha=0.85)
                            skew_after = df_preview[col].dropna().skew()
                            ax.set_title(f"APRÈS √ — {col}", fontsize=9, fontweight="bold")
                            ax.set_xlabel(_skew_label(skew_after), fontsize=8)
                            ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                if len(cols_sqrt) > n_show:
                    st.caption(f"… et {len(cols_sqrt) - n_show} autre(s) colonne(s).")

                if st.button("📐 Appliquer √", type="primary", key="ts_apply_sqrt"):
                    st.session_state["df_courant"] = df_preview
                    rapport = st.session_state.get("rapport", {})
                    if rapport:
                        rapport.setdefault("nettoyage", {})["ts_transform"] = {
                            "type": "sqrt", "columns": cols_sqrt}
                        ajouter_historique(rapport,
                            f"Racine carrée sur {', '.join(cols_sqrt)}")
                        sauvegarder_rapport(rapport)
                    st.success(f"✅ Racine carrée appliquée sur {len(cols_sqrt)} colonne(s).")
                    st.rerun()

        elif transform_action.startswith("Différenciation"):
            cols_diff = st.multiselect(
                "Colonnes à différencier", numeric_cols,
                default=[val_col] if val_col in numeric_cols else [],
                key="ts_diff_cols",
                help="Colonnes à différencier pour retirer la tendance")

            diff_order = st.number_input("Ordre de différenciation",
                                          min_value=1, max_value=3, value=1,
                                          key="ts_diff_order",
                                          help="1 = retirer la tendance, 2 = retirer l'accélération")

            if cols_diff:
                # ── Aperçu avant/après ──
                df_preview = df.copy()
                for col in cols_diff:
                    for _ in range(diff_order):
                        df_preview[col] = df_preview[col].diff()
                df_preview = df_preview.dropna(
                    subset=cols_diff).reset_index(drop=True)

                st.markdown("#### 👁️ Aperçu de l'impact")
                n_show = min(len(cols_diff), 4)
                for i in range(0, n_show, 2):
                    row_cols = cols_diff[i:i+2]
                    st_cols = st.columns(len(row_cols) * 2)
                    for j, col in enumerate(row_cols):
                        with st_cols[j * 2]:
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            df[col].dropna().hist(ax=ax, bins=30, color="steelblue",
                                                   edgecolor="white", alpha=0.85)
                            ax.set_title(f"AVANT — {col}", fontsize=9, fontweight="bold")
                            ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        with st_cols[j * 2 + 1]:
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            df_preview[col].dropna().hist(ax=ax, bins=30, color="purple",
                                                          edgecolor="white", alpha=0.85)
                            ax.set_title(f"APRÈS diff({diff_order}) — {col}",
                                         fontsize=9, fontweight="bold")
                            ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                if len(cols_diff) > n_show:
                    st.caption(f"… et {len(cols_diff) - n_show} autre(s) colonne(s).")

                st.caption(f"⚠️ La différenciation supprimera "
                           f"{len(df) - len(df_preview)} ligne(s).")

                if st.button("📐 Appliquer la différenciation", type="primary",
                              key="ts_apply_diff"):
                    st.session_state["df_courant"] = df_preview
                    rapport = st.session_state.get("rapport", {})
                    if rapport:
                        rapport.setdefault("nettoyage", {})["ts_transform"] = {
                            "type": f"diff(order={diff_order})",
                            "columns": cols_diff}
                        ajouter_historique(rapport,
                            f"Différenciation d'ordre {diff_order} sur "
                            f"{', '.join(cols_diff)}")
                        sauvegarder_rapport(rapport)
                    st.success(f"✅ Différenciation d'ordre {diff_order} appliquée "
                               f"sur {len(cols_diff)} colonne(s).")
                    st.rerun()

    # ═══════════════════════════════════════
    # 6c. Scaling (optionnel)
    # ═══════════════════════════════════════
    with tab_scale:
        st.subheader("Mise à l'échelle (optionnel)")
        st.caption("**Objectif :** normaliser les colonnes numériques pour "
                   "améliorer la convergence de certains modèles. "
                   "Pas toujours nécessaire pour les séries temporelles.")

        numeric_cols_scale = [c for c in df.select_dtypes(include="number").columns]
        if not numeric_cols_scale:
            st.info("Aucune colonne numérique à scaler.")
        else:
            rec = recommend_normalization(df, numeric_cols_scale)
            if rec.get("needed"):
                st.markdown(f"💡 **Recommandé :** {rec.get('reason', '')}")
            else:
                st.markdown(f"💡 **Optionnel :** {rec.get('reason', '')}")

            cols_to_norm = st.multiselect("Colonnes à scaler",
                                           numeric_cols_scale,
                                           key="ts_norm_cols",
                                           help="Colonnes numériques à normaliser")
            method_scale = st.radio("Méthode",
                                    ["Standard (moyenne=0, écart-type=1)",
                                     "MinMax (entre 0 et 1)"],
                                    key="ts_norm_method")
            method_map = {"Standard (moyenne=0, écart-type=1)": "standard",
                          "MinMax (entre 0 et 1)": "minmax"}

            if cols_to_norm:
                if st.button("📐 Appliquer le scaling", type="primary",
                              key="ts_apply_scale"):
                    df_avant = df.copy()
                    df, scaler = normalize_columns(df, cols_to_norm,
                                                    method=method_map[method_scale])
                    st.session_state["df_courant"] = df
                    st.session_state["scaler"] = scaler
                    st.session_state["scaled_columns"] = cols_to_norm

                    _afficher_avant_apres(df_avant, df, "Résultat du scaling")

                    rapport = st.session_state.get("rapport", {})
                    if rapport:
                        rapport.setdefault("nettoyage", {})["normalisation"] = method_map[method_scale]
                        rapport["nettoyage"]["scaled_columns"] = cols_to_norm
                        ajouter_historique(rapport,
                            f"Scaling {method_map[method_scale]} sur "
                            f"{', '.join(cols_to_norm)}")
                        sauvegarder_objet(rapport, scaler, "scaler.pkl")
                        sauvegarder_rapport(rapport)

                    st.success("✅ Scaling appliqué.")
                    st.rerun()

        if st.button("⏭️ Passer (pas de scaling)", key="ts_skip_scale"):
            st.session_state["scaling_done"] = True

    # ═══════════════════════════════════════
    # 6d. Prédiction horizon (multivarié)
    # ═══════════════════════════════════════
    with tab_horizon:
        st.subheader("Prédiction à horizon (série multivariée)")
        st.markdown(
            "**Objectif :** Prédire la cible **N jours à l'avance** à partir "
            "des valeurs actuelles et des variables exogènes (météo, débits…).\n\n"
            "L'app va automatiquement :\n"
            "1. Créer la **cible décalée** (ex: niveau à t+15)\n"
            "2. Créer des **features lag** (valeurs passées)\n"
            "3. Créer des **features lead** (cumul futur, ex: pluie prévue)\n"
            "4. Basculer en **régression supervisée** pour l'entraînement\n"
        )

        numeric_cols = [c for c in df.select_dtypes(include="number").columns]
        other_cols = [c for c in numeric_cols if c != val_col]

        # Horizon
        horizon = st.number_input(
            "📅 Horizon de prédiction (en périodes / jours)",
            min_value=1, max_value=365, value=15,
            key="ts_horizon",
            help="Nombre de périodes dans le futur à prédire. "
                 "Ex: 15 pour prédire le niveau du barrage dans 15 jours.")

        st.divider()

        # Lag features
        st.markdown("### 🔙 Variables lag (valeurs actuelles et passées)")
        st.caption(
            "Les **lags** capturent l'inertie : le niveau actuel et des jours "
            "précédents, les débits récents, etc. Ce sont les données "
            "**disponibles au moment de la prédiction**.")

        lag_cols = st.multiselect(
            "Colonnes pour les lags", numeric_cols,
            default=numeric_cols,
            key="ts_horizon_lag_cols",
            help="Colonnes dont créer des décalages temporels (valeurs passées)")
        lag_values = st.text_input(
            "Lags à créer (séparés par des virgules)",
            value="0, 1, 2, 3, 7, 14",
            key="ts_horizon_lags",
            help="0 = valeur du jour, 1 = t−1, 7 = t−7, etc.")

        st.divider()

        # Lead features (cumul futur)
        st.markdown("### ⏩ Variables lead (prévisions futures)")
        st.caption(
            "Les **leads** injectent des prévisions connues à l'avance "
            "(ex: la pluviométrie prévue à 15 jours par Météo). "
            "L'agrégation (somme, moyenne…) cumule les valeurs sur l'horizon.")

        lead_cols = st.multiselect(
            "Colonnes pour les leads (prévisions futures)",
            other_cols,
            default=[],
            key="ts_horizon_lead_cols",
            help="Sélectionnez les variables dont vous disposez d'une prévision "
                 "future (ex: pluviométrie, température prévue).")
        lead_agg = st.selectbox(
            "Agrégation sur l'horizon",
            ["sum (cumul)", "mean (moyenne)", "max", "min"],
            key="ts_horizon_lead_agg",
            help="Comment agréger les valeurs futures sur la fenêtre de prédiction")
        lead_agg_key = lead_agg.split(" ")[0]

        # Rolling features
        st.divider()
        st.markdown("### 📈 Moyennes glissantes (rolling)")
        st.caption(
            "Capturent les **tendances récentes** : moyenne et écart-type "
            "sur une fenêtre glissante passée.")
        rolling_cols = st.multiselect(
            "Colonnes pour les moyennes glissantes",
            numeric_cols,
            default=[val_col] if val_col in numeric_cols else [],
            key="ts_horizon_rolling_cols",
            help="Colonnes pour calculer moyenne et écart-type glissants")
        rolling_windows_txt = st.text_input(
            "Tailles des fenêtres (séparées par des virgules)",
            value="7, 14, 30",
            key="ts_horizon_rolling_windows")

        st.divider()

        # Aperçu & application
        if st.button("🚀 Construire les features horizon", type="primary",
                      key="ts_apply_horizon"):
            df_h = df.copy()
            if dt_col and dt_col in df_h.columns:
                df_h = df_h.sort_values(dt_col).reset_index(drop=True)

            created_cols = []

            # 1. Cible décalée
            df_h, target_horizon_col = create_horizon_target(
                df_h, val_col, horizon, dt_col)
            created_cols.append(target_horizon_col)

            # 2. Lags
            try:
                lags_list = [int(x.strip()) for x in lag_values.split(",") if x.strip()]
            except ValueError:
                lags_list = [0, 1, 2, 3]

            for col in lag_cols:
                for lag in lags_list:
                    if lag == 0:
                        # lag 0 = garder la colonne originale (feature "actuelle")
                        continue
                    new_col = f"{col}_lag{lag}"
                    df_h[new_col] = df_h[col].shift(lag)
                    created_cols.append(new_col)

            # 3. Lead features
            for col in lead_cols:
                df_h, lead_col = create_lead_features(
                    df_h, col, horizon, agg=lead_agg_key, datetime_col=dt_col)
                created_cols.append(lead_col)

            # 4. Rolling features
            try:
                rolling_wins = [int(x.strip()) for x in rolling_windows_txt.split(",")
                                if x.strip()]
            except ValueError:
                rolling_wins = [7, 14]

            for col in rolling_cols:
                df_h, r_created = create_rolling_features(
                    df_h, col, windows=rolling_wins, datetime_col=dt_col)
                created_cols.extend(r_created)

            # 5. Supprimer les lignes avec NaN (début/fin de série, lags, rolling)
            n_before = len(df_h)
            # Nettoyer toutes les colonnes numériques (lags, rolling, target, features)
            numeric_cols_h = df_h.select_dtypes(include="number").columns.tolist()
            df_h = df_h.dropna(subset=numeric_cols_h).reset_index(drop=True)
            n_lost = n_before - len(df_h)

            # 6. Définir les features et basculer en régression
            feature_candidates = [c for c in df_h.columns
                                  if c != target_horizon_col
                                  and c != dt_col
                                  and c != val_col]

            st.session_state["df_courant"] = df_h
            st.session_state["target_col"] = target_horizon_col
            st.session_state["feature_cols"] = feature_candidates
            st.session_state["problem_type"] = "Régression"
            st.session_state["ts_horizon_mode"] = True
            st.session_state["ts_horizon_value"] = horizon

            # Valider automatiquement l'étape 6 (les features sont prêtes)
            st.session_state["encoding_done"] = True
            st.session_state["scaling_done"] = True
            st.session_state["transformation_done"] = True

            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["colonne_cible"] = target_horizon_col
                rapport["colonnes_features"] = feature_candidates
                rapport.setdefault("nettoyage", {})["ts_horizon"] = {
                    "horizon": horizon,
                    "lag_cols": lag_cols,
                    "lags": lags_list,
                    "lead_cols": lead_cols,
                    "lead_agg": lead_agg_key,
                    "rolling_cols": rolling_cols,
                    "rolling_windows": rolling_wins,
                    "target_col": target_horizon_col,
                    "n_features": len(feature_candidates),
                }
                ajouter_historique(rapport,
                    f"Features horizon t+{horizon} : "
                    f"{len(created_cols)} colonnes créées, "
                    f"{n_lost} lignes perdues, "
                    f"bascule en régression")
                sauvegarder_rapport(rapport)

            st.success(
                f"✅ **{len(created_cols)} features** créées · "
                f"Cible : `{target_horizon_col}` · "
                f"{len(feature_candidates)} variables explicatives · "
                f"{n_lost} lignes perdues (bords) · "
                f"**Mode : Régression supervisée**")

            # Auto-avancer vers l'étape 7
            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 7)
                sauvegarder_rapport(rapport)
            st.session_state["_pending_step"] = 7
            st.rerun()

        # Afficher un résumé si déjà appliqué
        if st.session_state.get("ts_horizon_mode"):
            h_val = st.session_state.get("ts_horizon_value", "?")
            t_col = st.session_state.get("target_col", "?")
            f_cols = st.session_state.get("feature_cols", [])
            st.info(
                f"🎯 **Mode horizon actif** : prédiction à t+{h_val}\n\n"
                f"Cible : `{t_col}` · {len(f_cols)} features · "
                f"Entraînement en **régression supervisée**")

    # ═══════════════════════════════════════
    # 6e. Validation
    # ═══════════════════════════════════════
    with tab_valid:
        st.subheader("Valider les transformations TS")

        st.markdown(f"**Données actuelles :** {len(df)} lignes × {len(df.columns)} colonnes")

        if st.session_state.get("ts_log_applied"):
            st.info("📌 Log appliqué sur la cible — les prédictions devront "
                    "être inversées (np.expm1).")

        rapport = st.session_state.get("rapport", {})
        nett = rapport.get("nettoyage", {})
        if nett.get("normalisation"):
            scaled = nett.get("scaled_columns", [])
            st.info(f"📌 Scaling {nett['normalisation']} appliqué sur "
                    f"{', '.join(scaled)}")

        seasonal = st.session_state.get("ts_seasonality", {})
        if seasonal.get("has_seasonality"):
            st.info(f"📌 Saisonnalité détectée (période={seasonal['period']}) "
                    f"→ SARIMA sera proposé à l'étape suivante.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⏭️ Continuer sans transformation", key="skip_transfo_ts"):
                _valider_transformation_ts(df)
        with c2:
            if st.button("✅ Valider les transformations", type="primary",
                          key="validate_transfo_ts"):
                _valider_transformation_ts(df)


def _valider_transformation_ts(df: pd.DataFrame):
    """Valide l'étape transformation TS et passe à la modélisation."""
    st.session_state["encoding_done"] = True
    st.session_state["scaling_done"] = True
    st.session_state["transformation_done"] = True

    rapport = st.session_state.get("rapport", {})
    if rapport:
        rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 7)
        # En mode horizon, persister la bascule régression dans le rapport
        if st.session_state.get("ts_horizon_mode"):
            rapport["problem_type"] = "Régression"
            rapport["ts_horizon_mode"] = True
        ajouter_historique(rapport, "Transformations TS validées")
        sauvegarder_rapport(rapport)
        sauvegarder_csv(rapport, df, "data_cleaned.csv")

    # Auto-avancer vers l'étape 7 — Modélisation
    st.session_state["_pending_step"] = 7
    st.rerun()


def afficher_transformation():
    """Étape 6 — Transformation : encoding → scaling → feature engineering."""
    st.caption("ÉTAPE 6")

    df = st.session_state.get("df_courant")
    if df is None:
        st.warning("⚠️ Données non disponibles.")
        return

    if not st.session_state.get("nettoyage_done"):
        st.info("🔒 **Verrouillé** — Terminez d'abord le nettoyage (étape 5).")
        return

    # Shortcut pour séries temporelles — transformations spécifiques TS
    problem_type = st.session_state.get("problem_type", "Régression")
    is_ts = problem_type == "Série temporelle" or st.session_state.get("ts_horizon_mode")
    if is_ts:
        _afficher_transformation_ts(df)
        return

    with st.expander("🎓 Pourquoi transformer les données ?", expanded=False):
        st.markdown("""
Les modèles de ML ne comprennent que les **nombres**. Il faut donc :

```
  ENCODING                  SCALING                 FEATURE ENGINEERING
  ────────                  ───────                 ───────────────────
  Texte → Nombres           Mise à la même échelle  Créer de nouvelles
                                                    variables

  "rouge" → [1,0,0]         Âge : 0–100             surface × étages
  "bleu"  → [0,1,0]         Salaire : 0–100 000     = volume habitable
  "vert"  → [0,0,1]               ↓
  (One-Hot Encoding)        Tout ramené entre
                            0 et 1
```

| Étape | Quand c'est nécessaire |
|---|---|
| **Encoding** | Dès qu'il y a des colonnes texte/catégories |
| **Scaling** | Quand les colonnes ont des échelles très différentes (âge vs salaire) |
| **Feature Engineering** | Pour enrichir les données avec des combinaisons, transformations… |

> **💡 Chaque étape est recommandée automatiquement.** Suivez les suggestions en vert.
""")

    encoding_done = st.session_state.get("encoding_done", False)
    scaling_done = st.session_state.get("scaling_done", False)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"{'🟢' if encoding_done else '🔵'} **6a. Encoding**")
    with c2:
        st.markdown(f"{'🟢' if scaling_done else ('🔵' if encoding_done else '🔒')} **6b. Scaling**")
    with c3:
        st.markdown("🟡 **6c. Feature Engineering** (optionnel)")

    tab_a, tab_b, tab_c = st.tabs([
        "🏷️ 6a. Encoding", "📐 6b. Scaling", "🔧 6c. Feature Engineering"
    ])

    # ═══════════════════════════════════════
    # 6a. Encoding catégoriel
    # ═══════════════════════════════════════
    with tab_a:
        st.subheader("Convertir le texte en chiffres")
        cat_cols = get_categorical_columns(df)

        if not cat_cols:
            st.success("✅ Aucune colonne texte à convertir.")
            if not encoding_done:
                st.session_state["encoding_done"] = True
                st.rerun()
        else:
            encode_strategies = {}
            for col in cat_cols:
                n_unique = df[col].nunique()
                with st.expander(f"🏷️ **{col}** — {n_unique} catégorie(s)"):
                    examples = df[col].dropna().unique()[:8]
                    st.markdown(f"Exemples : *{', '.join(str(v) for v in examples)}*")

                    rec = recommend_encoding(df[col], col)
                    st.markdown(f"💡 {rec['label']}")
                    st.caption(rec["reason"])

                    if n_unique > MAX_ONEHOT_CARDINALITY:
                        st.warning(f"⚠️ {n_unique} catégories — One-Hot déconseillé.")

                    options = {
                        "One-Hot (une colonne par catégorie)": "one_hot",
                        "Label (un chiffre par catégorie)": "label",
                        "Target (encoder selon la cible)": "target",
                        "Supprimer": "drop",
                    }
                    choice = st.selectbox(f"Méthode pour « {col} »",
                                          list(options.keys()), key=f"enc_{col}",
                                          help="One-Hot pour peu de catégories, Label pour ordinal, Target pour prédictif")
                    encode_strategies[col] = options[choice]

            c_enc1, c_enc2 = st.columns([3, 1])
            with c_enc1:
                do_enc = st.button("🏷️ Appliquer l'encodage", type="primary", key="apply_enc")
            with c_enc2:
                skip_enc = st.button("⏭️ Passer", key="skip_enc")

            if skip_enc:
                st.session_state["encoding_done"] = True
                st.rerun()

            if do_enc:
                df_avant = df.copy()
                target_col = st.session_state.get("target_col")

                cols_drop = [c for c, s in encode_strategies.items() if s == "drop"]
                encode_only = {c: s for c, s in encode_strategies.items() if s != "drop"}

                for c in cols_drop:
                    df = df.drop(columns=[c])
                if encode_only:
                    df, encoders = encode_categorical(df, encode_only, target_col=target_col)
                    st.session_state["encoders"] = encoders

                st.session_state["df_courant"] = df
                st.session_state["encoding_done"] = True

                _afficher_avant_apres(df_avant, df, "Résultat de l'encodage")

                rapport = st.session_state.get("rapport", {})
                if rapport:
                    rapport["nettoyage"]["encodage"] = str(list(encode_strategies.keys()))
                    ajouter_historique(rapport, f"Encodage de {len(encode_strategies)} colonnes")
                    if encode_only:
                        sauvegarder_objet(rapport, st.session_state.get("encoders", {}), "encoders.pkl")
                    sauvegarder_rapport(rapport)

                st.success("✅ Encodage appliqué.")
                st.rerun()

    # ═══════════════════════════════════════
    # 6b. Scaling
    # ═══════════════════════════════════════
    with tab_b:
        if not encoding_done:
            st.info("🔒 **Verrouillé** — Terminez d'abord l'encodage (6a).")
            return

        st.subheader("Mise à l'échelle")
        num_cols = get_numeric_columns(df)

        if not num_cols:
            st.info("Aucune colonne numérique.")
            if not scaling_done:
                st.session_state["scaling_done"] = True
                st.rerun()
        else:
            rec = recommend_normalization(df, num_cols)
            if rec.get("needed"):
                st.markdown(f"💡 **Recommandé :** {rec.get('reason', '')}")
            else:
                st.markdown(f"💡 **Optionnel :** {rec.get('reason', '')}")

            cols_to_norm = st.multiselect("Colonnes à scaler", num_cols, key="norm_cols",
                                          help="Colonnes numériques à normaliser (mise à l'échelle)")
            method = st.radio("Méthode",
                              ["Standard (moyenne=0, écart-type=1)",
                               "MinMax (entre 0 et 1)",
                               "Aucune"],
                              key="norm_method")
            method_map = {"Standard (moyenne=0, écart-type=1)": "standard",
                          "MinMax (entre 0 et 1)": "minmax", "Aucune": None}

            if cols_to_norm and method_map[method]:
                if st.button("📐 Appliquer le scaling", type="primary", key="apply_scale"):
                    df_avant = df.copy()
                    df, scaler = normalize_columns(df, cols_to_norm,
                                                    method=method_map[method])
                    st.session_state["df_courant"] = df
                    st.session_state["scaler"] = scaler
                    st.session_state["scaled_columns"] = cols_to_norm
                    st.session_state["scaling_done"] = True

                    _afficher_avant_apres(df_avant, df, "Résultat du scaling")

                    rapport = st.session_state.get("rapport", {})
                    if rapport:
                        rapport["nettoyage"]["normalisation"] = method_map[method]
                        rapport["nettoyage"]["scaled_columns"] = cols_to_norm
                        ajouter_historique(rapport, f"Scaling {method_map[method]} sur {len(cols_to_norm)} colonnes")
                        sauvegarder_objet(rapport, scaler, "scaler.pkl")
                        sauvegarder_rapport(rapport)

                    st.success("✅ Scaling appliqué.")
                    st.rerun()

            if st.button("⏭️ Passer (pas de scaling)", key="skip_scale"):
                st.session_state["scaling_done"] = True
                st.rerun()

    # ═══════════════════════════════════════
    # 6c. Feature Engineering (optionnel)
    # ═══════════════════════════════════════
    with tab_c:
        st.subheader("Créer / modifier des colonnes")
        st.caption("Cette sous-étape est optionnelle.")

        if "modification_history" not in st.session_state:
            st.session_state["modification_history"] = []
        if "fe_operations" not in st.session_state:
            st.session_state["fe_operations"] = []

        st.dataframe(df.head(), use_container_width=True)

        fe_action = st.selectbox("Action", [
            "Combiner 2 colonnes",
            "Créer une dérivée (log, carré, racine…)",
            "Extraire depuis une date (année, mois, jour…)",
            "Supprimer une colonne",
            "Transformer en place",
            "Renommer",
            "Découper en tranches",
        ], key="fe_action",
           help="Opération d'ingénierie de features à appliquer")

        num_cols = get_numeric_columns(df)
        all_cols = df.columns.tolist()

        if fe_action == "Combiner 2 colonnes" and len(num_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                col_a = st.selectbox("Colonne A", num_cols, key="fe_a",
                                     help="Première colonne numérique")
            with c2:
                op = st.selectbox("Opération", ["sum", "diff", "product", "ratio"], key="fe_op",
                                  help="Opération mathématique entre les deux colonnes")
            with c3:
                col_b = st.selectbox("Colonne B", num_cols, key="fe_b",
                                     help="Deuxième colonne numérique")
            new_name = st.text_input("Nom", f"{col_a}_{op}_{col_b}", key="fe_name")
            if st.button("➕ Créer", type="primary", key="fe_create"):
                try:
                    df = combine_columns(df, col_a, col_b, op, new_name)
                    st.session_state["df_courant"] = df
                    st.session_state["modification_history"].append(f"➕ {new_name}")
                    st.session_state["fe_operations"].append({
                        "type": "combine", "col_a": col_a, "col_b": col_b,
                        "operation": op, "new_col": new_name})
                    st.success(f"✅ « {new_name} » créée !")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        elif fe_action.startswith("Créer une dérivée") and num_cols:
            col = st.selectbox("Colonne source", num_cols, key="fe_der_col")
            func = st.selectbox("Transformation",
                                ["square", "sqrt", "log", "inv", "abs"], key="fe_der_func")
            new_name = st.text_input("Nom", f"{col}_{func}", key="fe_der_name")
            if st.button("➕ Créer", type="primary", key="fe_der_create"):
                try:
                    min_val = float(df[col].min())
                    ops = {
                        "square": lambda: df[col] ** 2,
                        "sqrt": lambda: np.sqrt(df[col] - min_val) if min_val < 0 else np.sqrt(df[col]),
                        "log": lambda: np.log1p(df[col] - min_val) if min_val <= 0 else np.log1p(df[col]),
                        "inv": lambda: 1.0 / df[col].replace(0, np.nan),
                        "abs": lambda: df[col].abs(),
                    }
                    df[new_name] = ops[func]()
                    st.session_state["df_courant"] = df
                    st.session_state["modification_history"].append(f"➕ {new_name} ({func})")
                    st.session_state["fe_operations"].append({
                        "type": "derive", "col": col, "func": func,
                        "new_col": new_name, "min_val": min_val})
                    st.success(f"✅ « {new_name} » créée !")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        elif fe_action.startswith("Extraire depuis une date"):
            dt_cols = detect_datetime_columns(df)
            if not dt_cols:
                st.info("Aucune colonne datetime détectée.")
            else:
                col = st.selectbox("Colonne datetime", dt_cols, key="fe_dt_col",
                                   help="Colonne de dates dont extraire les composantes")
                available_features = ["year", "month", "day", "dayofweek", "hour", "quarter", "weekofyear"]
                selected_features = st.multiselect("Composantes à extraire",
                                                    available_features,
                                                    default=["year", "month", "dayofweek"],
                                                    key="fe_dt_features",
                                                    help="Éléments de date à extraire en nouvelles colonnes")
                if selected_features and st.button("📅 Extraire", type="primary", key="fe_dt_extract"):
                    try:
                        df_avant = df.copy()
                        df = extract_datetime_features(df, col, features=selected_features)
                        st.session_state["df_courant"] = df
                        new_cols = [c for c in df.columns if c not in df_avant.columns]
                        st.session_state["modification_history"].append(
                            f"📅 Datetime : {', '.join(new_cols)}")
                        st.session_state["fe_operations"].append({
                            "type": "datetime_extract", "col": col,
                            "features": selected_features})
                        _afficher_avant_apres(df_avant, df, "Extraction datetime")
                        st.success(f"✅ {len(new_cols)} colonne(s) extraite(s) !")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        elif fe_action == "Supprimer une colonne":
            col = st.selectbox("Colonne à supprimer", all_cols, key="fe_drop",
                               help="Cette colonne sera définitivement retirée du jeu de données")
            if st.button("🗑️ Supprimer", key="fe_drop_btn"):
                df = drop_column(df, col)
                st.session_state["df_courant"] = df
                st.session_state["modification_history"].append(f"🗑️ {col}")
                st.success(f"✅ « {col} » supprimée.")
                st.rerun()

        elif fe_action == "Transformer en place" and num_cols:
            col = st.selectbox("Colonne", num_cols, key="fe_trans_col",
                               help="Colonne numérique à transformer en place")
            func = st.selectbox("Fonction", ["log", "sqrt", "square"], key="fe_trans_func",
                                help="log = logarithme, sqrt = racine carrée, square = carré")
            if st.button("🔄 Transformer", key="fe_trans_btn"):
                try:
                    df = transform_column(df, col, func)
                    st.session_state["df_courant"] = df
                    st.session_state["modification_history"].append(f"🔄 {col} ({func})")
                    st.session_state["fe_operations"].append({
                        "type": "transform_inplace", "col": col, "func": func,
                        "min_val": float(df[col].min())})
                    st.success(f"✅ « {col} » transformée.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        elif fe_action == "Renommer":
            col = st.selectbox("Colonne", all_cols, key="fe_ren_col",
                               help="Colonne à renommer")
            new_n = st.text_input("Nouveau nom", col, key="fe_ren_name")
            if st.button("✏️ Renommer", key="fe_ren_btn") and new_n != col:
                df = rename_column(df, col, new_n)
                st.session_state["df_courant"] = df
                st.session_state["modification_history"].append(f"✏️ {col} → {new_n}")
                st.rerun()

        elif fe_action == "Découper en tranches" and num_cols:
            col = st.selectbox("Colonne", num_cols, key="fe_disc_col",
                               help="Colonne numérique à découper en catégories")
            n_bins = st.slider("Tranches", 2, 10, 4, key="fe_disc_n",
                               help="Nombre de catégories dans lesquelles découper")
            if st.button("✂️ Découper", key="fe_disc_btn"):
                try:
                    df = discretize_column(df, col, n_bins=n_bins)
                    st.session_state["df_courant"] = df
                    st.session_state["modification_history"].append(f"✂️ {col} en {n_bins} tranches")
                    st.session_state["fe_operations"].append({
                        "type": "discretize", "col": col, "n_bins": n_bins})
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        # Historique
        hist = st.session_state.get("modification_history", [])
        if hist:
            st.divider()
            st.markdown("**Modifications effectuées :**")
            for h in hist:
                st.write(f"  • {h}")

        # Bouton de validation finale
        st.divider()

        # Analyse et propositions
        with st.expander("💡 Analyse et recommandations", expanded=True):
            na_count = df.isna().sum().sum()
            n_cat = len(get_categorical_columns(df))
            n_num = len(get_numeric_columns(df))
            st.markdown(f"**État des données :** {len(df)} lignes × {len(df.columns)} colonnes")
            st.markdown(f"- Valeurs manquantes restantes : **{na_count}**")
            st.markdown(f"- Colonnes numériques : **{n_num}** | Catégorielles : **{n_cat}**")

            props = []
            if na_count > 0:
                props.append("- ⚠️ Il reste des valeurs manquantes — retournez au nettoyage")
            if n_cat > 0 and not st.session_state.get("encoding_done"):
                props.append("- Pensez à encoder les colonnes catégorielles avant la modélisation")
            if n_num > 0 and not st.session_state.get("scaling_done"):
                props.append("- Le scaling peut améliorer les performances (SVM, KNN, régression)")
            props.append("- Vos données sont prêtes → passez à la modélisation (étape 7)")
            st.markdown("**Propositions :**")
            for p in props:
                st.markdown(p)

        if st.button("✅ Valider les transformations", type="primary", key="validate_transform"):
            st.session_state["transformation_done"] = True
            st.session_state["prepared_df"] = df.copy()

            rapport = st.session_state.get("rapport", {})
            if rapport:
                rapport["etape_courante"] = max(rapport.get("etape_courante", 0), 7)
                ajouter_historique(rapport, "Transformations validées")
                sauvegarder_rapport(rapport)
                sauvegarder_csv(rapport, df, "data_cleaned.csv")

            # Auto-avancer vers l'étape 7 — Modélisation
            st.session_state["_pending_step"] = 7
            st.rerun()
