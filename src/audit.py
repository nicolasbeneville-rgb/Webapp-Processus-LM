# -*- coding: utf-8 -*-
"""
audit.py — Analyse statistique et EDA (Exploratory Data Analysis).

Fonctions principales :
    - quality_table          : tableau de qualité par colonne
    - detect_anomalies       : colonnes constantes, quasi-constantes, outliers
    - correlation_matrix     : matrice de corrélation et top corrélations
    - compute_quality_score  : score global de qualité sur 100
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from config import (
    CORRELATION_THRESHOLD,
    QUASI_CONSTANT_PCT,
    OUTLIER_IQR_FACTOR,
    CLASS_IMBALANCE_RATIO,
)


# ═══════════════════════════════════════════════════════════════════
# 4.1 — TABLEAU DE QUALITÉ
# ═══════════════════════════════════════════════════════════════════
def quality_table(df: pd.DataFrame, max_missing_pct: float = 20.0) -> pd.DataFrame:
    """Génère un tableau de qualité pour chaque colonne du DataFrame.

    Args:
        df: DataFrame à analyser.
        max_missing_pct: Seuil (%) pour colorer les colonnes problématiques.

    Returns:
        DataFrame avec les statistiques par colonne.
    """
    records = []
    for col in df.columns:
        series = df[col]
        n_missing = series.isna().sum()
        pct_missing = round(n_missing / len(df) * 100, 1)
        n_unique = series.nunique()

        row = {
            "Colonne": col,
            "Type": str(series.dtype),
            "Valeurs manquantes": n_missing,
            "% manquant": pct_missing,
            "Valeurs uniques": n_unique,
        }

        if pd.api.types.is_numeric_dtype(series):
            row["Min"] = series.min()
            row["Max"] = series.max()
            row["Moyenne"] = round(series.mean(), 2) if not series.isna().all() else None
            row["Médiane"] = round(series.median(), 2) if not series.isna().all() else None
        else:
            row["Min"] = None
            row["Max"] = None
            row["Moyenne"] = None
            row["Médiane"] = None

        # Indicateur de sévérité
        if pct_missing > max_missing_pct:
            row["Sévérité"] = "🔴"
        elif pct_missing > 5:
            row["Sévérité"] = "🟠"
        else:
            row["Sévérité"] = "🟢"

        records.append(row)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════
# 4.2 — STATISTIQUES DESCRIPTIVES
# ═══════════════════════════════════════════════════════════════════
def descriptive_stats(df: pd.DataFrame) -> tuple:
    """Sépare les statistiques descriptives numériques et catégorielles.

    Args:
        df: DataFrame source.

    Returns:
        Tuple (stats_numériques: DataFrame, stats_catégorielles: DataFrame).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_stats = df[num_cols].describe().T if num_cols else pd.DataFrame()
    cat_stats = df[cat_cols].describe().T if cat_cols else pd.DataFrame()

    return num_stats, cat_stats


# ═══════════════════════════════════════════════════════════════════
# 4.3 — DÉTECTION DES ANOMALIES
# ═══════════════════════════════════════════════════════════════════
def detect_anomalies(df: pd.DataFrame) -> dict:
    """Détecte les anomalies courantes dans le DataFrame.

    Args:
        df: DataFrame à analyser.

    Returns:
        Dictionnaire avec les anomalies détectées par catégorie.
    """
    anomalies = {
        "constant": [],
        "quasi_constant": [],
        "high_cardinality": [],
        "outliers": {},
    }

    for col in df.columns:
        series = df[col].dropna()
        n_unique = series.nunique()

        # Colonnes constantes
        if n_unique <= 1:
            anomalies["constant"].append(col)
            continue

        # Colonnes quasi-constantes
        if n_unique > 1:
            most_common_pct = series.value_counts(normalize=True).iloc[0] * 100
            if most_common_pct >= QUASI_CONSTANT_PCT:
                anomalies["quasi_constant"].append(
                    {"col": col, "pct": round(most_common_pct, 1)}
                )

        # Colonnes à trop de valeurs uniques (potentiels identifiants)
        if df[col].dtype == "object":
            ratio = n_unique / max(len(df), 1)
            if ratio > 0.9 and n_unique > 50:
                anomalies["high_cardinality"].append(
                    {"col": col, "n_unique": n_unique}
                )

        # Outliers (IQR) pour colonnes numériques
        if pd.api.types.is_numeric_dtype(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - OUTLIER_IQR_FACTOR * iqr
            upper = q3 + OUTLIER_IQR_FACTOR * iqr
            n_outliers = ((series < lower) | (series > upper)).sum()
            if n_outliers > 0:
                anomalies["outliers"][col] = {
                    "count": int(n_outliers),
                    "pct": round(n_outliers / len(series) * 100, 1),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                }

    return anomalies


def get_anomaly_actions(anomalies: dict) -> list:
    """Propose des actions pour chaque anomalie détectée.

    Args:
        anomalies: Résultat de detect_anomalies().

    Returns:
        Liste de recommandations textuelles.
    """
    actions = []

    for col in anomalies["constant"]:
        actions.append(
            f"🔴 « {col} » est constante (1 seule valeur). "
            "→ Supprimez-la, elle n'apporte aucune information."
        )

    for item in anomalies["quasi_constant"]:
        actions.append(
            f"🟠 « {item['col']} » est quasi-constante ({item['pct']}% même valeur). "
            "→ Envisagez de la supprimer."
        )

    for item in anomalies["high_cardinality"]:
        actions.append(
            f"🟠 « {item['col']} » a {item['n_unique']} valeurs uniques "
            "(possible identifiant). → Supprimez ou regroupez."
        )

    for col, info in anomalies["outliers"].items():
        actions.append(
            f"🟡 « {col} » : {info['count']} outliers ({info['pct']}%) "
            f"en dehors de [{info['lower_bound']}, {info['upper_bound']}]. "
            "→ Plafonner ou supprimer."
        )

    return actions


# ═══════════════════════════════════════════════════════════════════
# 4.4 — MATRICE DE CORRÉLATION
# ═══════════════════════════════════════════════════════════════════
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la matrice de corrélation des variables numériques.

    Args:
        df: DataFrame source.

    Returns:
        Matrice de corrélation (DataFrame).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return pd.DataFrame()
    return df[num_cols].corr()


def high_correlations(corr_matrix: pd.DataFrame,
                      threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    """Identifie les paires de variables fortement corrélées.

    Args:
        corr_matrix: Matrice de corrélation.
        threshold: Seuil de corrélation (valeur absolue).

    Returns:
        DataFrame avec les paires (var1, var2, corrélation).
    """
    if corr_matrix.empty:
        return pd.DataFrame(columns=["Variable 1", "Variable 2", "Corrélation"])

    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "Variable 1": cols[i],
                    "Variable 2": cols[j],
                    "Corrélation": round(val, 3),
                })

    result = pd.DataFrame(pairs)
    if not result.empty:
        result = result.sort_values("Corrélation", key=abs, ascending=False)
    return result


def top_correlations_with_target(df: pd.DataFrame, target: str,
                                 n: int = 10) -> pd.DataFrame:
    """Top N variables les plus corrélées avec la cible.

    Args:
        df: DataFrame source.
        target: Nom de la variable cible.
        n: Nombre de variables à retourner.

    Returns:
        DataFrame trié par corrélation absolue.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in num_cols or len(num_cols) < 2:
        return pd.DataFrame()

    corrs = df[num_cols].corr()[target].drop(target, errors="ignore")
    top = corrs.abs().sort_values(ascending=False).head(n)
    result = pd.DataFrame({
        "Variable": top.index,
        "Corrélation": [round(corrs[v], 3) for v in top.index],
        "Corrélation abs.": [round(abs(corrs[v]), 3) for v in top.index],
    })
    return result


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Génère une heatmap de corrélation avec seaborn.

    Args:
        corr_matrix: Matrice de corrélation.

    Returns:
        Figure matplotlib.
    """
    if corr_matrix.empty:
        return None

    n = len(corr_matrix.columns)
    size = max(3.5, n * 0.4)
    fig, ax = plt.subplots(figsize=(size, size))
    fig.set_facecolor("#F8F9FC")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=n <= 15,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=2,
        linecolor="#F8F9FC",
        ax=ax,
        annot_kws={"fontsize": 8, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8, "aspect": 25},
    )
    ax.set_title("Matrice de correlation", fontsize=10, fontweight="bold",
                 color="#111827", pad=10)
    ax.tick_params(labelsize=7, colors="#6B7280")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# 4.5 — SCORE DE QUALITÉ
# ═══════════════════════════════════════════════════════════════════
def compute_quality_score(df: pd.DataFrame, anomalies: dict = None) -> dict:
    """Calcule un score de qualité global sur 100.

    Critères :
        - Complétude (% de valeurs non manquantes)   : 40 points
        - Absence de colonnes constantes              : 15 points
        - Absence de colonnes quasi-constantes        : 10 points
        - Diversité des types                         : 10 points
        - Absence d'outliers extrêmes                 : 15 points
        - Absence de doublons                         : 10 points

    Args:
        df: DataFrame à évaluer.
        anomalies: Résultat de detect_anomalies() (optionnel, sera calculé sinon).

    Returns:
        Dict avec score, détails et points d'attention.
    """
    if anomalies is None:
        anomalies = detect_anomalies(df)

    total = 0
    details = []

    # Complétude (40 pts)
    completeness = 1 - df.isna().sum().sum() / max(df.size, 1)
    score_completeness = round(completeness * 40, 1)
    total += score_completeness
    details.append(f"Complétude : {score_completeness}/40 ({completeness:.1%} de données présentes)")

    # Colonnes constantes (15 pts)
    n_const = len(anomalies["constant"])
    score_const = max(0, 15 - n_const * 5)
    total += score_const
    details.append(f"Colonnes constantes : {score_const}/15 ({n_const} détectée(s))")

    # Quasi-constantes (10 pts)
    n_quasi = len(anomalies["quasi_constant"])
    score_quasi = max(0, 10 - n_quasi * 3)
    total += score_quasi
    details.append(f"Colonnes quasi-constantes : {score_quasi}/10 ({n_quasi} détectée(s))")

    # Diversité des types (10 pts)
    n_types = df.dtypes.nunique()
    score_types = min(10, n_types * 3)
    total += score_types
    details.append(f"Diversité des types : {score_types}/10 ({n_types} type(s) différent(s))")

    # Outliers (15 pts)
    n_outlier_cols = len(anomalies["outliers"])
    total_outlier_pct = sum(v["pct"] for v in anomalies["outliers"].values()) if anomalies["outliers"] else 0
    score_outliers = max(0, 15 - int(total_outlier_pct))
    total += score_outliers
    details.append(f"Outliers : {score_outliers}/15 ({n_outlier_cols} colonne(s) concernée(s))")

    # Doublons (10 pts)
    n_dups = df.duplicated().sum()
    dup_pct = n_dups / max(len(df), 1)
    score_dups = max(0, round(10 * (1 - dup_pct)))
    total += score_dups
    details.append(f"Doublons : {score_dups}/10 ({n_dups} ligne(s) dupliquée(s))")

    total = min(100, round(total))

    # Points d'attention classés par priorité
    attention = []
    if score_completeness < 30:
        attention.append("🔴 Beaucoup de valeurs manquantes — nettoyage prioritaire.")
    if n_const > 0:
        attention.append(f"🔴 {n_const} colonne(s) constante(s) à supprimer.")
    if n_outlier_cols > 3:
        attention.append("🟠 Nombreuses colonnes avec outliers — vérifiez les données.")
    if n_dups > 0:
        attention.append(f"🟠 {n_dups} ligne(s) dupliquée(s) détectée(s).")
    if n_quasi > 0:
        attention.append(f"🟡 {n_quasi} colonne(s) quasi-constante(s).")

    return {
        "score": total,
        "details": details,
        "attention": attention,
    }


def check_target_imbalance(series: pd.Series,
                           max_ratio: float = CLASS_IMBALANCE_RATIO) -> bool:
    """Vérifie si la variable cible est déséquilibrée (classification).

    Args:
        series: Variable cible.
        max_ratio: Ratio max entre la classe majoritaire et minoritaire.

    Returns:
        True si déséquilibrée.
    """
    counts = series.value_counts()
    if len(counts) < 2:
        return True
    ratio = counts.iloc[0] / max(counts.iloc[-1], 1)
    return ratio > max_ratio
