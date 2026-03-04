# -*- coding: utf-8 -*-
"""
feature_engineering.py — Création et sélection de variables.

Fonctions principales :
    - combine_columns      : crée une variable par combinaison
    - transform_column     : applique une transformation mathématique
    - discretize_column    : discrétise une variable numérique
    - auto_select_features : sélection automatique des N meilleures
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# COMBINAISON DE COLONNES
# ═══════════════════════════════════════════════════════════════════
def combine_columns(df: pd.DataFrame, col_a: str, col_b: str,
                    operation: str, new_name: str = None) -> pd.DataFrame:
    """Crée une nouvelle colonne par combinaison de deux colonnes numériques.

    Args:
        df: DataFrame source.
        col_a: Première colonne.
        col_b: Deuxième colonne.
        operation: Opération à appliquer ("sum", "diff", "ratio", "product").
        new_name: Nom de la nouvelle colonne (auto-généré si None).

    Returns:
        DataFrame avec la nouvelle colonne ajoutée.
    """
    result = df.copy()

    if new_name is None:
        new_name = f"{col_a}_{operation}_{col_b}"

    if operation == "sum":
        result[new_name] = result[col_a] + result[col_b]
    elif operation == "diff":
        result[new_name] = result[col_a] - result[col_b]
    elif operation == "ratio":
        result[new_name] = result[col_a] / result[col_b].replace(0, np.nan)
    elif operation == "product":
        result[new_name] = result[col_a] * result[col_b]
    else:
        raise ValueError(f"Opération inconnue : {operation}")

    return result


# ═══════════════════════════════════════════════════════════════════
# TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════
def transform_column(df: pd.DataFrame, col: str,
                     transformation: str) -> pd.DataFrame:
    """Applique une transformation mathématique à une colonne.

    Args:
        df: DataFrame source.
        col: Colonne à transformer.
        transformation: Type de transformation ("log", "sqrt", "square").

    Returns:
        DataFrame avec la colonne transformée.
    """
    result = df.copy()

    if transformation == "log":
        min_val = result[col].min()
        if min_val <= 0:
            result[col] = np.log1p(result[col] - min_val)
        else:
            result[col] = np.log1p(result[col])

    elif transformation == "sqrt":
        min_val = result[col].min()
        if min_val < 0:
            result[col] = np.sqrt(result[col] - min_val)
        else:
            result[col] = np.sqrt(result[col])

    elif transformation == "square":
        result[col] = result[col] ** 2

    else:
        raise ValueError(f"Transformation inconnue : {transformation}")

    return result


# ═══════════════════════════════════════════════════════════════════
# DISCRÉTISATION
# ═══════════════════════════════════════════════════════════════════
def discretize_column(df: pd.DataFrame, col: str,
                      n_bins: int = 5, strategy: str = "quantile",
                      labels: list = None) -> pd.DataFrame:
    """Discrétise une variable numérique en tranches.

    Args:
        df: DataFrame source.
        col: Colonne à discrétiser.
        n_bins: Nombre de tranches.
        strategy: "quantile" ou "uniform".
        labels: Labels personnalisés (optionnel).

    Returns:
        DataFrame avec la colonne discrétisée.
    """
    result = df.copy()
    new_col = f"{col}_binned"

    if strategy == "quantile":
        result[new_col] = pd.qcut(result[col], q=n_bins,
                                  labels=labels, duplicates="drop")
    elif strategy == "uniform":
        result[new_col] = pd.cut(result[col], bins=n_bins,
                                 labels=labels)
    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")

    return result


# ═══════════════════════════════════════════════════════════════════
# RENOMMAGE
# ═══════════════════════════════════════════════════════════════════
def rename_column(df: pd.DataFrame, old_name: str,
                  new_name: str) -> pd.DataFrame:
    """Renomme une colonne du DataFrame.

    Args:
        df: DataFrame source.
        old_name: Nom actuel de la colonne.
        new_name: Nouveau nom.

    Returns:
        DataFrame avec la colonne renommée.
    """
    return df.rename(columns={old_name: new_name})


# ═══════════════════════════════════════════════════════════════════
# SUPPRESSION
# ═══════════════════════════════════════════════════════════════════
def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Supprime une colonne du DataFrame.

    Args:
        df: DataFrame source.
        col: Nom de la colonne à supprimer.

    Returns:
        DataFrame sans la colonne.
    """
    return df.drop(columns=[col], errors="ignore")


# ═══════════════════════════════════════════════════════════════════
# SÉLECTION AUTOMATIQUE DES FEATURES
# ═══════════════════════════════════════════════════════════════════
def auto_select_features(df: pd.DataFrame, target_col: str,
                         n: int = 10) -> list:
    """Sélectionne automatiquement les N variables les plus corrélées à la cible.

    Args:
        df: DataFrame source.
        target_col: Variable cible.
        n: Nombre de variables à sélectionner.

    Returns:
        Liste des noms de colonnes sélectionnées.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in num_cols:
        return [c for c in num_cols if c != target_col][:n]

    corrs = df[num_cols].corr()[target_col].drop(target_col, errors="ignore")
    top = corrs.abs().sort_values(ascending=False).head(n)
    return top.index.tolist()


def get_modification_summary(df_before: pd.DataFrame,
                             df_after: pd.DataFrame) -> dict:
    """Résume les modifications entre deux versions du DataFrame.

    Args:
        df_before: DataFrame avant modification.
        df_after: DataFrame après modification.

    Returns:
        Dict avec les changements détectés.
    """
    added_cols = set(df_after.columns) - set(df_before.columns)
    removed_cols = set(df_before.columns) - set(df_after.columns)

    return {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "cols_before": len(df_before.columns),
        "cols_after": len(df_after.columns),
        "added_columns": sorted(added_cols),
        "removed_columns": sorted(removed_cols),
    }
