# -*- coding: utf-8 -*-
"""
consolidation.py — Jointures et agrégations multi-bases.

Fonctions principales :
    - preview_join    : estime le résultat d'une jointure avant exécution
    - perform_join    : exécute une jointure entre deux DataFrames
    - aggregate       : agrège un DataFrame avant jointure
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# PRÉVISUALISATION
# ═══════════════════════════════════════════════════════════════════
def preview_join(df_left: pd.DataFrame, df_right: pd.DataFrame,
                 key_left: str, key_right: str,
                 how: str = "inner") -> dict:
    """Estime le résultat d'une jointure sans l'exécuter.

    Args:
        df_left: DataFrame de gauche.
        df_right: DataFrame de droite.
        key_left: Colonne clé dans df_left.
        key_right: Colonne clé dans df_right.
        how: Type de jointure (inner, left, right, outer).

    Returns:
        Dictionnaire avec estimations et avertissements.
    """
    warnings = []

    # Doublons dans les clés
    dup_left = df_left[key_left].duplicated().sum()
    dup_right = df_right[key_right].duplicated().sum()
    if dup_left > 0:
        warnings.append(
            f"⚠️ La clé « {key_left} » contient {dup_left} doublon(s) "
            f"dans la table de gauche."
        )
    if dup_right > 0:
        warnings.append(
            f"⚠️ La clé « {key_right} » contient {dup_right} doublon(s) "
            f"dans la table de droite."
        )

    # Estimation du nombre de lignes
    set_left = set(df_left[key_left].dropna().unique())
    set_right = set(df_right[key_right].dropna().unique())
    common = set_left & set_right

    if how == "inner":
        estimated_rows = len(common) if (dup_left == 0 and dup_right == 0) else "variable"
    elif how == "left":
        estimated_rows = len(df_left)
    elif how == "right":
        estimated_rows = len(df_right)
    else:  # outer
        estimated_rows = len(set_left | set_right) if (dup_left == 0 and dup_right == 0) else "variable"

    if how == "inner":
        only_left = len(set_left - set_right)
        only_right = len(set_right - set_left)
        if only_left > 0:
            warnings.append(
                f"ℹ️ {only_left} valeur(s) de clé présente(s) uniquement à gauche "
                f"seront exclues (INNER JOIN)."
            )
        if only_right > 0:
            warnings.append(
                f"ℹ️ {only_right} valeur(s) de clé présente(s) uniquement à droite "
                f"seront exclues (INNER JOIN)."
            )

    # Colonnes en commun (hors clé de jointure)
    overlap = set(df_left.columns) & set(df_right.columns)
    overlap -= {key_left, key_right}
    if overlap:
        warnings.append(
            f"⚠️ Colonnes en commun (seront suffixées _x / _y) : "
            f"{', '.join(sorted(overlap))}"
        )

    return {
        "estimated_rows": estimated_rows,
        "left_rows": len(df_left),
        "right_rows": len(df_right),
        "common_keys": len(common),
        "total_cols": len(df_left.columns) + len(df_right.columns) - 1,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════
# JOINTURE
# ═══════════════════════════════════════════════════════════════════
def perform_join(df_left: pd.DataFrame, df_right: pd.DataFrame,
                 key_left: str, key_right: str,
                 how: str = "inner") -> pd.DataFrame:
    """Effectue une jointure entre deux DataFrames.

    Args:
        df_left: DataFrame de gauche.
        df_right: DataFrame de droite.
        key_left: Colonne clé dans df_left.
        key_right: Colonne clé dans df_right.
        how: Type de jointure (inner, left, right, outer).

    Returns:
        DataFrame résultat de la jointure.
    """
    result = pd.merge(
        df_left,
        df_right,
        left_on=key_left,
        right_on=key_right,
        how=how,
        suffixes=("_x", "_y"),
    )
    return result


def get_join_stats(df_left: pd.DataFrame, df_result: pd.DataFrame) -> dict:
    """Calcule les statistiques après jointure.

    Args:
        df_left: DataFrame de gauche (référence).
        df_result: DataFrame résultat de la jointure.

    Returns:
        Dictionnaire avec les statistiques.
    """
    rows_before = len(df_left)
    rows_after = len(df_result)
    rows_lost = max(0, rows_before - rows_after)
    loss_pct = round(rows_lost / max(rows_before, 1) * 100, 1)
    new_nans = df_result.isna().sum().sum() - 0  # total NaN dans résultat

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_lost": rows_lost,
        "loss_pct": loss_pct,
        "new_nans": int(new_nans),
        "nb_cols": len(df_result.columns),
    }


# ═══════════════════════════════════════════════════════════════════
# AGRÉGATION
# ═══════════════════════════════════════════════════════════════════
def aggregate(df: pd.DataFrame, group_col: str,
              agg_cols: list, agg_func: str) -> pd.DataFrame:
    """Agrège un DataFrame selon une colonne de regroupement.

    Args:
        df: DataFrame source.
        group_col: Colonne de regroupement.
        agg_cols: Colonnes à agréger.
        agg_func: Fonction d'agrégation (sum, mean, median, count, min, max).

    Returns:
        DataFrame agrégé.
    """
    agg_dict = {col: agg_func for col in agg_cols}
    result = df.groupby(group_col, as_index=False).agg(agg_dict)
    return result
