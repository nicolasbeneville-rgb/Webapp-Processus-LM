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


# ═══════════════════════════════════════════════════════════════════
# DÉTECTION ET FEATURE ENGINEERING TEMPOREL
# ═══════════════════════════════════════════════════════════════════
def detect_datetime_columns(df: pd.DataFrame) -> list:
    """Détecte les colonnes qui contiennent des dates/heures."""
    dt_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:
                dt_cols.append(col)
        except Exception:
            continue
    return dt_cols


def extract_datetime_features(df: pd.DataFrame, col: str,
                               features: list = None) -> pd.DataFrame:
    """Extrait des composantes temporelles d'une colonne datetime.

    Args:
        df: DataFrame source.
        col: Colonne datetime.
        features: Liste de composantes à extraire parmi :
            "year", "month", "day", "weekday", "hour", "quarter",
            "is_weekend", "day_of_year", "week_of_year"

    Returns:
        DataFrame avec les nouvelles colonnes ajoutées.
    """
    result = df.copy()
    dt = pd.to_datetime(result[col], errors="coerce")

    all_features = features or ["year", "month", "day", "weekday", "quarter", "is_weekend"]
    prefix = col

    extractors = {
        "year": lambda s: s.dt.year,
        "month": lambda s: s.dt.month,
        "day": lambda s: s.dt.day,
        "weekday": lambda s: s.dt.weekday,
        "hour": lambda s: s.dt.hour,
        "quarter": lambda s: s.dt.quarter,
        "is_weekend": lambda s: (s.dt.weekday >= 5).astype(int),
        "day_of_year": lambda s: s.dt.dayofyear,
        "week_of_year": lambda s: s.dt.isocalendar().week.astype(int),
    }

    created = []
    for feat in all_features:
        if feat in extractors:
            new_col = f"{prefix}_{feat}"
            result[new_col] = extractors[feat](dt)
            created.append(new_col)

    return result, created


def create_lag_features(df: pd.DataFrame, col: str,
                        lags: list = None,
                        datetime_col: str = None) -> pd.DataFrame:
    """Crée des variables de décalage (lag) pour une colonne numérique.

    Args:
        df: DataFrame source (doit être trié chronologiquement).
        col: Colonne numérique pour laquelle créer les lags.
        lags: Liste des décalages (ex: [1, 2, 3] pour t-1, t-2, t-3).
        datetime_col: Colonne datetime pour trier (optionnel).

    Returns:
        Tuple (DataFrame avec lags, liste des colonnes créées).
    """
    result = df.copy()
    if datetime_col and datetime_col in result.columns:
        result = result.sort_values(datetime_col).reset_index(drop=True)

    lags = lags or [1, 2, 3]
    created = []
    for lag in lags:
        new_col = f"{col}_lag{lag}"
        result[new_col] = result[col].shift(lag)
        created.append(new_col)

    return result, created


def create_rolling_features(df: pd.DataFrame, col: str,
                             windows: list = None,
                             datetime_col: str = None) -> pd.DataFrame:
    """Crée des moyennes glissantes pour une colonne numérique.

    Args:
        df: DataFrame source (doit être trié chronologiquement).
        col: Colonne numérique.
        windows: Tailles des fenêtres (ex: [3, 7, 14]).
        datetime_col: Colonne datetime pour trier (optionnel).

    Returns:
        Tuple (DataFrame avec rolling features, liste des colonnes créées).
    """
    result = df.copy()
    if datetime_col and datetime_col in result.columns:
        result = result.sort_values(datetime_col).reset_index(drop=True)

    windows = windows or [3, 7]
    created = []
    for w in windows:
        new_mean = f"{col}_rmean{w}"
        new_std = f"{col}_rstd{w}"
        result[new_mean] = result[col].shift(1).rolling(window=w, min_periods=1).mean()
        result[new_std] = result[col].shift(1).rolling(window=w, min_periods=1).std()
        created.extend([new_mean, new_std])

    return result, created


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


# ═══════════════════════════════════════════════════════════════════
# PRÉDICTION HORIZON (cible décalée + features lead/lag)
# ═══════════════════════════════════════════════════════════════════
def create_horizon_target(df: pd.DataFrame, target_col: str,
                           horizon: int,
                           datetime_col: str = None) -> pd.DataFrame:
    """Crée la colonne cible décalée pour prédire à +horizon périodes.

    Args:
        df: DataFrame trié chronologiquement.
        target_col: Colonne à prédire.
        horizon: Nombre de périodes dans le futur.
        datetime_col: Colonne datetime pour tri (optionnel).

    Returns:
        Tuple (DataFrame avec colonne cible horizon, nom de la colonne créée).
    """
    result = df.copy()
    if datetime_col and datetime_col in result.columns:
        result = result.sort_values(datetime_col).reset_index(drop=True)

    new_col = f"{target_col}_t+{horizon}"
    result[new_col] = result[target_col].shift(-horizon)
    return result, new_col


def create_lead_features(df: pd.DataFrame, col: str,
                          horizon: int,
                          agg: str = "sum",
                          datetime_col: str = None) -> pd.DataFrame:
    """Crée des features « futur » (lead) : agrégation sur les N prochaines périodes.

    Utile pour injecter une prévision exogène (ex: pluvio prévue à 15 jours).

    Args:
        df: DataFrame trié chronologiquement.
        col: Colonne source.
        horizon: Nombre de périodes à regarder vers l'avant.
        agg: Type d'agrégation — "sum", "mean", "max", "min".
        datetime_col: Colonne datetime pour tri (optionnel).

    Returns:
        Tuple (DataFrame avec feature lead, nom de la colonne créée).
    """
    result = df.copy()
    if datetime_col and datetime_col in result.columns:
        result = result.sort_values(datetime_col).reset_index(drop=True)

    new_col = f"{col}_lead{horizon}_{agg}"
    # Rolling inversé : on inverse, on calcule le rolling, on ré-inverse
    reversed_col = result[col].iloc[::-1]
    if agg == "sum":
        rolled = reversed_col.rolling(window=horizon, min_periods=1).sum()
    elif agg == "mean":
        rolled = reversed_col.rolling(window=horizon, min_periods=1).mean()
    elif agg == "max":
        rolled = reversed_col.rolling(window=horizon, min_periods=1).max()
    elif agg == "min":
        rolled = reversed_col.rolling(window=horizon, min_periods=1).min()
    else:
        rolled = reversed_col.rolling(window=horizon, min_periods=1).sum()

    result[new_col] = rolled.iloc[::-1].values
    return result, new_col
