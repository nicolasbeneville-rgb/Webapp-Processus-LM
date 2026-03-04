# -*- coding: utf-8 -*-
"""
preprocessing.py — Nettoyage, normalisation, encodage des données.

Fonctions principales :
    - handle_missing        : traitement des valeurs manquantes
    - handle_outliers       : traitement des outliers
    - normalize_columns     : normalisation / standardisation
    - encode_categorical    : encodage des variables catégorielles
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from config import OUTLIER_IQR_FACTOR, CAPPING_LOWER_PERCENTILE, CAPPING_UPPER_PERCENTILE


# ═══════════════════════════════════════════════════════════════════
# 5.1 — TRAITEMENT DES VALEURS MANQUANTES
# ═══════════════════════════════════════════════════════════════════
def handle_missing(df: pd.DataFrame, strategies: dict,
                   fixed_values: dict = None) -> pd.DataFrame:
    """Applique les stratégies de traitement des valeurs manquantes.

    Args:
        df: DataFrame source.
        strategies: Dict {col: stratégie}. Stratégies possibles :
            - "drop_column" : supprimer la colonne
            - "drop_rows"   : supprimer les lignes avec NaN
            - "mean"        : imputer par la moyenne
            - "median"      : imputer par la médiane
            - "mode"        : imputer par le mode
            - "fixed"       : imputer par une valeur fixe
            - "indicator"   : créer une colonne indicatrice
        fixed_values: Dict {col: valeur} pour la stratégie "fixed".

    Returns:
        Nouveau DataFrame nettoyé.
    """
    result = df.copy()
    fixed_values = fixed_values or {}

    for col, strategy in strategies.items():
        if col not in result.columns:
            continue
        if result[col].isna().sum() == 0:
            continue

        if strategy == "drop_column":
            result = result.drop(columns=[col])

        elif strategy == "drop_rows":
            result = result.dropna(subset=[col])

        elif strategy == "mean":
            result[col] = result[col].fillna(result[col].mean())

        elif strategy == "median":
            result[col] = result[col].fillna(result[col].median())

        elif strategy == "mode":
            mode_val = result[col].mode()
            if len(mode_val) > 0:
                result[col] = result[col].fillna(mode_val[0])

        elif strategy == "fixed":
            val = fixed_values.get(col, 0)
            result[col] = result[col].fillna(val)

        elif strategy == "indicator":
            result[f"{col}_missing"] = result[col].isna().astype(int)
            # Imputer par la médiane en plus
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
            else:
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val[0])

    result = result.reset_index(drop=True)
    return result


# ═══════════════════════════════════════════════════════════════════
# 5.2 — TRAITEMENT DES OUTLIERS
# ═══════════════════════════════════════════════════════════════════
def detect_outliers_iqr(series: pd.Series,
                        factor: float = OUTLIER_IQR_FACTOR) -> dict:
    """Détecte les outliers d'une série numérique par la méthode IQR.

    Args:
        series: Série numérique.
        factor: Facteur multiplicatif de l'IQR (défaut 1.5).

    Returns:
        Dict avec count, indices, bounds.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (series < lower) | (series > upper)
    return {
        "count": int(mask.sum()),
        "indices": series[mask].index.tolist(),
        "lower_bound": lower,
        "upper_bound": upper,
    }


def handle_outliers(df: pd.DataFrame, strategies: dict) -> pd.DataFrame:
    """Applique les stratégies de traitement des outliers.

    Args:
        df: DataFrame source.
        strategies: Dict {col: stratégie}. Stratégies possibles :
            - "keep"    : ne rien faire
            - "drop"    : supprimer les lignes
            - "cap"     : plafonner (capping)
            - "log"     : transformation logarithmique

    Returns:
        Nouveau DataFrame.
    """
    result = df.copy()

    for col, strategy in strategies.items():
        if col not in result.columns:
            continue
        if not pd.api.types.is_numeric_dtype(result[col]):
            continue

        if strategy == "keep":
            continue

        elif strategy == "drop":
            info = detect_outliers_iqr(result[col])
            result = result.drop(index=info["indices"], errors="ignore")

        elif strategy == "cap":
            lower = result[col].quantile(CAPPING_LOWER_PERCENTILE / 100)
            upper = result[col].quantile(CAPPING_UPPER_PERCENTILE / 100)
            result[col] = result[col].clip(lower=lower, upper=upper)

        elif strategy == "log":
            min_val = result[col].min()
            if min_val <= 0:
                result[col] = np.log1p(result[col] - min_val)
            else:
                result[col] = np.log1p(result[col])

    result = result.reset_index(drop=True)
    return result


# ═══════════════════════════════════════════════════════════════════
# 5.3 — NORMALISATION
# ═══════════════════════════════════════════════════════════════════
def normalize_columns(df: pd.DataFrame, columns: list,
                      method: str = "standard") -> tuple:
    """Normalise les colonnes sélectionnées.

    Args:
        df: DataFrame source.
        columns: Liste de colonnes à normaliser.
        method: "standard" (StandardScaler) ou "minmax" (MinMaxScaler).

    Returns:
        Tuple (DataFrame normalisé, scaler utilisé).
    """
    result = df.copy()

    if not columns:
        return result, None

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return result, None

    result[columns] = scaler.fit_transform(result[columns])
    return result, scaler


# ═══════════════════════════════════════════════════════════════════
# 5.4 — ENCODAGE DES VARIABLES CATÉGORIELLES
# ═══════════════════════════════════════════════════════════════════
def encode_categorical(df: pd.DataFrame, strategies: dict,
                       target_col: str = None) -> tuple:
    """Encode les variables catégorielles.

    Args:
        df: DataFrame source.
        strategies: Dict {col: stratégie}. Stratégies :
            - "onehot"   : One-Hot Encoding
            - "label"    : Label Encoding
            - "target"   : Target Encoding (requiert target_col)
            - "drop"     : supprimer la colonne
        target_col: Variable cible (pour Target Encoding).

    Returns:
        Tuple (DataFrame encodé, dict des encoders utilisés).
    """
    result = df.copy()
    encoders = {}

    for col, strategy in strategies.items():
        if col not in result.columns:
            continue

        if strategy == "onehot":
            dummies = pd.get_dummies(result[col], prefix=col, dtype=int)
            result = pd.concat([result.drop(columns=[col]), dummies], axis=1)
            encoders[col] = {"type": "onehot", "categories": dummies.columns.tolist()}

        elif strategy == "label":
            le = LabelEncoder()
            mask = result[col].notna()
            result.loc[mask, col] = le.fit_transform(result.loc[mask, col].astype(str))
            result[col] = pd.to_numeric(result[col], errors="coerce")
            encoders[col] = {"type": "label", "encoder": le}

        elif strategy == "target":
            if target_col and target_col in result.columns:
                means = result.groupby(col)[target_col].mean()
                result[col] = result[col].map(means)
                encoders[col] = {"type": "target", "mapping": means.to_dict()}

        elif strategy == "drop":
            result = result.drop(columns=[col])

    return result, encoders


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Retourne la liste des colonnes catégorielles / textuelles.

    Args:
        df: DataFrame source.

    Returns:
        Liste de noms de colonnes.
    """
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Retourne la liste des colonnes numériques.

    Args:
        df: DataFrame source.

    Returns:
        Liste de noms de colonnes.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()
