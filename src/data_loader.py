# -*- coding: utf-8 -*-
"""
data_loader.py — Chargement, typage et conversion des données.

Fonctions principales :
    - load_file       : charge un fichier CSV ou Excel en DataFrame
    - detect_types    : propose un type cible pour chaque colonne
    - apply_typing    : convertit les colonnes selon les choix utilisateur
"""

import re
import pandas as pd
import numpy as np
from config import BOOLEAN_PATTERNS, TARGET_TYPES


# ═══════════════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════════════
def load_file(uploaded_file, separator: str = ",",
              encoding: str = "utf-8", header_row: int = 0) -> pd.DataFrame:
    """Charge un fichier uploadé (CSV ou Excel) en DataFrame pandas.

    Args:
        uploaded_file: Objet UploadedFile de Streamlit.
        separator: Séparateur pour les CSV.
        encoding: Encodage du fichier.
        header_row: Numéro de la ligne d'en-tête (0-indexed).

    Returns:
        pd.DataFrame chargé.

    Raises:
        ValueError: Si le format n'est pas supporté.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(
            uploaded_file,
            sep=separator,
            encoding=encoding,
            header=header_row,
        )
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(
            uploaded_file,
            header=header_row,
            engine="openpyxl" if filename.endswith(".xlsx") else None,
        )
    else:
        raise ValueError(
            f"Format non supporté : {uploaded_file.name}. "
            "Utilisez un fichier CSV ou Excel (.xlsx / .xls)."
        )

    return df


def get_file_info(df: pd.DataFrame) -> dict:
    """Retourne des informations descriptives sur un DataFrame.

    Args:
        df: DataFrame à analyser.

    Returns:
        Dictionnaire avec nb_rows, nb_cols, memory_mb, dtypes.
    """
    return {
        "nb_rows": len(df),
        "nb_cols": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════
# DÉTECTION AUTOMATIQUE DES TYPES
# ═══════════════════════════════════════════════════════════════════
def _is_boolean_column(series: pd.Series) -> bool:
    """Détecte si une colonne textuelle contient des valeurs booléennes."""
    if series.dtype == "object":
        unique_vals = series.dropna().astype(str).str.strip().str.lower().unique()
        if len(unique_vals) <= 3:
            return all(v in BOOLEAN_PATTERNS for v in unique_vals)
    return False


def _is_numeric_text(series: pd.Series) -> bool:
    """Détecte si une colonne texte contient des nombres avec symboles (€, %, etc.)."""
    if series.dtype != "object":
        return False
    sample = series.dropna().head(50).astype(str)
    pattern = re.compile(r'^[\s€$%,.\d\-+]+$')
    matches = sample.apply(lambda x: bool(pattern.match(x.strip())))
    return matches.mean() > 0.7


def detect_column_type(series: pd.Series) -> str:
    """Propose un type cible pour une colonne.

    Args:
        series: Série pandas à analyser.

    Returns:
        Type cible suggéré parmi TARGET_TYPES.
    """
    dtype = series.dtype

    # Déjà numérique
    if pd.api.types.is_float_dtype(dtype):
        return "Numérique (float)"
    if pd.api.types.is_integer_dtype(dtype):
        return "Entier (int)"
    if pd.api.types.is_bool_dtype(dtype):
        return "Booléen (bool)"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "Date (datetime)"

    # Texte — analyse approfondie
    if series.dtype == "object":
        if _is_boolean_column(series):
            return "Booléen (bool)"
        if _is_numeric_text(series):
            return "Numérique (float)"

        n_unique = series.nunique()
        ratio = n_unique / max(len(series), 1)
        if ratio < 0.05 or n_unique <= 20:
            return "Catégoriel (category)"

        # Tentative de détection de date
        try:
            sample = series.dropna().head(20)
            pd.to_datetime(sample)
            return "Date (datetime)"
        except (ValueError, TypeError):
            pass

        return "Texte (string)"

    return "Texte (string)"


def detect_types(df: pd.DataFrame) -> dict:
    """Détecte automatiquement le type cible pour chaque colonne.

    Args:
        df: DataFrame source.

    Returns:
        Dict {nom_colonne: type_cible_suggéré}.
    """
    return {col: detect_column_type(df[col]) for col in df.columns}


# ═══════════════════════════════════════════════════════════════════
# CONVERSION / TYPAGE
# ═══════════════════════════════════════════════════════════════════
def _clean_numeric_string(value):
    """Nettoie une chaîne pour extraction numérique (€, %, espaces, virgules)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    s = re.sub(r'[€$%\s]', '', s)
    s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan


def _convert_boolean(series: pd.Series) -> pd.Series:
    """Convertit une série texte en booléen."""
    mapping = {k: v for k, v in BOOLEAN_PATTERNS.items()}
    return series.astype(str).str.strip().str.lower().map(mapping)


def apply_typing(df: pd.DataFrame, type_mapping: dict,
                 date_formats: dict = None) -> pd.DataFrame:
    """Applique les conversions de types choisies par l'utilisateur.

    Args:
        df: DataFrame source.
        type_mapping: Dict {nom_colonne: type_cible}.
        date_formats: Dict {nom_colonne: format_date} optionnel.

    Returns:
        Nouveau DataFrame avec les types convertis.
    """
    result = df.copy()
    date_formats = date_formats or {}
    cols_to_drop = []

    for col, target_type in type_mapping.items():
        if col not in result.columns:
            continue

        try:
            if target_type == "À supprimer":
                cols_to_drop.append(col)

            elif target_type == "Numérique (float)":
                if result[col].dtype == "object":
                    result[col] = result[col].apply(_clean_numeric_string)
                else:
                    result[col] = pd.to_numeric(result[col], errors="coerce")

            elif target_type == "Entier (int)":
                if result[col].dtype == "object":
                    result[col] = result[col].apply(_clean_numeric_string)
                result[col] = pd.to_numeric(result[col], errors="coerce")
                # On garde en float si NaN présents (Int64 nullable)
                if result[col].isna().any():
                    result[col] = result[col].astype("Int64")
                else:
                    result[col] = result[col].astype(int)

            elif target_type == "Texte (string)":
                result[col] = result[col].astype(str).replace("nan", np.nan)

            elif target_type == "Catégoriel (category)":
                result[col] = result[col].astype("category")

            elif target_type == "Booléen (bool)":
                result[col] = _convert_boolean(result[col])

            elif target_type == "Date (datetime)":
                fmt = date_formats.get(col)
                if fmt:
                    result[col] = pd.to_datetime(result[col], format=fmt, errors="coerce")
                else:
                    result[col] = pd.to_datetime(result[col], errors="coerce", dayfirst=True)

        except Exception:
            # En cas d'erreur, conserver la colonne telle quelle
            pass

    if cols_to_drop:
        result = result.drop(columns=cols_to_drop, errors="ignore")

    return result
