# -*- coding: utf-8 -*-
"""
data_utils.py — Fonctions utilitaires pour la manipulation des données.

Regroupe les fonctions de nettoyage, encodage, transformation
utilisées par les modules de l'interface.
"""

import pandas as pd
import numpy as np


def apercu_avant_apres(df_avant: pd.DataFrame, df_apres: pd.DataFrame,
                       n: int = 5) -> dict:
    """Compare avant/après une transformation.

    Returns:
        dict avec clés: avant, apres, colonnes_modifiees, lignes_avant, lignes_apres
    """
    colonnes_modifiees = []
    cols_communes = [c for c in df_avant.columns if c in df_apres.columns]
    for col in cols_communes:
        if not df_avant[col].equals(df_apres[col]):
            colonnes_modifiees.append(col)

    colonnes_ajoutees = [c for c in df_apres.columns if c not in df_avant.columns]
    colonnes_supprimees = [c for c in df_avant.columns if c not in df_apres.columns]

    return {
        "avant": df_avant.head(n),
        "apres": df_apres.head(n),
        "colonnes_modifiees": colonnes_modifiees,
        "colonnes_ajoutees": colonnes_ajoutees,
        "colonnes_supprimees": colonnes_supprimees,
        "lignes_avant": len(df_avant),
        "lignes_apres": len(df_apres),
        "lignes_diff": len(df_avant) - len(df_apres),
    }


def resume_dataframe(df: pd.DataFrame) -> dict:
    """Résumé rapide d'un DataFrame pour la sidebar."""
    if df is None or df.empty:
        return {"lignes": 0, "colonnes": 0, "statut": "Vide"}

    na_total = int(df.isna().sum().sum())
    na_pct = round(na_total / (len(df) * len(df.columns)) * 100, 1) if len(df) > 0 else 0

    n_numeriques = len(df.select_dtypes(include=[np.number]).columns)
    n_categoriques = len(df.select_dtypes(include=["object", "category"]).columns)
    n_datetime = len(df.select_dtypes(include=["datetime64"]).columns)

    return {
        "lignes": len(df),
        "colonnes": len(df.columns),
        "na_total": na_total,
        "na_pct": na_pct,
        "n_numeriques": n_numeriques,
        "n_categoriques": n_categoriques,
        "n_datetime": n_datetime,
    }


def detecter_statut_donnees(session_state: dict) -> str:
    """Détermine le statut actuel des données dans le pipeline."""
    if session_state.get("optimisation_done"):
        return "Optimisées"
    if session_state.get("entrainement_done"):
        return "Modélisées"
    if session_state.get("transformation_done"):
        return "Prêtes"
    if session_state.get("nettoyage_done"):
        return "Nettoyées"
    if session_state.get("diagnostic_done"):
        return "Diagnostiquées"
    if session_state.get("cible_done"):
        return "Cible définie"
    if session_state.get("consolidation_done"):
        return "Consolidées"
    if session_state.get("typage_done"):
        return "Typées"
    if session_state.get("chargement_done"):
        return "Brutes"
    return "Non chargées"


def recommend_models(df: pd.DataFrame, target_col: str, problem_type: str) -> list:
    """Recommande des modèles basés sur les caractéristiques des données.

    Critères :
    - < 1000 lignes → Modèles simples (Linear, Ridge, KNN)
    - > 10000 lignes → Gradient Boosting, Random Forest
    - Beaucoup de catégoriels → Tree-based (RF, GB)
    - Relations linéaires (corr > 0.7) → Linear, Ridge, Lasso
    - Cible déséquilibrée → RF avec class_weight, pas NaiveBayes
    - Peu de features (< 5) → KNN, Linear
    - Beaucoup de features (> 50) → Lasso, ElasticNet
    - Distribution cible asymétrique → Suggérer log-transform

    Returns:
        Liste de dicts: [{"nom": str, "raison": str, "priorite": int}]
    """
    recommendations = []
    n_rows = len(df)
    n_features = len(df.columns) - 1  # sans la cible
    n_cat = len(df.select_dtypes(include=["object", "category"]).columns)
    n_num = len(df.select_dtypes(include=[np.number]).columns)

    # Vérifier si des corrélations linéaires fortes existent
    has_linear = False
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols and len(num_cols) >= 2:
            corr_with_target = df[num_cols].corr()[target_col].drop(target_col, errors="ignore").abs()
            has_linear = (corr_with_target > 0.7).any()

    # Vérifier le déséquilibre de la cible (classification)
    is_imbalanced = False
    if problem_type == "Classification" and target_col in df.columns:
        counts = df[target_col].value_counts()
        if len(counts) >= 2:
            ratio = counts.iloc[0] / max(counts.iloc[-1], 1)
            is_imbalanced = ratio > 5

    # Vérifier l'asymétrie de la cible (régression)
    is_skewed = False
    if problem_type == "Régression" and target_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            skew = df[target_col].skew()
            is_skewed = abs(skew) > 1.5

    if problem_type == "Régression":
        if n_rows < 1000:
            recommendations.append({
                "nom": "Régression Linéaire",
                "raison": f"Peu de données ({n_rows} lignes) → modèle simple recommandé",
                "priorite": 1})
            recommendations.append({
                "nom": "Ridge",
                "raison": "Régularisation pour éviter le surapprentissage sur petit dataset",
                "priorite": 2})
            if n_features < 10:
                recommendations.append({
                    "nom": "KNN",
                    "raison": f"Peu de features ({n_features}) — KNN fonctionne bien",
                    "priorite": 3})
        else:
            recommendations.append({
                "nom": "Random Forest",
                "raison": f"Dataset de taille correcte ({n_rows} lignes), bon compromis",
                "priorite": 1})
            recommendations.append({
                "nom": "Gradient Boosting",
                "raison": "Souvent le plus performant sur les datasets moyens à grands",
                "priorite": 2})

        if has_linear:
            recommendations.append({
                "nom": "Régression Linéaire",
                "raison": "Relations linéaires détectées (corrélation > 0.7 avec la cible)",
                "priorite": 1})
            recommendations.append({
                "nom": "Ridge",
                "raison": "Linéaire avec régularisation pour plus de robustesse",
                "priorite": 2})

        if n_features > 50:
            recommendations.append({
                "nom": "Lasso",
                "raison": f"Beaucoup de features ({n_features}) → sélection auto",
                "priorite": 1})
            recommendations.append({
                "nom": "ElasticNet",
                "raison": "Combinaison Lasso + Ridge pour la sélection de variables",
                "priorite": 2})

        if n_cat > n_num:
            recommendations.append({
                "nom": "Random Forest",
                "raison": f"Beaucoup de variables catégorielles ({n_cat}) → arbres recommandés",
                "priorite": 1})

    elif problem_type == "Classification":
        if n_rows < 1000:
            recommendations.append({
                "nom": "Régression Logistique",
                "raison": f"Peu de données ({n_rows} lignes) → modèle simple",
                "priorite": 1})
            recommendations.append({
                "nom": "KNN",
                "raison": "Simple et efficace sur petits datasets",
                "priorite": 2})
        else:
            recommendations.append({
                "nom": "Random Forest",
                "raison": "Robuste et performant sur datasets moyens à grands",
                "priorite": 1})
            recommendations.append({
                "nom": "Gradient Boosting",
                "raison": "Souvent le meilleur en classification",
                "priorite": 2})

        if is_imbalanced:
            recommendations.append({
                "nom": "Random Forest",
                "raison": "Classes déséquilibrées → RF avec class_weight='balanced'",
                "priorite": 1})
            # Retirer Naive Bayes si présent
            recommendations = [r for r in recommendations if r["nom"] != "Naive Bayes"]

        if n_cat > n_num:
            recommendations.append({
                "nom": "Random Forest",
                "raison": f"Données mixtes avec beaucoup de catégoriels ({n_cat})",
                "priorite": 1})
            recommendations.append({
                "nom": "Gradient Boosting",
                "raison": "Arbres de décision : gèrent bien les données mixtes",
                "priorite": 2})

        if has_linear:
            recommendations.append({
                "nom": "Régression Logistique",
                "raison": "Relations linéaires détectées — logistique souvent suffisant",
                "priorite": 2})

    # Dédupliquer et trier par priorité
    seen = set()
    unique_recs = []
    for r in sorted(recommendations, key=lambda x: x["priorite"]):
        if r["nom"] not in seen:
            seen.add(r["nom"])
            unique_recs.append(r)

    # Ajouter des alertes
    alertes = []
    if is_skewed:
        alertes.append("⚠️ Cible très asymétrique → envisagez une transformation log avant modélisation")
    if is_imbalanced:
        alertes.append("⚠️ Classes déséquilibrées → utilisez class_weight='balanced'")

    return {
        "modeles": unique_recs[:5],
        "alertes": alertes,
        "n_rows": n_rows,
        "n_features": n_features,
        "has_linear": has_linear,
        "is_imbalanced": is_imbalanced,
        "is_skewed": is_skewed,
    }


def recommend_preprocessing(df: pd.DataFrame, target_col: str = None) -> dict:
    """Recommande les étapes de preprocessing basées sur l'analyse des données.

    Returns:
        dict avec clés: scaling, encoding, alertes
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Scaling
    scaling_needed = False
    scaling_reason = ""
    if len(num_cols) >= 2:
        ranges = {}
        for col in num_cols:
            if col == target_col:
                continue
            s = df[col].dropna()
            if len(s) > 0:
                ranges[col] = (s.min(), s.max())
        if ranges:
            mins = [v[0] for v in ranges.values()]
            maxs = [v[1] for v in ranges.values()]
            ratio = max(maxs) / max(abs(min(mins)), 1) if min(mins) != 0 else max(maxs)
            if ratio > 100 or (max(maxs) - min(mins) > 10000):
                scaling_needed = True
                examples = sorted(ranges.items(), key=lambda x: x[1][1] - x[1][0], reverse=True)[:3]
                scaling_reason = ("Échelles très différentes : " +
                                  ", ".join(f"{c} [{v[0]:.0f}-{v[1]:.0f}]" for c, v in examples))

    # Encoding
    encoding_recommendations = []
    for col in cat_cols:
        if col == target_col:
            continue
        n_unique = df[col].nunique()
        if n_unique <= 2:
            encoding_recommendations.append({
                "colonne": col,
                "n_valeurs": n_unique,
                "methode": "Label Encoding",
                "raison": f"Seulement {n_unique} valeurs → simple numérotation",
            })
        elif n_unique <= 10:
            encoding_recommendations.append({
                "colonne": col,
                "n_valeurs": n_unique,
                "methode": "One-Hot Encoding",
                "raison": f"{n_unique} valeurs → une colonne par catégorie",
            })
        else:
            encoding_recommendations.append({
                "colonne": col,
                "n_valeurs": n_unique,
                "methode": "Target Encoding",
                "raison": f"{n_unique} valeurs (trop pour One-Hot) → encoder par la cible",
            })

    # Alertes
    alertes = []
    for col in num_cols:
        if col == target_col:
            continue
        skew = df[col].dropna().skew()
        if abs(skew) > 2:
            alertes.append(f"{col} très asymétrique (skew={skew:.1f}) → envisagez log transform")

    na_cols = [(c, df[c].isna().mean() * 100) for c in df.columns if df[c].isna().any()]
    for col, pct in sorted(na_cols, key=lambda x: -x[1]):
        if pct > 5:
            alertes.append(f"{pct:.0f}% de valeurs manquantes dans « {col} »")

    return {
        "scaling": {"needed": scaling_needed, "reason": scaling_reason},
        "encoding": encoding_recommendations,
        "alertes": alertes,
    }
