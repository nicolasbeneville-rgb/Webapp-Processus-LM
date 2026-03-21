# -*- coding: utf-8 -*-
"""
validators.py — Points de validation systématiques du pipeline ML.

Chaque fonction renvoie un dictionnaire :
    {
        "passed": bool,
        "status": "✅" | "⚠️" | "❌",
        "message": str,          # message court affiché à l'utilisateur
        "details": list[str],    # liste des détails / recommandations
    }
"""

import pandas as pd
import numpy as np
from config import (
    MIN_ROWS,
    MIN_COLS,
    MAX_NAN_AFTER_CONVERSION_PCT,
    MAX_JOIN_LOSS_PCT,
    DEFAULT_QUALITY_THRESHOLD,
    CORRELATION_THRESHOLD,
    MIN_ROWS_AFTER_CLEANING,
    MIN_FEATURES,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_OVERFIT_PCT,
    MIN_IMPROVEMENT_PCT,
    CLASS_IMBALANCE_RATIO,
)


# ─────────────────────────── helpers ──────────────────────────────
def _result(passed: bool, message: str, details: list = None, warning: bool = False):
    """Fabrique un dictionnaire-résultat normalisé."""
    if warning:
        status = "⚠️"
    else:
        status = "✅" if passed else "❌"
    return {
        "passed": passed,
        "status": status,
        "message": message,
        "details": details or [],
    }


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 1 — Chargement des fichiers
# ═══════════════════════════════════════════════════════════════════
def validate_loaded_file(df: pd.DataFrame, filename: str,
                         min_rows: int = MIN_ROWS,
                         min_cols: int = MIN_COLS) -> dict:
    """Vérifie qu'un fichier chargé respecte les critères minimum.

    Args:
        df: DataFrame chargé.
        filename: Nom du fichier (pour les messages).
        min_rows: Nombre minimum de lignes.
        min_cols: Nombre minimum de colonnes.

    Returns:
        Dictionnaire de résultat de validation.
    """
    details = []
    passed = True

    if len(df) < min_rows:
        passed = False
        details.append(
            f"Le fichier « {filename} » ne contient que {len(df)} lignes "
            f"(minimum requis : {min_rows})."
        )

    if len(df.columns) < min_cols:
        passed = False
        details.append(
            f"Le fichier « {filename} » ne contient que {len(df.columns)} colonnes "
            f"(minimum requis : {min_cols})."
        )

    if passed:
        message = f"✅ « {filename} » — {len(df)} lignes × {len(df.columns)} colonnes — OK"
    else:
        message = f"❌ « {filename} » ne respecte pas les critères minimum."

    return _result(passed, message, details)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 2 — Typage et conversion
# ═══════════════════════════════════════════════════════════════════
def validate_after_conversion(df_before: pd.DataFrame, df_after: pd.DataFrame,
                              max_nan_pct: float = MAX_NAN_AFTER_CONVERSION_PCT) -> dict:
    """Vérifie que la conversion de types n'a pas créé trop de NaN.

    Args:
        df_before: DataFrame avant conversion.
        df_after: DataFrame après conversion.
        max_nan_pct: Pourcentage maximum de NaN tolérés après conversion.

    Returns:
        Dictionnaire de résultat de validation.
    """
    details = []
    passed = True

    for col in df_after.columns:
        if col not in df_before.columns:
            continue
        nan_before = df_before[col].isna().sum()
        nan_after = df_after[col].isna().sum()
        new_nans = nan_after - nan_before
        if new_nans > 0:
            pct = round(new_nans / len(df_after) * 100, 1)
            if pct > max_nan_pct:
                passed = False
                details.append(
                    f"Colonne « {col} » : {new_nans} nouvelles valeurs manquantes "
                    f"({pct}% — seuil : {max_nan_pct}%). Vérifiez le format source."
                )
            else:
                details.append(
                    f"Colonne « {col} » : {new_nans} nouvelles valeurs manquantes "
                    f"({pct}%) — acceptable."
                )

    if passed:
        message = "✅ Conversion des types effectuée sans anomalie majeure."
    else:
        message = "❌ Trop de valeurs manquantes créées lors de la conversion."

    return _result(passed, message, details)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 3 — Consolidation / jointures
# ═══════════════════════════════════════════════════════════════════
def validate_join(df_left: pd.DataFrame, df_result: pd.DataFrame,
                  max_loss_pct: float = MAX_JOIN_LOSS_PCT) -> dict:
    """Vérifie que la jointure n'a pas perdu trop de lignes.

    Args:
        df_left: DataFrame de gauche avant jointure.
        df_result: DataFrame résultat de la jointure.
        max_loss_pct: Perte maximum tolérée en pourcentage.

    Returns:
        Dictionnaire de résultat de validation.
    """
    details = []
    rows_before = len(df_left)
    rows_after = len(df_result)

    if rows_before == 0:
        return _result(False, "❌ La table de gauche est vide.", [])

    loss_pct = round((1 - rows_after / rows_before) * 100, 1)

    # Colonnes dupliquées
    dup_cols = [c for c in df_result.columns if c.endswith("_x") or c.endswith("_y")]
    if dup_cols:
        details.append(
            f"Colonnes dupliquées détectées : {', '.join(dup_cols)}. "
            "Renommez ou supprimez les doublons."
        )

    if loss_pct > max_loss_pct:
        details.append(
            f"La jointure a réduit le nombre de lignes de {rows_before} à {rows_after} "
            f"(perte de {loss_pct}% — seuil : {max_loss_pct}%)."
        )
        return _result(False, "❌ Trop de lignes perdues lors de la jointure.", details)

    if loss_pct > 0:
        details.append(
            f"Lignes : {rows_before} → {rows_after} (perte de {loss_pct}%)."
        )
        return _result(True, "⚠️ Jointure effectuée avec perte modérée.", details, warning=True)

    details.append(f"Lignes : {rows_before} → {rows_after}.")
    return _result(True, "✅ Jointure effectuée sans perte.", details)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 4 — Audit / qualité
# ═══════════════════════════════════════════════════════════════════
def validate_data_quality(quality_score: float, multicollinearity: bool,
                          target_imbalanced: bool,
                          threshold: float = DEFAULT_QUALITY_THRESHOLD) -> dict:
    """Vérifie la qualité globale du dataset.

    Args:
        quality_score: Score calculé par l'audit (0-100).
        multicollinearity: True si multicolinéarité détectée.
        target_imbalanced: True si la variable cible est déséquilibrée.
        threshold: Score minimum acceptable.

    Returns:
        Dictionnaire de résultat de validation.
    """
    details = []
    passed = True

    if quality_score < threshold:
        passed = False
        details.append(
            f"Score de qualité : {quality_score}/100 (seuil : {threshold}). "
            "Améliorez la qualité des données avant de poursuivre."
        )
    else:
        details.append(f"Score de qualité : {quality_score}/100 — OK.")

    if multicollinearity:
        details.append(
            "⚠️ Multicolinéarité détectée : certaines variables sont très corrélées "
            f"(> {CORRELATION_THRESHOLD}). Envisagez d'en supprimer certaines."
        )

    if target_imbalanced:
        details.append(
            "⚠️ La variable cible est déséquilibrée. "
            "Envisagez un rééchantillonnage ou utilisez des métriques adaptées (F1, AUC)."
        )

    if passed and not multicollinearity and not target_imbalanced:
        message = "✅ Qualité des données satisfaisante."
    elif passed:
        message = "⚠️ Qualité acceptable mais des points d'attention subsistent."
        passed = True  # avertissement uniquement
    else:
        message = "❌ Qualité insuffisante — actions correctives nécessaires."

    warning = passed and (multicollinearity or target_imbalanced)
    return _result(passed, message, details, warning=warning)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 5 — Préparation des données
# ═══════════════════════════════════════════════════════════════════
def validate_prepared_data(df: pd.DataFrame, target_col: str,
                           feature_cols: list,
                           min_rows: int = MIN_ROWS_AFTER_CLEANING,
                           min_features: int = MIN_FEATURES) -> dict:
    """Vérifie que le dataset est prêt pour la modélisation.

    Args:
        df: DataFrame nettoyé.
        target_col: Nom de la variable cible.
        feature_cols: Liste des variables explicatives.
        min_rows: Nombre minimum de lignes.
        min_features: Nombre minimum de features.

    Returns:
        Dictionnaire de résultat de validation.
    """
    details = []
    passed = True

    # Nombre de lignes
    if len(df) < min_rows:
        passed = False
        details.append(
            f"Seulement {len(df)} lignes restantes (minimum : {min_rows}). "
            "Revoyez vos choix de nettoyage."
        )

    # Nombre de features
    if len(feature_cols) < min_features:
        passed = False
        details.append(
            f"Seulement {len(feature_cols)} variable(s) explicative(s) "
            f"(minimum : {min_features})."
        )

    # NaN résiduels
    subset = feature_cols + [target_col] if target_col in df.columns else feature_cols
    existing = [c for c in subset if c in df.columns]
    total_nan = df[existing].isna().sum().sum()
    if total_nan > 0:
        passed = False
        nan_cols = df[existing].isna().sum()
        nan_detail = nan_cols[nan_cols > 0]
        details.append(
            f"Il reste {total_nan} valeur(s) manquante(s) dans le dataset final. "
            "Toutes les colonnes doivent être complètes."
        )
        for col, cnt in nan_detail.items():
            details.append(f"  • {col} : {cnt} NaN")

    if passed:
        message = (
            f"✅ Dataset prêt : {len(df)} lignes × {len(feature_cols)} features. "
            f"Variable cible : « {target_col} »."
        )
    else:
        message = "❌ Le dataset n'est pas encore prêt pour la modélisation."

    return _result(passed, message, details)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 6 — Modélisation (scores)
# ═══════════════════════════════════════════════════════════════════
def validate_model_scores(results: list,
                          min_score: float = DEFAULT_MIN_SCORE,
                          max_overfit_pct: float = DEFAULT_MAX_OVERFIT_PCT) -> dict:
    """Vérifie les scores des modèles entraînés.

    Args:
        results: Liste de dicts avec clés 'name', 'train_score', 'test_score'.
        min_score: Score test minimum attendu.
        max_overfit_pct: Écart maximum train/test (en %).

    Returns:
        Dictionnaire de résultat de validation.
    """
    details = []
    any_passed = False

    for r in results:
        name = r["name"]
        train_s = r["train_score"]
        test_s = r["test_score"]
        overfit = abs(train_s - test_s) * 100

        status_parts = []
        model_ok = True

        if test_s < min_score:
            model_ok = False
            status_parts.append(f"score test {test_s:.3f} < seuil {min_score}")

        if overfit > max_overfit_pct:
            model_ok = False
            status_parts.append(
                f"écart train/test {overfit:.1f}% > seuil {max_overfit_pct}%"
            )

        if model_ok:
            any_passed = True
            details.append(f"✅ {name} — score test : {test_s:.3f}")
        else:
            details.append(f"❌ {name} — {', '.join(status_parts)}")

    if any_passed:
        message = "✅ Au moins un modèle satisfait les critères de validation."
    else:
        message = (
            "❌ Aucun modèle n'atteint les critères fixés. "
            "Recommandations : retourner à l'étape de préparation, "
            "ajouter des variables ou ajuster les seuils."
        )

    return _result(any_passed, message, details)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 7 — Optimisation des hyperparamètres
# ═══════════════════════════════════════════════════════════════════
def validate_optimization(score_before: float, score_after: float,
                          min_improvement: float = MIN_IMPROVEMENT_PCT) -> dict:
    """Vérifie que l'optimisation apporte une amélioration significative.

    Args:
        score_before: Score avant optimisation.
        score_after: Score après optimisation.
        min_improvement: Amélioration minimum attendue (%).

    Returns:
        Dictionnaire de résultat de validation.
    """
    delta = score_after - score_before
    delta_pct = (delta / max(abs(score_before), 1e-9)) * 100

    if delta_pct >= min_improvement:
        return _result(
            True,
            f"✅ Amélioration de {delta_pct:.2f}% après optimisation.",
            [f"Score avant : {score_before:.4f} → après : {score_after:.4f}"],
        )
    elif delta_pct >= 0:
        return _result(
            True,
            f"⚠️ Amélioration négligeable ({delta_pct:.2f}% < {min_improvement}%).",
            [
                f"Score avant : {score_before:.4f} → après : {score_after:.4f}",
                "L'optimisation n'apporte pas de gain significatif. "
                "Vous pouvez conserver les paramètres par défaut.",
            ],
            warning=True,
        )
    else:
        return _result(
            False,
            f"❌ Dégradation du score ({delta_pct:.2f}%).",
            [
                f"Score avant : {score_before:.4f} → après : {score_after:.4f}",
                "Conservez les paramètres d'origine.",
            ],
        )


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 8 — Analyse des résidus
# ═══════════════════════════════════════════════════════════════════
def validate_residuals(residuals: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyse les résidus et détecte les problèmes courants.

    Args:
        residuals: Array des résidus (y_true - y_pred).
        y_pred: Array des valeurs prédites.

    Returns:
        Dictionnaire de résultat de validation.
    """
    from scipy import stats

    details = []
    passed = True

    # Biais systématique
    mean_resid = np.mean(residuals)
    if abs(mean_resid) > 0.1 * np.std(residuals):
        details.append(
            f"⚠️ Biais systématique détecté : moyenne des résidus = {mean_resid:.4f}. "
            "Le modèle surestime ou sous-estime systématiquement."
        )
        passed = False

    # Test de normalité (Shapiro-Wilk sur échantillon)
    sample = residuals[:min(5000, len(residuals))]
    try:
        _, p_value = stats.shapiro(sample)
        if p_value < 0.05:
            details.append(
                f"⚠️ Les résidus ne suivent pas une distribution normale "
                f"(test de Shapiro-Wilk, p = {p_value:.4f}). "
                "Envisagez une transformation de la variable cible."
            )
    except Exception:
        details.append("ℹ️ Test de normalité non effectué (échantillon trop petit).")

    # Hétéroscédasticité (corrélation entre |résidus| et prédictions)
    try:
        corr = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
        if abs(corr) > 0.3:
            details.append(
                f"⚠️ Hétéroscédasticité possible (corrélation |résidus|/prédictions = {corr:.3f}). "
                "La variance des erreurs dépend de la valeur prédite."
            )
            passed = False
    except Exception:
        pass

    if passed and not details:
        message = "✅ Résidus conformes — pas de problème détecté."
    elif passed:
        message = "⚠️ Résidus globalement corrects mais des points d'attention existent."
    else:
        message = "❌ Problèmes détectés dans les résidus."

    warning = not passed or bool(details)
    return _result(passed, message, details, warning=warning and passed)


# ═══════════════════════════════════════════════════════════════════
# RÉSUMÉ GLOBAL — Tableau de bord de toutes les validations
# ═══════════════════════════════════════════════════════════════════
def validation_dashboard(validations: dict) -> dict:
    """Agrège tous les résultats de validation en un tableau de bord.

    Args:
        validations: Dict {nom_étape: résultat_validation}.

    Returns:
        Dictionnaire avec le résumé et le score de confiance.
    """
    total = len(validations)
    passed_count = sum(1 for v in validations.values() if v["passed"])
    confidence = round(passed_count / total * 100) if total > 0 else 0

    summary = []
    for step_name, result in validations.items():
        summary.append({
            "Étape": step_name,
            "Statut": result["status"],
            "Message": result["message"],
        })

    return {
        "summary": summary,
        "confidence_score": confidence,
        "total_checks": total,
        "passed_checks": passed_count,
    }


# ═══════════════════════════════════════════════════════════════════
# VALIDATION 9 — Readiness production
# ═══════════════════════════════════════════════════════════════════

def validate_production_readiness(
    rapport: dict,
    model_result: dict,
    problem_type: str,
    metric_threshold: float = DEFAULT_MIN_SCORE,
) -> dict:
    """Vérifie qu'un modèle est prêt pour la production.

    Gates vérifiés :
    1. Score test ≥ seuil métier
    2. Modèle sauvegardé (chemin projet présent)
    3. Sur-apprentissage < 15% (seuil critique)
    4. Métadonnées minimales présentes (model card)

    Args:
        rapport: Dictionnaire du projet (état complet).
        model_result: Résultat d'entraînement du meilleur modèle.
        problem_type: Type de problème ML.
        metric_threshold: Seuil métier minimum accepté.

    Returns:
        Dictionnaire de résultat de validation.
    """
    from src.rules_engine import evaluate_stage_gates

    modele_meta = (rapport or {}).get("modele", {})
    mr = model_result or {}

    ctx = {
        "best_test_score": mr.get("test_score", 0),
        "overfit_pct": mr.get("overfit_pct", 0),
        "model_saved": bool(rapport and rapport.get("chemin")),
        "metric_threshold": metric_threshold,
        "nom_modele": modele_meta.get("nom"),
        "type_probleme": problem_type,
        "score_test": mr.get("test_score"),
        "date_entrainement": modele_meta.get("date") or modele_meta.get("trained_at"),
    }

    result = evaluate_stage_gates("production", ctx)
    details = result["blocking"] + result["warnings"]
    passed = result["passed"]

    if passed and not result["warnings"]:
        message = "✅ Modèle prêt pour la production."
    elif passed:
        message = "⚠️ Modèle déployable avec précautions."
    else:
        message = "❌ Modèle non prêt pour la production — corrigez les blocages."

    return _result(passed, message, details, warning=passed and bool(result["warnings"]))
