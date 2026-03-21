# -*- coding: utf-8 -*-
"""
rules_engine.py — Moteur de règles métier centralisé.

Centralise les décisions clés du pipeline ML :
  - Inférence du type de problème à partir de la cible
  - Politique de split adaptée au contexte (chronologique, stratifié, groupe, CV)
  - Métriques recommandées selon le type de problème et le déséquilibre
  - Gates de validation par étape (bloquants et avertissements)
"""

import pandas as pd
import numpy as np

from config import (
    CLASS_IMBALANCE_RATIO,
    MIN_ROWS,
    MIN_COLS,
    DEFAULT_QUALITY_THRESHOLD,
    MAX_NAN_AFTER_CONVERSION_PCT,
    MIN_ROWS_AFTER_CLEANING,
    MIN_FEATURES,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_OVERFIT_PCT,
)


# ═══════════════════════════════════════════════════════════════════
# 1. INFÉRENCE DU TYPE DE PROBLÈME
# ═══════════════════════════════════════════════════════════════════

def infer_problem_type(target_series: pd.Series, user_hint: str = None) -> dict:
    """Déduit le type de problème à partir de la variable cible.

    Args:
        target_series: Colonne cible du DataFrame.
        user_hint: Type choisi par l'utilisateur
                   ("Régression", "Classification", "Série temporelle").

    Returns:
        Dict avec 'inferred', 'confidence', 'reason', 'warning'.
    """
    n_unique = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    n_rows = len(target_series)
    warning = None

    if user_hint in ("Série temporelle", "Détection d'anomalies"):
        return {
            "inferred": user_hint,
            "confidence": "high",
            "reason": "Type choisi explicitement par l'utilisateur.",
            "warning": None,
        }

    if not is_numeric:
        inferred = "Classification"
        reason = f"Cible non-numérique ({n_unique} valeurs uniques)."
        if n_unique > 20:
            warning = (
                f"La cible a {n_unique} valeurs uniques — "
                "vérifiez qu'il ne s'agit pas d'un identifiant."
            )
    elif n_unique <= 2:
        inferred = "Classification"
        reason = f"Cible binaire ({n_unique} valeurs)."
    elif n_unique <= 10 or (n_rows > 0 and n_unique / n_rows < 0.05):
        inferred = "Classification"
        reason = f"Peu de valeurs uniques ({n_unique}) → Classification probable."
        warning = "Si la cible est un score continu, sélectionnez Régression."
    else:
        inferred = "Régression"
        reason = f"Cible numérique continue ({n_unique} valeurs uniques)."

    if user_hint and user_hint != inferred and user_hint != "Série temporelle":
        prefix = (warning + " ") if warning else ""
        warning = (
            f"{prefix}⚠️ Type choisi '{user_hint}' mais la cible ressemble à '{inferred}'."
        ).strip()

    return {
        "inferred": inferred,
        "confidence": "high" if not warning else "medium",
        "reason": reason,
        "warning": warning,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. POLITIQUE DE SPLIT AUTOMATIQUE
# ═══════════════════════════════════════════════════════════════════

def recommend_split_strategy(
    problem_type: str,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list = None,
    imbalance_ratio: float = CLASS_IMBALANCE_RATIO,
) -> dict:
    """Recommande la stratégie de split selon le contexte du dataset.

    Règles de décision (par ordre de priorité) :
    1. Série temporelle              → Chronologique (obligatoire)
    2. Classification déséquilibrée → Stratifié
    3. < 100 lignes                  → Cross-validation recommandée
    4. Colonne groupe/ID détectée    → Groupe
    5. Défaut                        → Aléatoire

    Args:
        problem_type: "Régression", "Classification" ou "Série temporelle".
        df: DataFrame de travail.
        target_col: Nom de la variable cible.
        feature_cols: Colonnes features (pour détecter les colonnes groupes).
        imbalance_ratio: Seuil de ratio déséquilibre classes (défaut config).

    Returns:
        Dict avec 'method', 'reason', 'warning', 'mandatory'.
        method ∈ {"Chronologique", "Stratifié", "Cross-validation", "Groupe", "Aléatoire"}
    """
    feature_cols = feature_cols or []

    # Règle 1 — Série temporelle → split chronologique obligatoire
    if problem_type == "Série temporelle":
        return {
            "method": "Chronologique",
            "reason": (
                "Les séries temporelles exigent un split chronologique : "
                "le train = passé, le test = futur."
            ),
            "warning": (
                "Mélanger aléatoirement des données temporelles crée une fuite "
                "d'information (le modèle verrait le futur à l'entraînement)."
            ),
            "mandatory": True,
        }

    # Règle 2 — Classification déséquilibrée → stratifié
    if problem_type == "Classification" and target_col in df.columns:
        counts = df[target_col].value_counts()
        if len(counts) >= 2:
            ratio = counts.iloc[0] / counts.iloc[-1]
            if ratio > imbalance_ratio:
                return {
                    "method": "Stratifié",
                    "reason": (
                        f"Déséquilibre de classes détecté "
                        f"(ratio {ratio:.1f}:1 > seuil {imbalance_ratio}:1)."
                    ),
                    "warning": (
                        "Sans stratification, la classe rare serait "
                        "sous-représentée dans le jeu de test."
                    ),
                    "mandatory": False,
                }

    # Règle 3 — Très peu de lignes → cross-validation préférable
    if len(df) < 100:
        return {
            "method": "Cross-validation",
            "reason": (
                f"Dataset de {len(df)} lignes : la cross-validation "
                "donne une estimation plus fiable qu'un split unique."
            ),
            "warning": (
                "Avec si peu de données, les résultats d'un split simple "
                "dépendent beaucoup du hasard de la partition."
            ),
            "mandatory": False,
        }

    # Règle 4 — Colonne groupe/ID détectée → split par groupe recommandé
    group_cols = _detect_group_columns(df, feature_cols)
    if group_cols:
        return {
            "method": "Groupe",
            "reason": (
                f"Colonne(s) identifiant groupe détectée(s) : "
                f"{', '.join(group_cols)}."
            ),
            "warning": (
                "Un split aléatoire causerait une fuite de données : "
                "la même entité apparaîtrait dans train et test."
            ),
            "mandatory": False,
            "group_cols": group_cols,
        }

    # Défaut
    return {
        "method": "Aléatoire",
        "reason": "Aucune contrainte spécifique détectée — split aléatoire standard.",
        "warning": None,
        "mandatory": False,
    }


def _detect_group_columns(df: pd.DataFrame, feature_cols: list) -> list:
    """Détecte les colonnes potentiellement identifiants de groupe parmi les features."""
    suspects = []
    id_keywords = {"id", "code", "ref", "num", "key", "uid", "ident", "idx", "identifiant"}
    for col in feature_cols:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique()
        n_rows = len(df)
        if n_rows > 0 and n_unique / n_rows > 0.8:
            col_lower = col.lower()
            if any(kw in col_lower for kw in id_keywords):
                suspects.append(col)
    return suspects


# ═══════════════════════════════════════════════════════════════════
# 3. MÉTRIQUES RECOMMANDÉES PAR CONTEXTE
# ═══════════════════════════════════════════════════════════════════

def recommend_metrics(problem_type: str, is_imbalanced: bool = False) -> dict:
    """Retourne les métriques prioritaires selon le type et le contexte.

    Args:
        problem_type: "Régression", "Classification" ou "Série temporelle".
        is_imbalanced: True si les classes sont déséquilibrées.

    Returns:
        Dict avec 'primary', 'secondary', 'avoid', 'explanation'.
    """
    if problem_type == "Régression":
        return {
            "primary": ["R²", "RMSE"],
            "secondary": ["MAE"],
            "avoid": [],
            "explanation": (
                "R² mesure la variance expliquée. "
                "RMSE pénalise les grandes erreurs. "
                "MAE est plus robuste aux outliers."
            ),
        }

    if problem_type == "Classification":
        if is_imbalanced:
            return {
                "primary": ["F1-Score", "AUC-PR"],
                "secondary": ["AUC-ROC", "Recall"],
                "avoid": ["Accuracy"],
                "explanation": (
                    "Avec classes déséquilibrées, l'Accuracy est trompeuse "
                    "(un modèle qui prédit toujours la classe majoritaire paraît bon). "
                    "F1 et AUC-PR capturent mieux les performances sur les classes rares."
                ),
            }
        return {
            "primary": ["Accuracy", "F1-Score"],
            "secondary": ["AUC-ROC"],
            "avoid": [],
            "explanation": (
                "Accuracy = taux global de bonnes prédictions. "
                "F1 = équilibre précision/rappel. "
                "AUC-ROC mesure la capacité de discrimination."
            ),
        }

    if problem_type == "Série temporelle":
        return {
            "primary": ["MAE", "RMSE"],
            "secondary": ["MAPE"],
            "avoid": ["R²"],
            "explanation": (
                "MAE et RMSE s'interprètent dans l'unité de la série. "
                "MAPE = erreur relative en %. "
                "R² est peu adapté aux séries temporelles (auto-corrélation)."
            ),
        }

    if problem_type == "Détection d'anomalies":
        return {
            "primary": ["Taux d'anomalies", "Stabilité train/test"],
            "secondary": ["Score moyen d'anomalie"],
            "avoid": ["Accuracy", "R²"],
            "explanation": (
                "Sans labels de vérité terrain, on suit surtout la stabilité du "
                "taux d'anomalies entre train et test et la distribution des scores."
            ),
        }

    return {"primary": [], "secondary": [], "avoid": [], "explanation": "Type inconnu."}


def detect_leakage_suspects(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    corr_threshold: float = 0.98,
) -> list:
    """Détecte des features suspectes de fuite de cible.

    Heuristiques :
    - Similarité lexicale avec la cible (nom de colonne)
    - Corrélation quasi parfaite avec la cible (si numérique)
    """
    suspects = []
    if not target_col or target_col not in df.columns:
        return suspects

    tgt = str(target_col).lower().strip()
    for col in feature_cols or []:
        if col not in df.columns:
            continue
        name = str(col).lower().strip()
        if tgt in name or name in tgt:
            suspects.append(col)
            continue

        if (pd.api.types.is_numeric_dtype(df[col]) and
                pd.api.types.is_numeric_dtype(df[target_col])):
            s = pd.concat([df[col], df[target_col]], axis=1).dropna()
            if len(s) > 10:
                corr = abs(s.iloc[:, 0].corr(s.iloc[:, 1]))
                if np.isfinite(corr) and corr >= corr_threshold:
                    suspects.append(col)

    return sorted(set(suspects))


def detect_compliance_risks(columns: list) -> list:
    """Détecte des colonnes potentiellement sensibles (PII) via motifs de noms."""
    pii_keywords = {
        "email", "mail", "phone", "tel", "mobile", "address", "adresse",
        "ssn", "social", "security", "iban", "bic", "carte", "card",
        "birth", "naissance", "nom", "prenom", "firstname", "lastname",
    }
    risks = []
    for col in columns or []:
        c = str(col).lower()
        if any(k in c for k in pii_keywords):
            risks.append(col)
    return sorted(set(risks))


# ═══════════════════════════════════════════════════════════════════
# 4. GATES DE VALIDATION PAR ÉTAPE
# ═══════════════════════════════════════════════════════════════════

STAGE_GATES = {
    "chargement":   ["min_rows", "min_cols"],
    "qualite":      ["quality_score", "max_missing", "no_constant_cols"],
    "preparation":  ["min_rows_after_cleaning", "no_nan_residuals",
                     "min_features", "no_leakage_suspect"],
    "modelisation": ["split_coherent", "min_train_rows"],
    "evaluation":   ["min_score", "max_overfit"],
    "production":   ["metric_threshold", "model_saved", "model_card", "no_critical_overfit"],
}


def evaluate_stage_gates(stage: str, context: dict) -> dict:
    """Évalue les gates d'une étape et retourne les blocages.

    Args:
        stage: Clé d'étape parmi STAGE_GATES.
        context: Dict des valeurs nécessaires à l'évaluation.
                 Clés utilisées selon les gates actifs (voir _evaluate_gate).

    Returns:
        Dict avec 'passed' (bool), 'blocking' (list), 'warnings' (list), 'stage'.
    """
    gates = STAGE_GATES.get(stage, [])
    blocking, warnings = [], []

    for gate in gates:
        result = _evaluate_gate(gate, context)
        if result["level"] == "block":
            blocking.append(result["message"])
        elif result["level"] == "warn":
            warnings.append(result["message"])

    return {
        "passed": len(blocking) == 0,
        "blocking": blocking,
        "warnings": warnings,
        "stage": stage,
    }


def _evaluate_gate(gate: str, ctx: dict) -> dict:
    """Évalue un gate individuel.

    Returns:
        Dict avec 'level' ∈ {'ok', 'warn', 'block'} et 'message'.
    """
    def block(msg): return {"level": "block", "message": msg}
    def warn(msg):  return {"level": "warn",  "message": msg}
    def ok():       return {"level": "ok",    "message": ""}

    if gate == "min_rows":
        n = ctx.get("n_rows", 0)
        return ok() if n >= MIN_ROWS else block(
            f"Trop peu de lignes ({n} < {MIN_ROWS} requis).")

    if gate == "min_cols":
        n = ctx.get("n_cols", 0)
        return ok() if n >= MIN_COLS else block(
            f"Trop peu de colonnes ({n} < {MIN_COLS} requises).")

    if gate == "quality_score":
        score = ctx.get("quality_score", 0)
        return ok() if score >= DEFAULT_QUALITY_THRESHOLD else block(
            f"Score qualité insuffisant ({score:.0f} < {DEFAULT_QUALITY_THRESHOLD}).")

    if gate == "max_missing":
        pct = ctx.get("missing_pct", 0)
        return ok() if pct <= MAX_NAN_AFTER_CONVERSION_PCT else block(
            f"Trop de valeurs manquantes ({pct:.1f}% > {MAX_NAN_AFTER_CONVERSION_PCT}%).")

    if gate == "no_constant_cols":
        n = ctx.get("n_constant_cols", 0)
        return (warn(f"{n} colonne(s) constante(s) à supprimer avant modélisation.")
                if n > 0 else ok())

    if gate == "min_rows_after_cleaning":
        n = ctx.get("n_rows_after_cleaning", ctx.get("n_rows", 0))
        return ok() if n >= MIN_ROWS_AFTER_CLEANING else block(
            f"Trop peu de lignes après nettoyage ({n} < {MIN_ROWS_AFTER_CLEANING}).")

    if gate == "no_nan_residuals":
        n = ctx.get("n_nan_residuals", 0)
        return ok() if n == 0 else block(
            f"{n} valeur(s) manquante(s) résiduelles dans les features/cible.")

    if gate == "min_features":
        n = ctx.get("n_features", 0)
        return ok() if n >= MIN_FEATURES else block(
            f"Trop peu de features ({n} < {MIN_FEATURES} requises).")

    if gate == "no_leakage_suspect":
        suspects = ctx.get("leakage_suspects", [])
        return (warn(f"Features suspectes de leakage : {', '.join(suspects)}.")
                if suspects else ok())

    if gate == "split_coherent":
        problem = ctx.get("problem_type", "")
        split = ctx.get("split_method", "")
        if problem == "Série temporelle" and split != "Chronologique":
            return block(
                "Le split chronologique est obligatoire pour les séries temporelles.")
        return ok()

    if gate == "min_train_rows":
        n = ctx.get("n_train_rows", 0)
        return ok() if n >= 40 else block(
            f"Jeu d'entraînement trop petit ({n} lignes < 40 minimum).")

    if gate == "min_score":
        score = ctx.get("best_test_score", 0)
        return ok() if score >= DEFAULT_MIN_SCORE else warn(
            f"Score test faible ({score:.3f} < seuil {DEFAULT_MIN_SCORE}).")

    if gate == "max_overfit":
        overfit = ctx.get("overfit_pct", 0)
        return (warn(
            f"Sur-apprentissage détecté ({overfit:.1f}% > {DEFAULT_MAX_OVERFIT_PCT}%).")
                if overfit > DEFAULT_MAX_OVERFIT_PCT else ok())

    if gate == "metric_threshold":
        score = ctx.get("best_test_score", 0)
        threshold = ctx.get("metric_threshold", DEFAULT_MIN_SCORE)
        return ok() if score >= threshold else block(
            f"Score {score:.3f} sous le seuil métier {threshold:.3f}.")

    if gate == "model_saved":
        return ok() if ctx.get("model_saved", False) else block(
            "Modèle non sauvegardé — exportez le modèle avant de déployer.")

    if gate == "model_card":
        required = ["nom_modele", "type_probleme", "score_test", "date_entrainement"]
        missing = [k for k in required if not ctx.get(k)]
        return (warn(
            f"Model card incomplète — champs manquants : {', '.join(missing)}.")
                if missing else ok())

    if gate == "no_critical_overfit":
        overfit = ctx.get("overfit_pct", 0)
        return (block(
            f"Sur-apprentissage critique ({overfit:.1f}% > 15%) — modèle non déployable.")
                if overfit > 15 else ok())

    return ok()
