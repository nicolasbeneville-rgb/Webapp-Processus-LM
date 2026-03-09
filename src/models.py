# -*- coding: utf-8 -*-
"""
models.py — Entraînement des modèles de Machine Learning.

Fonctions principales :
    - get_model            : instancie un modèle selon son nom et paramètres
    - train_model          : entraîne un modèle et retourne les scores
    - train_multiple       : entraîne plusieurs modèles et les compare
    - optimize_model       : Grid Search / Random Search
"""

import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)

# Modèles de régression
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from config import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, DEFAULT_CV_FOLDS


# ═══════════════════════════════════════════════════════════════════
# INSTANCIATION DES MODÈLES
# ═══════════════════════════════════════════════════════════════════
def get_model(name: str, problem_type: str, params: dict = None):
    """Instancie un modèle scikit-learn selon son nom.

    Args:
        name: Nom du modèle (tel que dans config.REGRESSION_MODELS / CLASSIFICATION_MODELS).
        problem_type: "Régression" ou "Classification".
        params: Hyperparamètres optionnels.

    Returns:
        Objet modèle scikit-learn.
    """
    params = params or {}

    # Régression polynomiale : Pipeline spécial
    if name.startswith("Régression Polynomiale"):
        degree = 2 if "degré 2" in name else 3
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", LinearRegression()),
        ])

    models_registry = {
        # ── Régression ──
        ("Régression Linéaire", "Régression"): LinearRegression,
        ("Ridge", "Régression"): Ridge,
        ("Lasso", "Régression"): Lasso,
        ("ElasticNet", "Régression"): ElasticNet,
        ("Arbre de décision", "Régression"): DecisionTreeRegressor,
        ("Random Forest", "Régression"): RandomForestRegressor,
        ("Gradient Boosting", "Régression"): GradientBoostingRegressor,
        ("SVR", "Régression"): SVR,
        # ── Classification ──
        ("Régression Logistique", "Classification"): LogisticRegression,
        ("KNN", "Classification"): KNeighborsClassifier,
        ("Arbre de décision", "Classification"): DecisionTreeClassifier,
        ("Random Forest", "Classification"): RandomForestClassifier,
        ("Gradient Boosting", "Classification"): GradientBoostingClassifier,
        ("SVM", "Classification"): SVC,
        ("Naive Bayes", "Classification"): GaussianNB,
    }

    key = (name, problem_type)
    if key not in models_registry:
        raise ValueError(f"Modèle inconnu : {name} pour {problem_type}")

    model_class = models_registry[key]

    # Paramètres spécifiques
    safe_params = {}
    for p, v in params.items():
        try:
            model_class(**{p: v})
            safe_params[p] = v
        except TypeError:
            continue

    # Fixer le random_state si possible
    try:
        return model_class(random_state=DEFAULT_RANDOM_STATE, **safe_params)
    except TypeError:
        return model_class(**safe_params)


# ═══════════════════════════════════════════════════════════════════
# GRILLES D'HYPERPARAMÈTRES PAR DÉFAUT
# ═══════════════════════════════════════════════════════════════════
DEFAULT_PARAM_GRIDS = {
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet": {
        "alpha": [0.01, 0.1, 1],
        "l1_ratio": [0.2, 0.5, 0.8],
    },
    "Arbre de décision": {"max_depth": [3, 5, 7, 10, 15, None]},
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "Régression Logistique": {"C": [0.01, 0.1, 1, 10, 100]},
    "KNN": {"n_neighbors": [3, 5, 7, 11, 15]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
}


# ═══════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT D'UN MODÈLE
# ═══════════════════════════════════════════════════════════════════
def train_model(model, X_train, y_train, X_test, y_test,
                problem_type: str, cv_folds: int = 0) -> dict:
    """Entraîne un modèle et calcule les métriques.

    Args:
        model: Modèle scikit-learn.
        X_train, y_train: Données d'entraînement.
        X_test, y_test: Données de test.
        problem_type: "Régression" ou "Classification".
        cv_folds: Nombre de folds pour cross-validation (0 = pas de CV).

    Returns:
        Dictionnaire avec les scores et le modèle entraîné.
    """
    start = time.time()

    try:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        elapsed = round(time.time() - start, 2)

        result = {
            "model": model,
            "train_pred": train_pred,
            "test_pred": test_pred,
            "time": elapsed,
            "error": None,
        }

        if problem_type == "Régression":
            result["train_score"] = round(r2_score(y_train, train_pred), 4)
            result["test_score"] = round(r2_score(y_test, test_pred), 4)
            result["rmse"] = round(np.sqrt(mean_squared_error(y_test, test_pred)), 4)
            result["mae"] = round(mean_absolute_error(y_test, test_pred), 4)
        else:
            result["train_score"] = round(accuracy_score(y_train, train_pred), 4)
            result["test_score"] = round(accuracy_score(y_test, test_pred), 4)
            result["f1"] = round(f1_score(y_test, test_pred, average="weighted", zero_division=0), 4)
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        result["auc"] = round(roc_auc_score(y_test, proba[:, 1]), 4)
                    else:
                        result["auc"] = round(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"), 4)
                else:
                    result["auc"] = None
            except Exception:
                result["auc"] = None

        result["overfit_pct"] = round(abs(result["train_score"] - result["test_score"]) * 100, 2)

        # Cross-validation
        if cv_folds > 0:
            scoring = "r2" if problem_type == "Régression" else "accuracy"
            try:
                cv_scores = cross_val_score(model, X_train, y_train,
                                            cv=cv_folds, scoring=scoring)
                result["cv_mean"] = round(cv_scores.mean(), 4)
                result["cv_std"] = round(cv_scores.std(), 4)
            except Exception:
                result["cv_mean"] = None
                result["cv_std"] = None
        else:
            result["cv_mean"] = None
            result["cv_std"] = None

        return result

    except Exception as e:
        return {
            "model": None,
            "error": str(e),
            "train_score": None,
            "test_score": None,
            "time": round(time.time() - start, 2),
        }


# ═══════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT DE PLUSIEURS MODÈLES
# ═══════════════════════════════════════════════════════════════════
def train_multiple(model_names: list, X_train, y_train, X_test, y_test,
                   problem_type: str, model_params: dict = None,
                   cv_folds: int = 0, progress_callback=None) -> list:
    """Entraîne plusieurs modèles et retourne les résultats comparatifs.

    Args:
        model_names: Liste des noms de modèles à tester.
        X_train, y_train: Données d'entraînement.
        X_test, y_test: Données de test.
        problem_type: "Régression" ou "Classification".
        model_params: Dict {nom_modèle: {param: valeur}}.
        cv_folds: Folds de cross-validation (0 = pas de CV).
        progress_callback: Fonction callback(i, n, name) pour barre de progression.

    Returns:
        Liste de dicts avec les résultats de chaque modèle.
    """
    model_params = model_params or {}
    results = []

    for i, name in enumerate(model_names):
        if progress_callback:
            progress_callback(i, len(model_names), name)

        try:
            params = model_params.get(name, {})
            model = get_model(name, problem_type, params)
            result = train_model(model, X_train, y_train, X_test, y_test,
                                 problem_type, cv_folds)
            result["name"] = name
        except Exception as e:
            result = {
                "name": name,
                "error": str(e),
                "train_score": None,
                "test_score": None,
                "model": None,
            }

        results.append(result)

    # Trier par score test décroissant
    results.sort(
        key=lambda x: x.get("test_score") if x.get("test_score") is not None else -999,
        reverse=True,
    )

    return results


# ═══════════════════════════════════════════════════════════════════
# SPLIT TRAIN/TEST
# ═══════════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame, target_col: str, feature_cols: list,
               test_size: float = DEFAULT_TEST_SIZE,
               random_state: int = DEFAULT_RANDOM_STATE) -> tuple:
    """Sépare les données en jeux d'entraînement et de test.

    Args:
        df: DataFrame source.
        target_col: Variable cible.
        feature_cols: Variables explicatives.
        test_size: Proportion du jeu de test.
        random_state: Graine aléatoire.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    X = df[feature_cols].values
    y = df[target_col].values

    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state)


def split_data_chronological(df: pd.DataFrame, target_col: str,
                              feature_cols: list,
                              test_size: float = DEFAULT_TEST_SIZE,
                              datetime_col: str = None) -> tuple:
    """Sépare les données chronologiquement (pas de mélange).

    Les premières lignes servent à l'entraînement, les dernières au test.
    Le DataFrame doit être trié par date avant appel.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    if datetime_col and datetime_col in df.columns:
        df = df.sort_values(datetime_col).reset_index(drop=True)

    n = len(df)
    split_idx = int(n * (1 - test_size))

    X = df[feature_cols].values
    y = df[target_col].values

    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


# ═══════════════════════════════════════════════════════════════════
# OPTIMISATION DES HYPERPARAMÈTRES
# ═══════════════════════════════════════════════════════════════════
def optimize_model(model, X_train, y_train, param_grid: dict,
                   method: str = "grid", n_iter: int = 50,
                   cv: int = DEFAULT_CV_FOLDS,
                   problem_type: str = "Régression") -> dict:
    """Optimise les hyperparamètres d'un modèle.

    Args:
        model: Modèle scikit-learn.
        X_train, y_train: Données d'entraînement.
        param_grid: Grille de paramètres.
        method: "grid" (GridSearchCV) ou "random" (RandomizedSearchCV).
        n_iter: Nombre d'itérations pour Random Search.
        cv: Nombre de folds de cross-validation.
        problem_type: "Régression" ou "Classification".

    Returns:
        Dict avec le meilleur modèle, ses paramètres et les scores.
    """
    scoring = "r2" if problem_type == "Régression" else "accuracy"

    if method == "grid":
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=True,
        )
    else:
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=-1, random_state=DEFAULT_RANDOM_STATE,
            return_train_score=True,
        )

    search.fit(X_train, y_train)

    # Tableau des résultats
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score")

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": round(search.best_score_, 4),
        "cv_results": cv_results,
    }


# ═══════════════════════════════════════════════════════════════════
# SAUVEGARDE / CHARGEMENT
# ═══════════════════════════════════════════════════════════════════
def save_model(model, filepath: str):
    """Sauvegarde un modèle entraîné au format pickle.

    Args:
        model: Modèle scikit-learn.
        filepath: Chemin du fichier .pkl.
    """
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Charge un modèle depuis un fichier pickle.

    Args:
        filepath: Chemin du fichier .pkl.

    Returns:
        Modèle scikit-learn chargé.
    """
    return joblib.load(filepath)
