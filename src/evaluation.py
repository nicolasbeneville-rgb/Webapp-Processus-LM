# -*- coding: utf-8 -*-
"""
evaluation.py — Métriques, diagnostics et graphiques d'évaluation.

Fonctions principales :
    - results_table          : tableau comparatif des modèles
    - plot_real_vs_pred      : scatter réel vs prédit
    - plot_residuals         : graphique des résidus
    - plot_residual_dist     : distribution des résidus
    - plot_confusion_matrix  : matrice de confusion
    - plot_roc_curve         : courbe ROC
    - plot_feature_importance: importance des variables
    - classification_report  : rapport de classification
    - generate_html_report   : export rapport HTML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report as sk_classification_report,
)


# ═══════════════════════════════════════════════════════════════════
# TABLEAU COMPARATIF
# ═══════════════════════════════════════════════════════════════════
def results_table(results: list, problem_type: str) -> pd.DataFrame:
    """Construit le tableau comparatif des résultats de modélisation.

    Args:
        results: Liste de dicts retournés par train_multiple().
        problem_type: "Régression" ou "Classification".

    Returns:
        DataFrame formaté pour affichage.
    """
    rows = []
    for r in results:
        if r.get("error"):
            rows.append({
                "Modèle": r["name"],
                "Statut": "❌ Erreur",
                "Erreur": r["error"],
            })
            continue

        row = {"Modèle": r["name"]}

        if problem_type == "Régression":
            row["R² train"] = r.get("train_score")
            row["R² test"] = r.get("test_score")
            row["RMSE"] = r.get("rmse")
            row["MAE"] = r.get("mae")
        else:
            row["Acc. train"] = r.get("train_score")
            row["Acc. test"] = r.get("test_score")
            row["F1-Score"] = r.get("f1")
            row["AUC-ROC"] = r.get("auc")

        row["Écart train/test"] = f"{r.get('overfit_pct', 0)}%"
        row["Temps (s)"] = r.get("time")

        if r.get("cv_mean") is not None:
            row["CV moyen"] = r["cv_mean"]
            row["CV écart-type"] = r["cv_std"]

        # Statut de validation
        overfit = r.get("overfit_pct", 0)
        test_score = r.get("test_score", 0)
        if test_score and test_score >= 0.6 and overfit <= 10:
            row["Statut"] = "✅"
        elif test_score and test_score >= 0.5:
            row["Statut"] = "⚠️"
        else:
            row["Statut"] = "❌"

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# GRAPHIQUES — RÉGRESSION
# ═══════════════════════════════════════════════════════════════════
def plot_real_vs_pred(y_true, y_pred, title: str = "Réel vs Prédit"):
    """Scatter plot des valeurs réelles vs prédites.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        title: Titre du graphique.

    Returns:
        Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)

    # Droite idéale
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Idéal")

    ax.set_xlabel("Valeurs réelles", fontsize=12)
    ax.set_ylabel("Valeurs prédites", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title: str = "Résidus vs Valeurs prédites"):
    """Graphique des résidus en fonction des valeurs prédites.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        title: Titre du graphique.

    Returns:
        Figure matplotlib.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidth=0.5)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Valeurs prédites", fontsize=12)
    ax.set_ylabel("Résidus", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_residual_distribution(y_true, y_pred,
                               title: str = "Distribution des résidus"):
    """Histogramme + courbe de densité des résidus.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        title: Titre du graphique.

    Returns:
        Figure matplotlib.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax, color="steelblue")
    ax.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Résidus", fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def get_top_errors(y_true, y_pred, n: int = 10, indices=None) -> pd.DataFrame:
    """Identifie les N plus grandes erreurs.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        n: Nombre d'erreurs à retourner.
        indices: Index originaux (optionnel).

    Returns:
        DataFrame avec les plus grandes erreurs.
    """
    residuals = np.abs(np.array(y_true) - np.array(y_pred))
    top_idx = np.argsort(residuals)[-n:][::-1]

    records = []
    for idx in top_idx:
        records.append({
            "Index": indices[idx] if indices is not None else idx,
            "Réel": round(float(y_true[idx]), 4),
            "Prédit": round(float(y_pred[idx]), 4),
            "Erreur absolue": round(float(residuals[idx]), 4),
        })
    return pd.DataFrame(records)


def auto_comment_residuals(y_true, y_pred) -> str:
    """Génère un commentaire automatique sur la qualité des résidus.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.

    Returns:
        Commentaire textuel.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)

    from scipy.stats import skew, kurtosis

    skewness = skew(residuals)
    kurt = kurtosis(residuals)

    comments = []
    if abs(mean_r) < 0.01 * std_r:
        comments.append("✅ Les résidus sont centrés autour de zéro.")
    else:
        direction = "surestimation" if mean_r < 0 else "sous-estimation"
        comments.append(f"⚠️ Biais systématique détecté ({direction}).")

    if abs(skewness) < 0.5:
        comments.append("✅ Distribution symétrique des résidus.")
    elif skewness > 0:
        comments.append("⚠️ Distribution asymétrique vers la droite — transformation log recommandée.")
    else:
        comments.append("⚠️ Distribution asymétrique vers la gauche.")

    if abs(kurt) < 3:
        comments.append("✅ Queue de distribution normale.")
    else:
        comments.append("⚠️ Queues lourdes — présence possible de valeurs extrêmes.")

    return "\n".join(comments)


# ═══════════════════════════════════════════════════════════════════
# GRAPHIQUES — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, labels=None,
                          title: str = "Matrice de confusion"):
    """Heatmap de la matrice de confusion.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        labels: Labels des classes.
        title: Titre du graphique.

    Returns:
        Figure matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title: str = "Courbe ROC"):
    """Courbe ROC avec AUC.

    Args:
        y_true: Valeurs réelles (binaires).
        y_proba: Probabilités de la classe positive.
        title: Titre du graphique.

    Returns:
        Figure matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="blue", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig


def get_classification_report(y_true, y_pred) -> pd.DataFrame:
    """Rapport de classification sous forme de DataFrame.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.

    Returns:
        DataFrame du rapport.
    """
    report = sk_classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.round(3)


def get_misclassified(y_true, y_pred, X=None, n: int = 10) -> pd.DataFrame:
    """Identifie les cas mal classés.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        X: Features (optionnel).
        n: Nombre de cas à retourner.

    Returns:
        DataFrame des cas mal classés.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != y_pred
    indices = np.where(mask)[0][:n]

    records = []
    for idx in indices:
        row = {"Index": idx, "Réel": y_true[idx], "Prédit": y_pred[idx]}
        records.append(row)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════
# IMPORTANCE DES VARIABLES
# ═══════════════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_names: list,
                            title: str = "Importance des variables", n: int = 20):
    """Graphique de l'importance des variables (si disponible).

    Args:
        model: Modèle scikit-learn entraîné.
        feature_names: Noms des features.
        title: Titre du graphique.
        n: Nombre de variables à afficher.

    Returns:
        Figure matplotlib ou None.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        return None

    if len(importances) != len(feature_names):
        return None

    idx = np.argsort(importances)[-n:]
    fig, ax = plt.subplots(figsize=(10, max(4, len(idx) * 0.35)))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# GRAPHIQUES EXPLORATOIRES (Étape 6)
# ═══════════════════════════════════════════════════════════════════
def plot_histogram(df: pd.DataFrame, col: str, bins: int = 30,
                   title: str = None):
    """Histogramme d'une variable.

    Args:
        df: DataFrame source.
        col: Colonne à afficher.
        bins: Nombre de bins.
        title: Titre (auto-généré si None).

    Returns:
        Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df[col].dropna().hist(bins=bins, ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)
    ax.set_title(title or f"Distribution de « {col} »", fontsize=14)
    plt.tight_layout()
    return fig


def plot_boxplot(df: pd.DataFrame, col: str, group_col: str = None,
                 title: str = None):
    """Boxplot d'une variable, optionnellement groupé.

    Args:
        df: DataFrame source.
        col: Variable numérique.
        group_col: Variable de regroupement (optionnel).
        title: Titre.

    Returns:
        Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if group_col:
        df.boxplot(column=col, by=group_col, ax=ax)
        ax.set_title(title or f"Boxplot de « {col} » par « {group_col} »", fontsize=14)
        plt.suptitle("")  # Supprimer le titre auto
    else:
        df[[col]].boxplot(ax=ax)
        ax.set_title(title or f"Boxplot de « {col} »", fontsize=14)
    plt.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, col_x: str, col_y: str,
                 color_col: str = None, title: str = None):
    """Scatter plot de deux variables.

    Args:
        df: DataFrame source.
        col_x: Variable X.
        col_y: Variable Y.
        color_col: Variable de couleur (optionnel).
        title: Titre.

    Returns:
        Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col and color_col in df.columns:
        groups = df[color_col].unique()
        for g in groups:
            mask = df[color_col] == g
            ax.scatter(df.loc[mask, col_x], df.loc[mask, col_y],
                       alpha=0.6, label=str(g), edgecolors="k", linewidth=0.3)
        ax.legend(title=color_col)
    else:
        ax.scatter(df[col_x], df[col_y], alpha=0.5, edgecolors="k", linewidth=0.3)

    ax.set_xlabel(col_x, fontsize=12)
    ax.set_ylabel(col_y, fontsize=12)
    ax.set_title(title or f"« {col_x} » vs « {col_y} »", fontsize=14)
    plt.tight_layout()
    return fig


def plot_target_distribution(series: pd.Series,
                             title: str = "Distribution de la variable cible"):
    """Histogramme + densité de la variable cible.

    Args:
        series: Variable cible.
        title: Titre.

    Returns:
        Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(series.dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_xlabel(series.name or "Cible", fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def auto_comment_distribution(series: pd.Series) -> str:
    """Génère un commentaire sur la distribution d'une variable.

    Args:
        series: Série numérique.

    Returns:
        Commentaire textuel.
    """
    from scipy.stats import skew

    s = series.dropna()
    if len(s) == 0:
        return "Pas de données."

    skewness = skew(s)
    if abs(skewness) < 0.5:
        return "✅ Distribution relativement symétrique."
    elif skewness > 1:
        return "⚠️ Distribution fortement asymétrique vers la droite — transformation log recommandée."
    elif skewness > 0.5:
        return "ℹ️ Distribution légèrement asymétrique vers la droite."
    elif skewness < -1:
        return "⚠️ Distribution fortement asymétrique vers la gauche."
    else:
        return "ℹ️ Distribution légèrement asymétrique vers la gauche."


# ═══════════════════════════════════════════════════════════════════
# EXPORT DU GRAPHIQUE EN PNG (pour téléchargement)
# ═══════════════════════════════════════════════════════════════════
def fig_to_png_bytes(fig) -> bytes:
    """Convertit une figure matplotlib en bytes PNG.

    Args:
        fig: Figure matplotlib.

    Returns:
        Bytes PNG.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# RAPPORT HTML
# ═══════════════════════════════════════════════════════════════════
def generate_html_report(project_config: dict, data_summary: dict,
                         model_summary: dict, validation_summary: list,
                         figures: dict = None) -> str:
    """Génère un rapport HTML complet du projet.

    Args:
        project_config: Configuration du projet.
        data_summary: Résumé des données.
        model_summary: Résumé du meilleur modèle.
        validation_summary: Liste des validations.
        figures: Dict {nom: bytes_png} des graphiques.

    Returns:
        Chaîne HTML.
    """
    figures = figures or {}

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport ML — {project_config.get('name', 'Projet')}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .ok {{ color: green; font-weight: bold; }}
        .warn {{ color: orange; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .metric {{ display: inline-block; background: white; border: 1px solid #ddd;
                   border-radius: 8px; padding: 15px 25px; margin: 5px; text-align: center; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric .label {{ font-size: 12px; color: #7f8c8d; }}
        img {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>🤖 Rapport ML — {project_config.get('name', 'Projet')}</h1>

    <h2>📋 Configuration</h2>
    <table>
        <tr><th>Paramètre</th><th>Valeur</th></tr>
        <tr><td>Type de problème</td><td>{project_config.get('problem_type', '-')}</td></tr>
        <tr><td>Variable cible</td><td>{project_config.get('target', '-')}</td></tr>
    </table>

    <h2>📊 Données</h2>
    <div>
        <div class="metric"><div class="value">{data_summary.get('n_rows', '-')}</div><div class="label">Lignes</div></div>
        <div class="metric"><div class="value">{data_summary.get('n_cols', '-')}</div><div class="label">Colonnes</div></div>
        <div class="metric"><div class="value">{data_summary.get('n_features', '-')}</div><div class="label">Features</div></div>
    </div>

    <h2>🏆 Meilleur modèle</h2>
    <table>
        <tr><th>Propriété</th><th>Valeur</th></tr>
        <tr><td>Modèle</td><td>{model_summary.get('name', '-')}</td></tr>
        <tr><td>Score test</td><td>{model_summary.get('test_score', '-')}</td></tr>
        <tr><td>Score train</td><td>{model_summary.get('train_score', '-')}</td></tr>
        <tr><td>Écart train/test</td><td>{model_summary.get('overfit_pct', '-')}%</td></tr>
    </table>

    <h2>✅ Validations</h2>
    <table>
        <tr><th>Étape</th><th>Statut</th><th>Message</th></tr>"""

    for v in validation_summary:
        css = "ok" if "✅" in v.get("Statut", "") else ("warn" if "⚠" in v.get("Statut", "") else "fail")
        html += f"""
        <tr>
            <td>{v.get('Étape', '')}</td>
            <td class="{css}">{v.get('Statut', '')}</td>
            <td>{v.get('Message', '')}</td>
        </tr>"""

    html += """
    </table>"""

    # Graphiques embarqués
    for name, png_bytes in figures.items():
        b64 = base64.b64encode(png_bytes).decode()
        html += f"""
    <h2>📈 {name}</h2>
    <img src="data:image/png;base64,{b64}" alt="{name}">"""

    html += """
    <hr>
    <p style="color:#7f8c8d; font-size:12px;">
        Rapport généré par Pipeline ML — Guide interactif
    </p>
</body>
</html>"""

    return html
