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

# ── Palette de couleurs ──
_P = {
    "primary": "#4F5BD5",
    "secondary": "#818CF8",
    "accent": "#A5B4FC",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "bg": "#F8F9FC",
    "surface": "#FFFFFF",
    "grid": "#F3F4F6",
    "border": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
    "palette": ["#4F5BD5", "#059669", "#D97706", "#DC2626", "#0891B2",
                "#7C3AED", "#EA580C", "#2563EB", "#14B8A6", "#E11D48"],
}

def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Style commun applique a un axe."""
    ax.set_facecolor(_P["surface"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_P["border"])
    ax.spines["bottom"].set_color(_P["border"])
    ax.tick_params(colors=_P["muted"], labelsize=7)
    ax.grid(True, alpha=0.4, color=_P["grid"], linestyle="-")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=_P["text"], pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=_P["muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=_P["muted"])

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report as sk_classification_report,
    precision_recall_curve,
    average_precision_score,
)


# ═══════════════════════════════════════════════════════════════════
# MASE — Mean Absolute Scaled Error
# ═══════════════════════════════════════════════════════════════════

def compute_mase(y_true, y_pred, y_train) -> float:
    """Calcule le MASE (Mean Absolute Scaled Error).

    MASE = MAE(y_true, y_pred) / MAE_naïf
    où MAE_naïf = erreur moyenne du modèle naïf (prédire la valeur précédente)
    calculée sur l'ensemble d'entraînement.

    Retourne np.inf si le dénominateur est nul (série constante).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_train[1:] - y_train[:-1]))

    if mae_naive == 0:
        return np.inf
    return mae_model / mae_naive


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
    fig, ax = plt.subplots(figsize=(3, 2.2))
    fig.set_facecolor(_P["bg"])
    ax.scatter(y_true, y_pred, alpha=0.6, color=_P["primary"],
               edgecolors="white", linewidth=0.5, s=40, zorder=3)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color=_P["danger"],
            linestyle="--", linewidth=2, label="Ideal", alpha=0.8, zorder=2)

    _style_ax(ax, title, "Valeurs reelles", "Valeurs predites")
    ax.legend(fontsize=7, framealpha=0.9, edgecolor=_P["border"])
    plt.tight_layout()
    return fig


def plot_real_vs_pred_interactive(y_true, y_pred, title: str = "Réel vs Prédit"):
    """Scatter Plotly interactif des valeurs réelles vs prédites (hover, zoom)."""
    import plotly.graph_objects as go
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=_P["primary"], size=6, opacity=0.6,
                    line=dict(width=0.5, color="white")),
        name="Points",
    ))
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color=_P["danger"], dash="dash", width=2),
        name="Idéal",
    ))
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites",
        height=400,
    )
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

    fig, ax = plt.subplots(figsize=(3, 2.2))
    fig.set_facecolor(_P["bg"])
    ax.scatter(y_pred, residuals, alpha=0.6, color=_P["primary"],
               edgecolors="white", linewidth=0.5, s=40, zorder=3)
    ax.axhline(y=0, color=_P["danger"], linestyle="--", linewidth=2, alpha=0.8, zorder=2)
    _style_ax(ax, title, "Valeurs predites", "Residus")
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

    fig, ax = plt.subplots(figsize=(3, 2.2))
    fig.set_facecolor(_P["bg"])
    sns.histplot(residuals, kde=True, ax=ax, color=_P["primary"],
                 edgecolor="white", alpha=0.7, line_kws={"linewidth": 2.5})
    ax.axvline(x=0, color=_P["danger"], linestyle="--", linewidth=2, alpha=0.8)
    _style_ax(ax, title, "Residus", "Frequence")
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
        comments.append("⚠️ Distribution asymétrique vers la droite — transformation log recommandée. "
                        "→ Étape 4, onglet « 🔧 Modifier les colonnes » → Transformer en place → Logarithme.")
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
                          title: str = "Matrice de confusion",
                          normalize: bool = False):
    """Heatmap de la matrice de confusion.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        labels: Labels des classes.
        title: Titre du graphique.
        normalize: Si True, affiche les proportions (0-1) au lieu des comptages.

    Returns:
        Figure matplotlib.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fmt = "d"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # éviter division par zéro
        cm = cm.astype(float) / row_sums
        fmt = ".2f"
        title = title + " (normalisée)"
    n_classes = len(cm)
    fig_size = max(3, n_classes * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    fig.set_facecolor(_P["bg"])
    cmap = sns.light_palette(_P["primary"], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor=_P["bg"],
                annot_kws={"fontsize": 8, "fontweight": "bold"})
    _style_ax(ax, title, "Predit", "Reel")
    ax.tick_params(labelsize=7)
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

    fig, ax = plt.subplots(figsize=(3, 2.2))
    fig.set_facecolor(_P["bg"])
    ax.fill_between(fpr, tpr, alpha=0.15, color=_P["primary"])
    ax.plot(fpr, tpr, color=_P["primary"], linewidth=2.5, label=f"AUC = {auc:.3f}", zorder=3)
    ax.plot([0, 1], [0, 1], color=_P["muted"], linestyle="--", linewidth=1.5,
            label="Aleatoire", alpha=0.7, zorder=2)
    _style_ax(ax, title, "Taux de faux positifs", "Taux de vrais positifs")
    ax.legend(fontsize=7, framealpha=0.9, edgecolor=_P["border"], loc="lower right")
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
# CLASSIFICATION — Graphiques avancés
# ═══════════════════════════════════════════════════════════════════

def plot_precision_recall_curve(y_true, y_proba,
                                 title: str = "Courbe Precision-Recall"):
    """Courbe Precision-Recall avec AP score (binaire)."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(3, 2.2))
    fig.set_facecolor(_P["bg"])
    ax.fill_between(recall, precision, alpha=0.15, color=_P["success"])
    ax.plot(recall, precision, color=_P["success"], linewidth=2.5,
            label=f"AP = {ap:.3f}", zorder=3)
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color=_P["muted"], linestyle="--", linewidth=1.5,
               label=f"Aleatoire ({baseline:.2f})", alpha=0.7)
    _style_ax(ax, title, "Recall", "Precision")
    ax.legend(fontsize=7, framealpha=0.9, edgecolor=_P["border"], loc="lower left")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    return fig


def plot_confusion_matrix_detailed(y_true, y_pred, labels=None,
                                    title: str = "Matrice de confusion détaillée"):
    """Matrice de confusion avec TP, TN, FP, FN pour chaque classe."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(cm)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, n * 0.8)),
                             gridspec_kw={"width_ratios": [3, 2]})
    fig.set_facecolor(_P["bg"])

    # Heatmap
    ax = axes[0]
    cmap = sns.light_palette(_P["primary"], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor=_P["bg"],
                annot_kws={"fontsize": 8, "fontweight": "bold"})
    _style_ax(ax, title, "Predit", "Reel")
    ax.tick_params(labelsize=7)

    # Métriques par classe
    ax2 = axes[1]
    ax2.axis("off")
    total = cm.sum()
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append([str(label), tp, fp, fn, tn,
                     f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"])

    col_labels = ["Classe", "VP", "FP", "FN", "VN", "Precision", "Recall", "F1"]
    table = ax2.table(cellText=rows, colLabels=col_labels,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(_P["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(_P["surface"])
        cell.set_edgecolor(_P["border"])

    plt.tight_layout()
    return fig


def plot_classification_metrics_bar(y_true, y_pred,
                                     title: str = "Métriques par classe"):
    """Barplot horizontal de precision, recall, F1 par classe."""
    report = sk_classification_report(y_true, y_pred, output_dict=True,
                                       zero_division=0)
    classes = [k for k in report if k not in
               ("accuracy", "macro avg", "weighted avg")]
    metrics = {"Precision": [], "Recall": [], "F1-Score": []}
    for cls in classes:
        metrics["Precision"].append(report[cls]["precision"])
        metrics["Recall"].append(report[cls]["recall"])
        metrics["F1-Score"].append(report[cls]["f1-score"])

    fig, ax = plt.subplots(figsize=(6, max(2, len(classes) * 0.5)))
    fig.set_facecolor(_P["bg"])
    y_pos = np.arange(len(classes))
    bar_h = 0.25
    colors = [_P["primary"], _P["success"], _P["warning"]]

    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.barh(y_pos + i * bar_h, values, bar_h, label=metric_name,
                color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos + bar_h)
    ax.set_yticklabels(classes)
    ax.set_xlim([0, 1.05])
    _style_ax(ax, title, "Score", "")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# RAPPORT DE RÉGRESSION (formule, coefficients)
# ═══════════════════════════════════════════════════════════════════
def get_regression_report(model, feature_names: list, y_true, y_pred,
                          scaler=None, scaled_columns=None) -> dict:
    """Génère un rapport détaillé pour un modèle de régression.

    Args:
        scaler: Le scaler utilisé à l'entraînement (StandardScaler ou MinMaxScaler).
        scaled_columns: Liste des colonnes qui ont été scalées.

    Returns:
        Dict avec formula, raw_formula, coefficients, metrics, model_type.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Détecter si c'est un Pipeline (ex: Polynomiale)
    inner_model = model
    poly_features = None
    model_type_name = type(model).__name__
    if isinstance(model, Pipeline):
        model_type_name = "Pipeline Polynomiale"
        for step_name, step in model.named_steps.items():
            if hasattr(step, "coef_"):
                inner_model = step
            if hasattr(step, "get_feature_names_out"):
                poly_features = step

    report = {
        "model_type": model_type_name,
        "has_formula": False,
        "formula": "",
        "raw_formula": "",
        "coefficients": None,
        "raw_coefficients": None,
        "intercept": None,
        "raw_intercept": None,
        "is_scaled": bool(scaler and scaled_columns),
        "metrics": {
            "r2": round(r2_score(y_true, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
        },
    }

    # Extraire les coefficients si le modèle est linéaire
    if hasattr(inner_model, "coef_"):
        coefs = inner_model.coef_.flatten()
        intercept = inner_model.intercept_ if hasattr(inner_model, "intercept_") else 0

        # Noms des features (polynomiales ou originales)
        if poly_features is not None:
            try:
                names = poly_features.get_feature_names_out(feature_names).tolist()
            except Exception:
                names = [f"x{i}" for i in range(len(coefs))]
        else:
            names = feature_names

        # Ajuster si tailles différentes
        if len(names) != len(coefs):
            names = [f"x{i}" for i in range(len(coefs))]

        # Construire le tableau des coefficients (sur données scalées)
        coef_df = pd.DataFrame({
            "Variable": names,
            "Coefficient": coefs,
            "Abs. Coefficient": np.abs(coefs),
        }).sort_values("Abs. Coefficient", ascending=False)

        report["has_formula"] = True
        report["coefficients"] = coef_df
        report["intercept"] = float(intercept)

        # Formule sur données scalées
        def _build_formula(names_list, coefs_list, intercept_val):
            terms = []
            for name, c in sorted(zip(names_list, coefs_list),
                                   key=lambda x: abs(x[1]), reverse=True):
                if abs(c) < 1e-10:
                    continue
                sign = "+" if c >= 0 else "-"
                terms.append(f"{sign} {abs(c):.4f} x {name}")
            return f"y = {intercept_val:.4f} " + " ".join(terms)

        report["formula"] = _build_formula(names, coefs, intercept)

        # Calculer la formule sur données brutes si un scaler a été utilisé
        # Pour un modèle linéaire : y = b0 + b1*x1_scaled + b2*x2_scaled
        # Si x_scaled = (x - mean) / std  (StandardScaler) :
        #   y = b0 + b1*(x1-m1)/s1 + b2*(x2-m2)/s2
        #   y = (b0 - b1*m1/s1 - b2*m2/s2) + (b1/s1)*x1 + (b2/s2)*x2
        # Si x_scaled = (x - min) / (max - min)  (MinMaxScaler) :
        #   y = (b0 - b1*min1/range1 ...) + (b1/range1)*x1 + ...
        if (scaler is not None and scaled_columns
                and poly_features is None
                and len(names) == len(feature_names)):
            try:
                raw_coefs = coefs.copy()
                raw_intercept = float(intercept)

                if isinstance(scaler, StandardScaler):
                    means = scaler.mean_
                    scales = scaler.scale_
                elif isinstance(scaler, MinMaxScaler):
                    means = scaler.data_min_
                    scales = scaler.data_range_
                else:
                    means = None

                if means is not None and len(means) == len(scaled_columns):
                    # Construire un mapping colonne → index dans le scaler
                    scale_map = {col: i for i, col in enumerate(scaled_columns)}

                    for j, name in enumerate(feature_names):
                        if name in scale_map:
                            si = scale_map[name]
                            raw_intercept -= coefs[j] * means[si] / scales[si]
                            raw_coefs[j] = coefs[j] / scales[si]

                    raw_coef_df = pd.DataFrame({
                        "Variable": feature_names,
                        "Coefficient": raw_coefs,
                        "Abs. Coefficient": np.abs(raw_coefs),
                    }).sort_values("Abs. Coefficient", ascending=False)

                    report["raw_coefficients"] = raw_coef_df
                    report["raw_intercept"] = raw_intercept
                    report["raw_formula"] = _build_formula(
                        feature_names, raw_coefs, raw_intercept)
            except Exception:
                pass  # Si le calcul échoue, on n'affiche pas la formule brute

    # Pour les modèles à base d'arbres : pas de formule mais importance
    elif hasattr(inner_model, "feature_importances_"):
        imp = inner_model.feature_importances_
        if len(imp) == len(feature_names):
            coef_df = pd.DataFrame({
                "Variable": feature_names,
                "Importance": imp,
            }).sort_values("Importance", ascending=False)
            report["coefficients"] = coef_df

    return report


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
    n_bars = len(idx)
    fig, ax = plt.subplots(figsize=(5, max(1.5, n_bars * 0.22)))
    fig.set_facecolor(_P["bg"])
    bars = ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color=_P["primary"],
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )
    for bar, val in zip(bars, importances[idx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=_P["muted"])
    _style_ax(ax, title, "Importance", "")
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
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_facecolor(_P["bg"])
    df[col].dropna().hist(bins=bins, ax=ax, color=_P["primary"],
                          edgecolor="white", alpha=0.85, linewidth=0.5)
    _style_ax(ax, title or f'Distribution de "{col}"', col, "Frequence")
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
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_facecolor(_P["bg"])
    bp_props = dict(
        boxprops=dict(facecolor=_P["accent"], edgecolor=_P["primary"], linewidth=1.5),
        whiskerprops=dict(color=_P["primary"], linewidth=1.2),
        capprops=dict(color=_P["primary"], linewidth=1.2),
        medianprops=dict(color=_P["danger"], linewidth=2),
        flierprops=dict(marker="o", markerfacecolor=_P["warning"],
                        markeredgecolor="white", markersize=6, alpha=0.7),
    )
    if group_col:
        df.boxplot(column=col, by=group_col, ax=ax, patch_artist=True, **bp_props)
        plt.suptitle("")
        _style_ax(ax, title or f'Boxplot de "{col}" par "{group_col}"', group_col, col)
    else:
        df[[col]].boxplot(ax=ax, patch_artist=True, **bp_props)
        _style_ax(ax, title or f'Boxplot de "{col}"', "", col)
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
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_facecolor(_P["bg"])
    if color_col and color_col in df.columns:
        groups = df[color_col].unique()
        for i, g in enumerate(groups):
            mask = df[color_col] == g
            c = _P["palette"][i % len(_P["palette"])]
            ax.scatter(df.loc[mask, col_x], df.loc[mask, col_y],
                       alpha=0.65, label=str(g), color=c,
                       edgecolors="white", linewidth=0.5, s=40, zorder=3)
        ax.legend(title=color_col, fontsize=9, framealpha=0.9, edgecolor=_P["border"])
    else:
        ax.scatter(df[col_x], df[col_y], alpha=0.6, color=_P["primary"],
                   edgecolors="white", linewidth=0.5, s=40, zorder=3)

    _style_ax(ax, title or f'"{col_x}" vs "{col_y}"', col_x, col_y)
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
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_facecolor(_P["bg"])
    sns.histplot(series.dropna(), kde=True, ax=ax, color=_P["primary"],
                 edgecolor="white", alpha=0.75, line_kws={"linewidth": 2.5})
    _style_ax(ax, title, series.name or "Cible", "Frequence")
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
        return ("⚠️ Distribution fortement asymétrique vers la droite — transformation log recommandée. "
                "Allez à l'étape 4, onglet « 🔧 Modifier les colonnes » → « Transformer une colonne en place » → Logarithme.")
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
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
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
