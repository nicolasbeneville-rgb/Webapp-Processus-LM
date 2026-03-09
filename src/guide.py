# -*- coding: utf-8 -*-
"""
guide.py — Contenus pédagogiques, recommandations et vocabulaire vulgarisé.

Ce module fournit :
    - Des explications simples pour chaque concept
    - Des recommandations automatiques basées sur les données
    - Du vocabulaire accessible aux non-spécialistes
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════
# EXPLICATIONS PAR ÉTAPE (bandeau guide)
# ═══════════════════════════════════════════════════════════
STEP_GUIDES = {
    0: {
        "title": "🚀 Démarrez votre projet",
        "what": "Vous allez configurer votre projet (nom, objectif, qualité) puis charger vos fichiers de données (CSV ou Excel).",
        "why": "Ces choix orientent toute la suite. Prédire un **nombre** (prix, durée…) ou une **catégorie** (oui/non, type…) n'utilise pas les mêmes outils. Les données sont la matière première.",
        "tip": "Si vous avez déjà travaillé sur ce projet, vous pouvez le reprendre directement.",
    },
    1: {
        "title": "🔤 Vérifiez le type de chaque colonne",
        "what": "L'ordinateur doit savoir si chaque colonne contient des **nombres**, du **texte**, des **dates** ou des **catégories**.",
        "why": "Un mauvais type empêche les calculs. Par exemple, si des prix sont lus comme du texte (\"1 200 €\"), il faut les convertir en nombres.",
        "tip": "L'application détecte automatiquement les types. Vérifiez simplement que les suggestions sont correctes.",
    },
    2: {
        "title": "🔗 Combinez vos fichiers",
        "what": "Si vous avez chargé plusieurs fichiers, vous pouvez les fusionner en un seul tableau grâce à une **colonne commune** (comme un identifiant).",
        "why": "Travailler sur un seul tableau complet est nécessaire pour analyser les relations entre toutes vos données.",
        "tip": "Choisissez une colonne qui fait le lien entre vos fichiers (ex : numéro client, code produit…).",
    },
    3: {
        "title": "🔍 Inspectez la qualité de vos données",
        "what": "L'application analyse automatiquement vos données pour détecter les **trous** (valeurs manquantes), les **doublons**, les **valeurs aberrantes** et les **colonnes inutiles**.",
        "why": "Des données de mauvaise qualité donnent des résultats peu fiables. Mieux vaut détecter les problèmes maintenant.",
        "tip": "Regardez les indicateurs en rouge et orange : ce sont les points à traiter en priorité.",
    },
    4: {
        "title": "🧹 Nettoyez et préparez vos données",
        "what": "Vous allez traiter les problèmes détectés : boucher les trous, gérer les valeurs extrêmes, et mettre les données dans un format exploitable.",
        "why": "Un modèle prédictif est comme une recette : si les ingrédients sont mal préparés, le résultat sera mauvais.",
        "tip": "L'application vous recommande la meilleure action pour chaque problème. Suivez les suggestions en vert.",
    },
    5: {
        "title": "📊 Explorez visuellement vos données",
        "what": "Créez des graphiques pour mieux comprendre vos données : répartition, relations entre colonnes, tendances…",
        "why": "Un graphique révèle souvent ce que les chiffres seuls ne montrent pas : des groupes, des tendances, des anomalies.",
        "tip": "Commencez par regarder la répartition de ce que vous voulez prédire (la cible).",
    },
    6: {
        "title": "🤖 Testez plusieurs modèles prédictifs",
        "what": "L'application va entraîner plusieurs **algorithmes** sur vos données et les comparer pour trouver le plus performant.",
        "why": "Chaque algorithme a ses forces. En les testant tous, on trouve celui qui s'adapte le mieux à VOS données.",
        "tip": "Cochez au moins 3 modèles pour avoir une bonne comparaison. Le meilleur sera mis en évidence.",
    },
    7: {
        "title": "🔧 Affinez le meilleur modèle",
        "what": "Chaque modèle a des **réglages** (comme les boutons d'un appareil photo). On va tester plusieurs combinaisons pour trouver les meilleurs.",
        "why": "Un bon modèle avec de mauvais réglages peut sous-performer. L'optimisation peut gagner quelques points de précision.",
        "tip": "Laissez les valeurs par défaut si vous n'êtes pas sûr. L'application teste automatiquement les combinaisons.",
    },
    8: {
        "title": "🔬 Itérer sur les variables",
        "what": "Après avoir testé un modèle, revenez à l'**étape 4** (onglet « 🔧 Modifier les colonnes ») pour créer/modifier des variables, puis re-testez ici.",
        "why": "L'amélioration d'un modèle est un processus itératif : modifier les données → re-tester → comparer les scores.",
        "tip": "Créez des variables pertinentes à l'étape 4, puis cliquez sur « Re-tester » ici pour voir l'impact.",
    },
    9: {
        "title": "📉 Analysez les erreurs du modèle",
        "what": "On regarde **où et comment** le modèle se trompe pour comprendre ses limites.",
        "why": "Comprendre les erreurs permet de savoir si on peut faire confiance au modèle, et dans quels cas il est fiable.",
        "tip": "Si les erreurs suivent un pattern (le modèle se trompe toujours dans le même sens), c'est un signe qu'on peut encore améliorer.",
    },
    10: {
        "title": "📋 Bilan final et export",
        "what": "Récapitulatif de tout votre projet : données, modèle choisi, performances. Vous pouvez tout télécharger.",
        "why": "Ce rapport vous permet de documenter votre travail et de partager vos résultats.",
        "tip": "Téléchargez le modèle (.pkl) si vous voulez l'utiliser plus tard sur de nouvelles données.",
    },
    11: {
        "title": "🔮 Utilisez votre modèle",
        "what": "Chargez de **nouvelles données** (sans la colonne à prédire) et le modèle génère automatiquement les prédictions.",
        "why": "C'est l'objectif final : utiliser ce qu'on a construit pour prédire l'inconnu !",
        "tip": "Le fichier doit avoir exactement les mêmes colonnes que celles utilisées pour l'entraînement.",
    },
}


# ═══════════════════════════════════════════════════════════
# VOCABULAIRE VULGARISÉ
# ═══════════════════════════════════════════════════════════
GLOSSARY = {
    "R²": "**Score de précision** — Plus il est proche de 1, mieux le modèle prédit. "
          "À 0.8, il explique 80% des variations dans vos données.",
    "RMSE": "**Erreur moyenne** — La différence moyenne entre ce que le modèle prédit "
            "et la réalité. Plus c'est petit, mieux c'est.",
    "MAE": "**Écart moyen** — En moyenne, de combien le modèle se trompe. "
           "Plus facile à interpréter que le RMSE.",
    "Accuracy": "**Taux de bonnes réponses** — Sur 100 prédictions, combien sont justes. "
                "0.85 = 85% de bonnes réponses.",
    "F1-Score": "**Score d'équilibre** — Combine la capacité à trouver tous les cas positifs "
                "ET à ne pas se tromper. Utile quand les classes sont déséquilibrées.",
    "AUC-ROC": "**Capacité de discrimination** — Mesure si le modèle sait bien différencier "
               "les catégories. 1.0 = parfait, 0.5 = au hasard.",
    "Overfitting": "**Sur-apprentissage** — Le modèle a \"appris par cœur\" les données "
                   "d'entraînement au lieu de comprendre les règles générales. "
                   "Résultat : il est bon en entraînement mais mauvais sur de nouvelles données.",
    "Cross-validation": "**Validation croisée** — On découpe les données en plusieurs morceaux "
                        "et on teste le modèle sur chacun. Ça donne un score plus fiable.",
    "Normalisation": "**Mise à l'échelle** — Transformer les valeurs pour qu'elles soient "
                     "sur une échelle comparable (ex : âge 0-100 et salaire 0-100 000 → tous entre 0 et 1).",
    "Encodage": "**Traduction en chiffres** — Les modèles ne comprennent que les nombres. "
                "Il faut transformer le texte (ex : \"homme\"/\"femme\") en chiffres (0/1).",
    "One-Hot": "**Encodage par colonnes** — Chaque catégorie devient une colonne séparée avec 0 ou 1. "
               "Ex : couleur → rouge=1/0, bleu=1/0, vert=1/0.",
    "Label Encoding": "**Encodage par numéro** — Chaque catégorie reçoit un numéro. "
                      "Ex : petit=0, moyen=1, grand=2.",
    "Target Encoding": "**Encodage par la cible** — Chaque catégorie est remplacée par la moyenne "
                       "de la cible pour cette catégorie. Puissant mais risque de fuite d'information.",
    "Outlier": "**Valeur aberrante** — Un chiffre très éloigné des autres. "
               "Ex : un salaire de 500 000 € quand la moyenne est à 35 000 €.",
    "NaN": "**Valeur manquante** — Une case vide dans le tableau. "
           "L'information n'a pas été renseignée.",
    "Corrélation": "**Lien entre deux colonnes** — Quand une colonne augmente, l'autre aussi (ou diminue). "
                   "Ex : surface et prix sont très corrélés.",
    "Multicolinéarité": "**Colonnes qui disent la même chose** — Deux colonnes très similaires. "
                        "En garder les deux peut perturber le modèle.",
    "Imputation": "**Remplissage des trous** — Remplacer les valeurs manquantes par une valeur "
                  "estimée (moyenne, médiane, ou valeur la plus fréquente).",
    "Variable cible": "**Ce qu'on veut prédire** — La colonne qui contient la réponse. "
                      "Ex : le prix de vente, le diagnostic oui/non.",
    "Variable explicative": "**Les indices pour prédire** — Les colonnes qui contiennent les informations "
                            "utiles pour deviner la cible. Ex : surface, localisation, nombre de pièces.",
}


# ═══════════════════════════════════════════════════════════
# RECOMMANDATIONS AUTOMATIQUES
# ═══════════════════════════════════════════════════════════

def recommend_missing_strategy(series: pd.Series, col_name: str, na_pct: float) -> dict:
    """Recommande une stratégie pour traiter les valeurs manquantes d'une colonne.

    Returns:
        Dict avec 'strategy', 'label', 'reason', 'confidence' (haute/moyenne/basse).
    """
    is_numeric = pd.api.types.is_numeric_dtype(series)

    # Trop de manquants → supprimer la colonne
    if na_pct > 60:
        return {
            "strategy": "drop_column",
            "label": "🔴 Supprimer cette colonne",
            "reason": f"Plus de {na_pct:.0f}% de trous — cette colonne est trop incomplète pour être fiable.",
            "confidence": "haute",
        }

    if na_pct > 40:
        return {
            "strategy": "drop_column",
            "label": "🟠 Supprimer cette colonne (conseillé)",
            "reason": f"{na_pct:.0f}% de trous — la colonne est utilisable mais les valeurs "
                      "imputées risquent de biaiser les résultats.",
            "confidence": "moyenne",
        }

    if is_numeric:
        # Vérifier s'il y a des valeurs extrêmes
        clean = series.dropna()
        if len(clean) > 0:
            skew = abs(clean.skew()) if len(clean) > 2 else 0
            if skew > 2:
                return {
                    "strategy": "median",
                    "label": "🟢 Remplir par la valeur du milieu (médiane)",
                    "reason": f"Cette colonne a des valeurs extrêmes (asymétrie = {skew:.1f}). "
                              "La médiane n'est pas influencée par les extrêmes, contrairement à la moyenne.",
                    "confidence": "haute",
                }
            else:
                return {
                    "strategy": "mean",
                    "label": "🟢 Remplir par la moyenne",
                    "reason": "Les valeurs sont bien réparties. La moyenne est un bon choix pour boucher les trous.",
                    "confidence": "haute",
                }
    else:
        return {
            "strategy": "mode",
            "label": "🟢 Remplir par la valeur la plus fréquente",
            "reason": "Pour du texte ou des catégories, on remplace par la valeur qui revient le plus souvent.",
            "confidence": "haute",
        }

    return {
        "strategy": "median",
        "label": "🟢 Remplir par la médiane",
        "reason": "Choix par défaut le plus sûr pour les données numériques.",
        "confidence": "moyenne",
    }


def recommend_outlier_strategy(series: pd.Series, col_name: str, n_outliers: int) -> dict:
    """Recommande une stratégie pour les valeurs aberrantes."""
    n_total = len(series)
    pct = n_outliers / n_total * 100 if n_total > 0 else 0

    if pct > 10:
        return {
            "strategy": "cap",
            "label": "🟢 Plafonner les extrêmes (recommandé)",
            "reason": f"{pct:.1f}% de valeurs extrêmes — trop pour les supprimer. "
                      "Le plafonnement ramène les valeurs extrêmes à des limites raisonnables.",
            "confidence": "haute",
        }
    elif pct > 2:
        return {
            "strategy": "cap",
            "label": "🟢 Plafonner les extrêmes",
            "reason": f"{n_outliers} valeurs extrêmes détectées ({pct:.1f}%). "
                      "Les plafonner conserve l'information tout en limitant leur impact.",
            "confidence": "haute",
        }
    else:
        return {
            "strategy": None,
            "label": "🟢 Conserver (peu d'impact)",
            "reason": f"Seulement {n_outliers} valeur(s) extrême(s) ({pct:.1f}%). "
                      "Pas assez pour perturber l'analyse.",
            "confidence": "haute",
        }


def recommend_encoding(series: pd.Series, col_name: str) -> dict:
    """Recommande une méthode d'encodage pour une variable catégorielle."""
    n_unique = series.nunique()

    if n_unique == 2:
        return {
            "strategy": "label",
            "label": "🟢 Encodage simple (0/1)",
            "reason": f"Seulement 2 catégories → un simple 0 et 1 suffit.",
            "confidence": "haute",
        }
    elif n_unique <= 10:
        return {
            "strategy": "one_hot",
            "label": "🟢 Une colonne par catégorie (One-Hot)",
            "reason": f"{n_unique} catégories — assez peu pour créer une colonne par catégorie "
                      "sans exploser la taille du tableau.",
            "confidence": "haute",
        }
    elif n_unique <= 20:
        return {
            "strategy": "target",
            "label": "🟡 Encodage par la cible (Target Encoding)",
            "reason": f"{n_unique} catégories — trop pour créer une colonne chacune. "
                      "On remplace chaque catégorie par la moyenne de la cible.",
            "confidence": "moyenne",
        }
    else:
        return {
            "strategy": "drop",
            "label": "🔴 Supprimer (trop de catégories)",
            "reason": f"{n_unique} catégories différentes — probablement un identifiant unique. "
                      "Cette colonne n'apportera rien au modèle.",
            "confidence": "haute",
        }


def recommend_normalization(df: pd.DataFrame, columns: list) -> dict:
    """Recommande si la normalisation est nécessaire et quelle méthode utiliser."""
    if not columns:
        return {"needed": False, "method": None, "reason": "Aucune colonne numérique."}

    ranges = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            r = df[col].max() - df[col].min()
            if r > 0:
                ranges[col] = r

    if not ranges:
        return {"needed": False, "method": None, "reason": "Pas de variation dans les colonnes."}

    max_range = max(ranges.values())
    min_range = min(ranges.values())

    if max_range / max(min_range, 0.001) > 100:
        return {
            "needed": True,
            "method": "standard",
            "columns": list(ranges.keys()),
            "reason": "Les colonnes ont des échelles très différentes "
                      f"(de {min_range:.0f} à {max_range:.0f}). "
                      "Sans normalisation, les colonnes avec de grands chiffres "
                      "vont dominer les autres.",
        }
    else:
        return {
            "needed": False,
            "method": None,
            "reason": "Les colonnes ont des échelles similaires. La normalisation n'est pas indispensable.",
        }


def recommend_features(df: pd.DataFrame, target_col: str,
                       problem_type: str = "Régression") -> list:
    """Analyse et recommande les meilleures variables explicatives.

    Returns:
        Liste de dicts avec 'col', 'score', 'recommendation', 'reason'.
    """
    recommendations = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if not numeric_cols or target_col not in df.columns:
        return recommendations

    target = df[target_col]

    for col in numeric_cols:
        series = df[col]
        rec = {"col": col}

        # Corrélation avec la cible
        try:
            if pd.api.types.is_numeric_dtype(target):
                corr = abs(series.corr(target))
            else:
                corr = 0
        except Exception:
            corr = 0

        rec["correlation"] = round(corr, 3)

        # Nombre de NaN
        na_pct = series.isna().mean() * 100

        # Variance nulle
        if series.nunique() <= 1:
            rec["score"] = 0
            rec["recommendation"] = "🔴 Inutile"
            rec["reason"] = "Cette colonne a toujours la même valeur — elle n'apporte aucune information."
        elif na_pct > 50:
            rec["score"] = round(corr * 0.3, 3)
            rec["recommendation"] = "🔴 Déconseillée"
            rec["reason"] = f"Trop de trous ({na_pct:.0f}%). Le signal est noyé dans les données manquantes."
        elif corr > 0.7:
            rec["score"] = round(corr, 3)
            rec["recommendation"] = "🟢 Très utile"
            rec["reason"] = f"Forte relation avec la cible (corrélation : {corr:.0%}). À garder absolument."
        elif corr > 0.4:
            rec["score"] = round(corr, 3)
            rec["recommendation"] = "🟢 Utile"
            rec["reason"] = f"Relation modérée avec la cible ({corr:.0%}). Bonne candidate."
        elif corr > 0.15:
            rec["score"] = round(corr, 3)
            rec["recommendation"] = "🟡 Potentiellement utile"
            rec["reason"] = f"Relation faible ({corr:.0%}). Peut aider en complément d'autres variables."
        else:
            rec["score"] = round(corr, 3)
            rec["recommendation"] = "🟠 Peu utile"
            rec["reason"] = f"Très peu liée à la cible ({corr:.0%}). Peut être retirée sans perte."

        recommendations.append(rec)

    # Trier par score décroissant
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations


def suggest_next_step(current_step: int, session_state: dict) -> str:
    """Génère un conseil personnalisé pour la prochaine étape."""
    suggestions = {
        0: "Vos fichiers sont chargés ! Passez à l'étape 1 pour vérifier le type de chaque colonne.",
        1: "Types vérifiés ! Passez à l'étape 2 (si plusieurs fichiers) ou directement à l'étape 3.",
        2: "Données fusionnées ! Passez à l'étape 3 pour inspecter la qualité.",
        3: "Audit terminé ! Passez à l'étape 4 pour nettoyer les problèmes détectés.",
        4: "Données prêtes ! Vous pouvez explorer les graphiques (étape 5) ou lancer directement les modèles (étape 6).",
        5: "Passez à l'étape 6 pour entraîner vos modèles prédictifs !",
        6: "Modèles entraînés ! Vous pouvez optimiser (étape 7) ou analyser les erreurs (étape 9).",
        7: "Passez à l'étape 9 pour analyser les erreurs ou à l'étape 10 pour le rapport final.",
        8: "Relancez le modèle après vos modifications pour voir l'impact.",
        9: "Passez à l'étape 10 pour générer votre rapport final.",
        10: "Vous pouvez utiliser votre modèle sur de nouvelles données (étape 11).",
        11: "Bravo, votre projet est terminé ! 🎉",
    }
    return suggestions.get(current_step, "")


def interpret_model_results(results: list, problem_type: str) -> str:
    """Génère un commentaire en langage simple sur les résultats des modèles."""
    if not results:
        return "Aucun résultat à analyser."

    best = max(results, key=lambda r: r.get("test_score", 0))
    worst = min(results, key=lambda r: r.get("test_score", 0))
    score = best.get("test_score", 0)
    train_score = best.get("train_score", 0)
    gap = abs(train_score - score) * 100

    parts = []

    # Score global
    if problem_type == "Régression":
        if score > 0.85:
            parts.append(f"🟢 **Excellents résultats !** Le meilleur modèle ({best['name']}) "
                         f"explique **{score:.0%}** des variations de vos données.")
        elif score > 0.65:
            parts.append(f"🟡 **Résultats corrects.** Le modèle {best['name']} "
                         f"explique **{score:.0%}** des variations. Il y a de la marge d'amélioration.")
        elif score > 0.4:
            parts.append(f"🟠 **Résultats moyens.** Le modèle n'explique que **{score:.0%}** "
                         "des variations. Vos données ne contiennent peut-être pas assez d'informations.")
        else:
            parts.append(f"🔴 **Résultats faibles.** Le modèle n'explique que **{score:.0%}** "
                         "des variations. Il faut revoir les données ou ajouter des variables.")
    else:
        if score > 0.9:
            parts.append(f"🟢 **Excellents résultats !** Le meilleur modèle ({best['name']}) "
                         f"a **{score:.0%}** de bonnes réponses.")
        elif score > 0.75:
            parts.append(f"🟡 **Résultats corrects.** {best['name']} atteint "
                         f"**{score:.0%}** de bonnes réponses.")
        elif score > 0.6:
            parts.append(f"🟠 **Résultats moyens.** Seulement **{score:.0%}** de bonnes réponses.")
        else:
            parts.append(f"🔴 **Résultats faibles.** Seulement **{score:.0%}** de bonnes réponses.")

    # Overfitting
    if gap > 15:
        parts.append(f"\n⚠️ **Attention au sur-apprentissage** : écart de {gap:.0f} points "
                     "entre entraînement et test. Le modèle a peut-être \"appris par cœur\".")
    elif gap > 8:
        parts.append(f"\nℹ️ Léger sur-apprentissage détecté ({gap:.0f} points d'écart). À surveiller.")
    else:
        parts.append(f"\n✅ Pas de sur-apprentissage détecté (écart de {gap:.0f} points seulement).")

    # Conseils
    if score < 0.65:
        parts.append("\n**💡 Conseils pour améliorer :**")
        parts.append("- Retournez à l'étape 4 pour créer de nouvelles variables (combinaisons)")
        parts.append("- Vérifiez que la bonne variable cible est sélectionnée")
        parts.append("- Essayez d'ajouter plus de données si possible")

    return "\n".join(parts)


def get_step_summary(step: int, session_state: dict) -> dict:
    """Génère un résumé de bilan pour une étape terminée."""
    summary = {"title": "", "metrics": [], "alerts": [], "next": ""}

    if step == 0:
        dfs = session_state.get("raw_dataframes", {})
        total_rows = sum(len(df) for df in dfs.values())
        total_cols = sum(len(df.columns) for df in dfs.values())
        summary["title"] = "Bilan du démarrage"
        summary["metrics"] = [
            ("Fichiers", len(dfs)),
            ("Lignes totales", total_rows),
            ("Colonnes totales", total_cols),
        ]
    elif step == 4:
        df = session_state.get("prepared_df")
        if df is not None:
            na_total = df.isna().sum().sum()
            summary["title"] = "Bilan de la préparation"
            summary["metrics"] = [
                ("Lignes restantes", len(df)),
                ("Colonnes", len(df.columns)),
                ("Trous restants", na_total),
            ]
            if na_total > 0:
                summary["alerts"].append(f"⚠️ Il reste {na_total} valeurs manquantes.")
    elif step == 6:
        results = session_state.get("model_results", [])
        if results:
            best = max(results, key=lambda r: r.get("test_score", 0))
            summary["title"] = "Bilan de la modélisation"
            summary["metrics"] = [
                ("Modèles testés", len(results)),
                ("Meilleur", best.get("name", "?")),
                ("Score", f"{best.get('test_score', 0):.4f}"),
            ]

    summary["next"] = suggest_next_step(step, session_state)
    return summary
