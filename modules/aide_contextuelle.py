# -*- coding: utf-8 -*-
"""
aide_contextuelle.py — Aide contextuelle dynamique affichée dans la sidebar.

Pour chaque étape, fournit :
    - Quoi : ce que fait l'étape (2 lignes)
    - Pourquoi : justification du choix recommandé
    - Piège courant : 1 point d'attention
"""

import streamlit as st


AIDE = {
    0: {
        "quoi": "Créez ou reprenez un projet, chargez vos fichiers CSV/Excel.",
        "pourquoi": "Le type de problème (régression, classification…) détermine quels outils seront disponibles.",
        "piege": "Vérifiez que le séparateur et l'encodage sont corrects si l'aperçu semble étrange.",
    },
    1: {
        "quoi": "Vérifiez que chaque colonne est reconnue dans le bon type (nombre, texte, date…).",
        "pourquoi": "Un prix lu comme du texte empêche tout calcul. Corriger maintenant évite des erreurs plus tard.",
        "piege": "Les codes postaux (75001) ressemblent à des nombres mais sont en réalité des catégories !",
    },
    2: {
        "quoi": "Si vous avez plusieurs fichiers, fusionnez-les en un seul tableau via une colonne commune.",
        "pourquoi": "Le modèle a besoin d'un tableau unique avec toutes les informations sur chaque ligne.",
        "piege": "Si beaucoup de lignes disparaissent après la fusion, vérifiez que la colonne de jointure est la bonne.",
    },
    3: {
        "quoi": "Analyse approfondie : distributions, corrélations, anomalies, et recommandations automatiques.",
        "pourquoi": "Comprendre vos données AVANT de les transformer évite des erreurs coûteuses.",
        "piege": "Ne sautez pas cette étape — les recommandations guident toute la suite du pipeline.",
    },
    4: {
        "quoi": "Choisissez la colonne à prédire (cible) et les colonnes explicatives (features).",
        "pourquoi": "Des features mal choisies = un modèle inutile. La recommandation auto vous aide.",
        "piege": "N'incluez pas de colonne qui 'triche' (ex: une colonne calculée à partir de la cible).",
    },
    5: {
        "quoi": "Traitez les valeurs manquantes puis les outliers, dans cet ordre.",
        "pourquoi": "Il faut d'abord boucher les trous avant de pouvoir jauger les extrêmes correctement.",
        "piege": "Ne supprimez pas trop de lignes — préférez le remplacement par médiane quand c'est possible.",
    },
    6: {
        "quoi": "Encodez le texte en chiffres, mettez à l'échelle, puis créez de nouvelles variables (optionnel).",
        "pourquoi": "Les modèles ne comprennent que les nombres. Le scaling évite qu'une variable domine.",
        "piege": "Ne pas normaliser avant le split train/test — le scaler doit être fitté uniquement sur le train.",
    },
    7: {
        "quoi": "Entraînez et comparez les modèles pré-sélectionnés par le diagnostic.",
        "pourquoi": "Tester plusieurs algorithmes permet de trouver celui qui s'adapte le mieux à VOS données.",
        "piege": "Un bon score d'entraînement mais mauvais en test = surapprentissage. Regardez l'écart.",
    },
    8: {
        "quoi": "Analysez les résidus, courbes ROC, importance des features — tous les graphiques de diagnostic.",
        "pourquoi": "Le score seul ne suffit pas : comprendre WHERE le modèle se trompe est essentiel.",
        "piege": "Des résidus non aléatoires indiquent que le modèle rate un pattern dans les données.",
    },
    9: {
        "quoi": "Affinez les hyperparamètres du meilleur modèle avec comparaison claire avant/après.",
        "pourquoi": "Des réglages fins peuvent gagner quelques points de performance sans changer de modèle.",
        "piege": "Si le score baisse après optimisation, gardez les paramètres d'origine.",
    },
}

AIDE_GRAPHIQUES = {
    "reel_vs_predit": {
        "titre": "📈 Réel vs Prédit",
        "definition": "Nuage de points comparant les valeurs réelles (axe X) aux valeurs prédites (axe Y). Chaque point est une observation du jeu de test.",
        "lecture": [
            "**Diagonale parfaite** : si tous les points sont sur la ligne rouge y=x, le modèle prédit parfaitement.",
            "**Points dispersés** : plus les points s'éloignent de la diagonale, plus les erreurs sont grandes.",
            "**Biais systématique** : points décalés au-dessus (surestimation) ou en-dessous (sous-estimation) de la diagonale.",
            "**Forme d'entonnoir** : la dispersion augmente avec la valeur → le modèle est moins fiable sur les grandes valeurs (hétéroscédasticité).",
        ],
        "bon_signe": "Points serrés autour de la diagonale, sans biais visible.",
        "mauvais_signe": "Points très dispersés ou formant un nuage sans structure.",
    },
    "residus": {
        "titre": "📊 Résidus",
        "definition": "Graphique montrant l'erreur (réel − prédit) en fonction de la valeur prédite. Un résidu = 0 signifie une prédiction parfaite.",
        "lecture": [
            "**Dispersion aléatoire** autour de 0 : le modèle capture bien les patterns.",
            "**Forme en U ou en courbe** : le modèle rate un pattern non-linéaire → essayez un polynôme ou un modèle plus complexe.",
            "**Entonnoir** (dispersion croissante) : hétéroscédasticité → appliquer un log sur la cible peut aider.",
            "**Résidus surtout positifs ou négatifs** : biais → le modèle surestime ou sous-estime systématiquement.",
        ],
        "bon_signe": "Nuage aléatoire centré autour de 0, sans forme reconnaissable.",
        "mauvais_signe": "Pattern visible (courbe, entonnoir, clusters) dans les résidus.",
    },
    "distribution_residus": {
        "titre": "📉 Distribution des résidus",
        "definition": "Histogramme des erreurs (réel − prédit). Idéalement, les résidus suivent une distribution normale (en cloche) centrée sur 0.",
        "lecture": [
            "**Cloche symétrique** centrée sur 0 : le modèle est bien calibré.",
            "**Centre décalé de 0** : biais systématique dans les prédictions.",
            "**Queues épaisses** : le modèle produit beaucoup d'erreurs extrêmes.",
            "**Asymétrie** : erreurs surtout dans un sens → le modèle favorise la sous-estimation ou la surestimation.",
        ],
        "bon_signe": "Distribution symétrique, centrée sur 0, en forme de cloche.",
        "mauvais_signe": "Distribution asymétrique, décalée, ou avec des pics multiples.",
    },
    "top_erreurs": {
        "titre": "🔍 Top erreurs",
        "definition": "Tableau des observations où le modèle se trompe le plus. Permet d'identifier les cas difficiles et d'orienter les améliorations.",
        "lecture": [
            "**Valeurs atypiques** : ces lignes contiennent-elles des outliers ou des données inhabituelles ?",
            "**Points communs** : cherchez un pattern entre les erreurs (même catégorie, même tranche de valeurs…).",
            "**Colonne d'erreur** : montre l'écart entre prédiction et réalité.",
            "**Action** : si les erreurs concernent un sous-groupe précis, ajoutez des features pour le capturer.",
        ],
        "bon_signe": "Erreurs rares et dispersées, sans pattern commun.",
        "mauvais_signe": "Erreurs concentrées sur un type de données spécifique.",
    },
    "importance_features": {
        "titre": "🏆 Importance des features",
        "definition": "Classement des variables par contribution au modèle. Pour les arbres : fractionnement (gain). Pour les modèles linéaires : valeur absolue des coefficients.",
        "lecture": [
            "**Barres longues** : la variable a un fort pouvoir prédictif.",
            "**Barres courtes** : la variable contribue peu → possible candidate à la suppression.",
            "**Dominance** : si une seule variable concentre toute l'importance, le modèle est fragile.",
            "**Variable inattendue en tête** : vérifiez qu'il ne s'agit pas d'une fuite de données (data leakage).",
        ],
        "bon_signe": "Plusieurs features contribuent significativement, pas de dominance extrême.",
        "mauvais_signe": "Une seule feature > 80 % de l'importance, ou variable inattendue en tête.",
    },
    "permutation_importance": {
        "titre": "🌀 Importance par permutation",
        "definition": "Mesure l'impact de chaque variable en mélangeant aléatoirement ses valeurs et en observant la baisse du score. Fonctionne avec **tous** les modèles (model-agnostic).",
        "lecture": [
            "**Barre haute** : mélanger cette variable dégrade fortement le score → elle est cruciale.",
            "**Barre proche de 0** : la variable n'apporte rien au modèle → candidate à la suppression.",
            "**Barre négative** : mélanger la variable *améliore* le score → probable bruit.",
            "**Moustaches (barres d'erreur)** : grande variabilité = importance instable selon les données.",
        ],
        "bon_signe": "Variables les plus importantes cohérentes avec le métier, petites barres d'erreur.",
        "mauvais_signe": "Beaucoup de variables à importance négative ou importance condensée sur une seule.",
    },
    "qq_plot": {
        "titre": "📐 QQ-plot des résidus",
        "definition": "Compare la distribution des résidus à une distribution normale théorique. Chaque point représente un quantile observé vs le quantile attendu.",
        "lecture": [
            "**Points sur la diagonale** : résidus normaux → hypothèses du modèle linéaire respectées.",
            "**Queues déviantes** (extrémités qui s'éloignent) : valeurs extrêmes non captées par le modèle.",
            "**Courbe en S** : résidus avec des queues plus épaisses qu'une gaussienne.",
            "**Test de Shapiro** : p > 0.05 → normalité acceptée ; p < 0.05 → résidus non normaux.",
        ],
        "bon_signe": "Points alignés sur la diagonale, Shapiro p > 0.05.",
        "mauvais_signe": "Déviations importantes aux extrémités, Shapiro p très petit.",
    },
    "learning_curve": {
        "titre": "📈 Courbe d'apprentissage",
        "definition": "Score du modèle en fonction du nombre de données d'entraînement. Deux courbes : train (bleu) et validation (vert).",
        "lecture": [
            "**Courbes convergentes** et proches : le modèle a assez de données, il généralise bien.",
            "**Grand écart train/validation** : sur-apprentissage → le modèle mémorise au lieu d'apprendre.",
            "**Deux courbes basses** : sous-apprentissage → modèle trop simple ou features insuffisantes.",
            "**La validation remonte encore** à droite : plus de données améliorerait le score.",
            "**Zones colorées** : intervalle de confiance (±1 écart-type sur les 5 folds de CV).",
        ],
        "bon_signe": "Les deux courbes convergent vers un score élevé avec un petit écart.",
        "mauvais_signe": "Grand écart persistant ou deux courbes basses sans convergence.",
    },
    "matrice_confusion": {
        "titre": "🎯 Matrice de confusion",
        "definition": "Tableau croisé : classes réelles (lignes) vs classes prédites (colonnes). Chaque case compte le nombre d'observations.",
        "lecture": [
            "**Diagonale** : les bonnes prédictions → plus les valeurs sont élevées, mieux c'est.",
            "**Hors diagonale** : les erreurs → case (A, B) signifie 'le modèle a prédit B alors que c'était A'.",
            "**Version normalisée (0→1)** : compare les classes même si elles sont de tailles différentes.",
            "**Couleurs** : cases foncées = nombre élevé. Idéalement, seule la diagonale est foncée.",
            "**Classes confondues** : repérez les couples de classes souvent inversés → ajoutez des features pour les distinguer.",
        ],
        "bon_signe": "Diagonale foncée, hors-diagonale claire. Proportions diagonales > 0.8.",
        "mauvais_signe": "Beaucoup de valeurs hors diagonale, surtout entre deux classes.",
    },
    "roc_curve": {
        "titre": "📈 Courbe ROC",
        "definition": "Compromis entre le taux de vrais positifs (sensibilité) et le taux de faux positifs, à différents seuils de classification.",
        "lecture": [
            "**AUC (aire sous la courbe)** : 1.0 = parfait, 0.5 = aléatoire, < 0.5 = pire que le hasard.",
            "**Courbe vers le coin supérieur gauche** : le modèle détecte bien les positifs sans trop de faux positifs.",
            "**Courbe collée à la diagonale** : le modèle ne fait pas mieux que le hasard.",
            "**Choix du seuil** : chaque point de la courbe correspond à un seuil de décision différent.",
        ],
        "bon_signe": "AUC > 0.85 et courbe nettement au-dessus de la diagonale.",
        "mauvais_signe": "AUC < 0.7 ou courbe proche de la diagonale.",
    },
    "precision_recall": {
        "titre": "📊 Courbe Precision-Recall",
        "definition": "Compromis entre la précision (prédictions positives correctes) et le rappel (positifs réels trouvés). Essentielle quand les classes sont déséquilibrées.",
        "lecture": [
            "**Courbe en haut à droite** : bonne précision ET bon rappel simultanément.",
            "**Chute rapide** : le modèle sacrifie vite la précision pour gagner en rappel → beaucoup de faux positifs.",
            "**AP (Average Precision)** : aire sous la courbe, résume la performance globale.",
            "**Préférer au ROC** quand les classes sont déséquilibrées : le ROC peut être trop optimiste.",
        ],
        "bon_signe": "Courbe qui reste haute même à rappel élevé, AP > 0.8.",
        "mauvais_signe": "Chute brutale de la précision dès les premiers rappels.",
    },
    "erreurs_classification": {
        "titre": "🔍 Exemples mal classés",
        "definition": "Tableau des observations mal prédites, montrant la vraie classe et la classe prédite.",
        "lecture": [
            "**Vrai vs Prédit** : identifiez les confusions récurrentes entre deux classes.",
            "**Valeurs des features** : les erreurs ont-elles des valeurs inhabituelles sur certaines variables ?",
            "**Cas limites** : les erreurs se situent-elles à la frontière entre deux classes ?",
            "**Action** : si un type d'erreur domine, ajoutez des features pour distinguer ces classes.",
        ],
        "bon_signe": "Erreurs rares et dispersées, sans pattern récurrent.",
        "mauvais_signe": "Toujours les mêmes classes confondues, ou erreurs liées à un sous-groupe.",
    },
    "impact_classe": {
        "titre": "🔬 Impact par classe",
        "definition": "Coefficients signés du modèle pour chaque feature, par classe. Disponible uniquement pour les modèles linéaires (régression logistique, SVM linéaire…).",
        "lecture": [
            "🟢 **Barre verte (positive)** : une valeur élevée de la feature **augmente** la probabilité de cette classe.",
            "🔴 **Barre rouge (négative)** : une valeur élevée de la feature **diminue** la probabilité de cette classe.",
            "**Amplitude** : coefficient de grande amplitude = fort impact sur la décision.",
            "**Comparer les classes** : changez la classe dans le menu déroulant pour voir les différences.",
        ],
        "bon_signe": "Les coefficients importants correspondent à des variables logiques pour la classe.",
        "mauvais_signe": "Variables non pertinentes avec des coefficients élevés → possible fuite de données.",
    },
    "calibration_curve": {
        "titre": "🎯 Courbe de calibration",
        "definition": "Compare les probabilités prédites aux fréquences réellement observées. Disponible uniquement en classification binaire.",
        "lecture": [
            "**Courbe sur la diagonale** : probabilités fiables (quand le modèle dit 70 %, c'est vrai 70 % du temps).",
            "**Au-dessus de la diagonale** : le modèle **sous-estime** les probabilités.",
            "**En-dessous de la diagonale** : le modèle **sur-estime** les probabilités.",
            "**Utilité** : une bonne calibration est essentielle si les probabilités servent à prendre des décisions (scoring, triage…).",
        ],
        "bon_signe": "Courbe proche de la diagonale, probabilités fiables.",
        "mauvais_signe": "Forte déviation → envisager une calibration post-hoc (Platt, isotonique).",
    },
}


def afficher_aide_graphique(chart_key: str):
    """Affiche un expander ❓ avec l'aide détaillée pour un graphique."""
    aide = AIDE_GRAPHIQUES.get(chart_key)
    if not aide:
        return
    with st.expander("❓ Comment lire ce graphique"):
        st.markdown(f"**{aide['titre']}**")
        st.markdown(f"📖 {aide['definition']}")
        st.markdown("**🔎 Comment le lire :**")
        for point in aide["lecture"]:
            st.markdown(f"  {point}")
        st.success(f"✅ **Bon signe :** {aide['bon_signe']}")
        st.error(f"⚠️ **Mauvais signe :** {aide['mauvais_signe']}")


GLOSSAIRE = {
    "Feature": "Une colonne du tableau qui sert d'information pour prédire. Synonyme : variable explicative.",
    "Cible (Target)": "La colonne qu'on veut prédire. Ex : le prix, la catégorie.",
    "Overfitting": "Le modèle a 'appris par cœur' les données d'entraînement et performe mal sur de nouvelles données.",
    "Underfitting": "Le modèle est trop simple pour capter les patterns des données.",
    "Scaling": "Mettre toutes les colonnes numériques à la même échelle (ex : 0 à 1).",
    "Encoding": "Convertir du texte en chiffres pour que le modèle puisse l'utiliser.",
    "One-Hot": "Crée une colonne 0/1 par catégorie. Ex : couleur → couleur_rouge, couleur_bleu.",
    "Label Encoding": "Remplace chaque catégorie par un numéro. Ex : A→0, B→1, C→2.",
    "Cross-Validation": "Tester le modèle plusieurs fois sur différentes portions des données.",
    "Train/Test Split": "Séparer les données : une partie pour apprendre, une partie pour vérifier.",
    "R²": "Score de 0 à 1. Plus c'est proche de 1, mieux le modèle prédit les nombres.",
    "Accuracy": "% de bonnes réponses en classification (ex : 90% = 9 sur 10 correct).",
    "RMSE": "Erreur moyenne. Plus c'est petit, plus le modèle est précis.",
    "F1-Score": "Équilibre entre précision et rappel. Utile quand les classes sont déséquilibrées.",
    "Résidu": "Erreur = valeur réelle − valeur prédite. Idéalement petit et aléatoire.",
    "Outlier": "Valeur extrême, très éloignée des autres. Peut fausser le modèle.",
    "Corrélation": "Mesure du lien entre deux colonnes. 1 = lien parfait, 0 = aucun lien.",
    "Random Forest": "Combine plein d'arbres de décision pour une prédiction robuste.",
    "Gradient Boosting": "Améliore progressivement les prédictions, très performant.",
    "Hyperparamètre": "Réglage du modèle (ex : nombre d'arbres, profondeur max).",
    "NaN": "Not a Number — une case vide ou une valeur manquante dans les données.",
    "Pipeline": "Enchaînement d'étapes automatiques : nettoyage → transformation → modélisation.",
    "Série temporelle": "Données ordonnées dans le temps (ex : ventes par jour).",
    "Stationnarité": "Une série est stationnaire si sa moyenne et sa variance ne changent pas dans le temps.",
    "ARIMA": "Modèle classique pour prédire les séries temporelles.",
}


def afficher_aide(etape: int):
    """Affiche le bandeau d'aide contextuelle dans la sidebar."""
    aide = AIDE.get(etape)
    if not aide:
        return
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Aide")
    st.sidebar.markdown(f"**Quoi :** {aide['quoi']}")
    st.sidebar.markdown(f"**Pourquoi :** {aide['pourquoi']}")
    st.sidebar.warning(f"⚠️ **Piège :** {aide['piege']}")


def afficher_aide_etape(etape: int):
    """Affiche le numéro d'étape + aide contextuelle en haut de la page principale."""
    st.caption(f"ÉTAPE {etape}")
    aide = AIDE.get(etape)
    if not aide:
        return
    with st.expander("💡 Pourquoi cette étape ?", expanded=False):
        st.markdown(f"**Ce que vous faites ici :** {aide['quoi']}")
        st.markdown(f"**Pourquoi c'est important :** {aide['pourquoi']}")
        st.warning(f"⚠️ **Piège courant :** {aide['piege']}")


def afficher_glossaire():
    """Affiche le contenu du glossaire (appelé depuis un expander dans app.py)."""
    for terme, explication in GLOSSAIRE.items():
        st.markdown(f"**{terme}** : {explication}")
        st.divider()
