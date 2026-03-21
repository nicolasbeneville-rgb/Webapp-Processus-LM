# ML Studio — Document de revue Data Scientist

> **Objectif** : Ce document décrit exhaustivement le fonctionnement de l'application ML Studio, un pipeline ML interactif sous Streamlit. Il est destiné à un data scientist pour validation du contenu, des choix méthodologiques, des garde-fous et de la complétude.
>
> **Public cible de l'appli** : Analystes métier non-codeurs. Toute l'interface est en français simple avec des guides pédagogiques intégrés à chaque étape.
>
> **Stack technique** : Python, Streamlit, scikit-learn, statsmodels, pandas, matplotlib, plotly.

---

## Table des matières

1. [Vue d'ensemble du pipeline](#1-vue-densemble-du-pipeline)
2. [Étape 0 — Démarrage](#2-étape-0--démarrage)
3. [Étape 1 — Typage](#3-étape-1--typage)
4. [Étape 2 — Consolidation](#4-étape-2--consolidation)
5. [Étape 3 — Diagnostic](#5-étape-3--diagnostic)
6. [Étape 4 — Cible & Variables](#6-étape-4--cible--variables)
7. [Étape 5 — Nettoyage](#7-étape-5--nettoyage)
8. [Étape 6 — Transformation](#8-étape-6--transformation)
9. [Étape 7 — Modélisation](#9-étape-7--modélisation)
10. [Étape 8 — Évaluation](#10-étape-8--évaluation)
11. [Étape 9 — Optimisation & Prédiction](#11-étape-9--optimisation--prédiction)
12. [Branchement par type de problème](#12-branchement-par-type-de-problème)
13. [Tableau complet des seuils et constantes](#13-tableau-complet-des-seuils-et-constantes)
14. [Logique des recommandations automatiques](#14-logique-des-recommandations-automatiques)
15. [Modèles disponibles et hyperparamètres](#15-modèles-disponibles-et-hyperparamètres)
16. [Persistance et versionnement](#16-persistance-et-versionnement)
17. [Cas d'usage non couvert : Détection d'anomalies dans les séries](#17-cas-dusage-non-couvert--détection-danomalies-dans-les-séries)
18. [Points d'attention et limites](#18-points-dattention-et-limites)

---

## 1. Vue d'ensemble du pipeline

Le pipeline suit un déroulé linéaire en **10 étapes** (0 à 9). Chaque étape est verrouillée tant que la précédente n'est pas validée. L'utilisateur progresse via des boutons de validation explicites.

```
Étape 0         Étape 1       Étape 2           Étape 3        Étape 4
Démarrage   →   Typage    →   Consolidation →   Diagnostic →   Cible &
(chargement,     (conversion    (jointure/        (EDA, score     Variables
 type problème)  des types)     agrégation)       qualité)       (target,
                                                                 features)
    ↓
Étape 5         Étape 6          Étape 7          Étape 8         Étape 9
Nettoyage   →   Transformation → Modélisation →   Évaluation  →   Optimisation
(manquants,      (encoding,       (entraînement,   (métriques,      & Prédiction
 doublons,       scaling,         split,           résidus,         (hyperparams,
 outliers)       feature eng.)    cross-val)       importance)      inférence, API)
```

**Trois types de problèmes** supportés, avec branchement conditionnel à partir de l'étape 3 :
- **Régression** : prédire une valeur numérique continue
- **Classification** : prédire une catégorie
- **Série temporelle** : prédire l'évolution future d'une variable dans le temps (ARIMA/SARIMA)

---

## 2. Étape 0 — Démarrage

### Ce que fait l'étape
L'utilisateur nomme son projet, choisit le type de problème et charge ses fichiers de données.

### Choix proposés à l'utilisateur
| Élément | Options | Défaut |
|---------|---------|--------|
| Nom du projet | Texte libre | — |
| Type de problème | Régression, Classification, Série temporelle | — |
| Fichiers (1 à 3) | CSV, XLSX, XLS | — |
| Séparateur (par fichier) | Virgule, Point-virgule, Tabulation | Virgule |
| Encodage (par fichier) | utf-8, latin-1, ISO-8859-1, cp1252 | utf-8 |
| Ligne d'en-tête | 0 à 10 | 0 |

### Guide pédagogique intégré
Un arbre de décision visuel aide l'utilisateur à choisir entre Régression, Classification et Série temporelle, avec des exemples concrets.

### Garde-fous
- **Minimum 50 lignes** par fichier (`MIN_ROWS = 50`) → bloquant
- **Minimum 2 colonnes** par fichier (`MIN_COLS = 2`) → bloquant
- **Maximum 3 fichiers** simultanés (`MAX_FILES = 3`)
- Alerte si série temporelle sélectionnée mais aucune colonne numérique détectée
- Alerte si < 3 colonnes pour Régression/Classification

### Données sauvegardées
- `data_raw.csv` (version 1)
- Métadonnées projet (nom, type de problème)

---

## 3. Étape 1 — Typage

### Ce que fait l'étape
L'application détecte automatiquement le type de chaque colonne, puis l'utilisateur peut corriger ou confirmer.

### Détection automatique des types
L'algorithme teste dans l'ordre :
1. **Float** → la colonne est-elle numérique ?
2. **Int** → tous les floats sont-ils des entiers ?
3. **Bool** → correspondance avec les motifs booléens (oui/non, true/false, vrai/faux, 1/0, actif/inactif)
4. **DateTime** → essai `dayfirst=True` (format européen DD/MM) puis ISO
5. **Category** → < 5% de valeurs uniques OU ≤ 20 valeurs uniques
6. **String** → défaut

#### Détection spéciale
- **Texte numérique** : reconnaît les valeurs contenant `€`, `$`, `%`, espaces, virgules décimales (ex: `1 234,56 €` → nettoyé en `1234.56`)
- **Booléens** : mapping complet FR/EN (oui/non, yes/no, vrai/faux, actif/inactif, o/n, 1/0)
- **Dates** : priorité au format européen (`dayfirst=True`), fallback ISO

### Choix proposés à l'utilisateur
Pour chaque colonne :
- Type cible parmi : Numérique (float), Entier (int), Texte (string), Catégoriel (category), Booléen (bool), Date (datetime), À supprimer
- Aperçu : valeurs d'exemple, nombre d'uniques, nombre de manquants, histogramme (si numérique)

### Garde-fous
- **Alerte si la conversion crée > 20% de NaN** (`MAX_NAN_AFTER_CONVERSION_PCT = 20`) — ex: conversion d'une colonne texte en numérique qui échoue pour beaucoup de valeurs
- Les entiers avec NaN utilisent le type nullable `Int64` de pandas

### Optimisation
- Si un seul fichier chargé → l'étape 2 (Consolidation) est automatiquement marquée comme faite et sautée

---

## 4. Étape 2 — Consolidation

### Ce que fait l'étape
Permet de joindre plusieurs fichiers entre eux et/ou d'agréger des données.

### Choix proposés à l'utilisateur

#### Agrégation (optionnelle, pré-jointure)
| Élément | Options |
|---------|---------|
| Fichier à agréger | Sélection parmi les fichiers chargés |
| Colonne de regroupement | Sélection parmi les colonnes |
| Colonnes à résumer | Multi-sélection |
| Fonction d'agrégation | sum, mean, median, count, min, max |

#### Jointure
| Élément | Options |
|---------|---------|
| Fichier gauche + clé | Sélection |
| Fichier droit + clé | Sélection |
| Type de jointure | INNER, LEFT, RIGHT, OUTER (avec description en français) |

### Prévisualisation avant jointure
- Estimation du nombre de lignes résultant (sans exécuter la jointure)
- Détection des doublons dans les clés → avertissement
- Détection des colonnes homonymes → suffixes `_x` / `_y`
- Nombre de clés communes

### Garde-fous
- **Alerte si > 30% de lignes perdues** (`MAX_JOIN_LOSS_PCT = 30`) — avertissement non bloquant
- Affichage avant/après : nombre de lignes, NaN créés, colonnes

### Si un seul fichier
L'étape est automatiquement validée avec un aperçu des données.

---

## 5. Étape 3 — Diagnostic

### Ce que fait l'étape
Analyse exploratoire complète des données (EDA) avec calcul d'un score qualité.

### Onglets affichés

#### Onglet 1 : Distributions
- Grille d'histogrammes pour toutes les colonnes numériques
- Barres pour les colonnes catégorielles (top 8 modalités)
- Annotation du coefficient d'asymétrie (skewness) sur chaque histogramme
  - Plage normale : -0.5 à 0.5
- Boxplots pour les colonnes numériques

#### Onglet 2 : Valeurs manquantes
- Heatmap (bleu = présent, rouge = manquant)
- Pourcentage de manquants par colonne avec code couleur :
  - 🟢 ≤ 20% — 🟠 20-50% — 🔴 > 50%

#### Onglet 3 : Corrélations
- Matrice de corrélation (Pearson) sous forme de heatmap triangulaire
- **Alerte multicolinéarité** : paires avec |corrélation| > 0.80 (`CORRELATION_THRESHOLD`)
- Top 10 des paires les plus corrélées

#### Onglet 4 : Anomalies
Détection automatique de 4 types d'anomalies :

| Type | Critère | Sévérité |
|------|---------|----------|
| Colonnes constantes | 1 seule valeur unique | 🔴 Erreur |
| Colonnes quasi-constantes | ≥ 95% même valeur | 🟠 Avertissement |
| Haute cardinalité | > 90% uniques ET > 50 valeurs (probable identifiant) | 🟠 Avertissement |
| Outliers | Méthode IQR × 1.5 | 📌 Info avec comptage |

#### Onglet 5 : Recommandations personnalisées
- **Scaling** : nécessaire si le ratio entre les plages de valeurs des features > 100
- **Encoding** : OneHot si ≤ 20 modalités, sinon Label encoding
- **Modèles recommandés** : top 3 avec justification
- **Score qualité global** (0-100), décomposé en :

| Composante | Points | Calcul |
|-----------|--------|--------|
| Complétude (% non-manquant) | /40 | Proportionnel au % de données présentes |
| Absence de colonnes constantes | /15 | -5 par colonne constante |
| Absence de quasi-constantes | /10 | -3 par colonne quasi-constante |
| Diversité des types | /10 | min(10, nb_types × 3) |
| Absence d'outliers | /15 | max(0, 15 - somme des % d'outliers) |
| Absence de doublons | /10 | max(0, 10 × (1 - ratio_doublons)) |

Interprétation : 🟢 ≥ 70 — 🟠 40 à 70 — 🔴 < 40

#### Onglet 6 : Série temporelle (conditionnel)
Affiché uniquement si `problem_type == "Série temporelle"` :
- Sélection de la colonne date et de la colonne valeur
- Visualisation interactive (Plotly avec range slider)
- Moyennes mobiles et analyse de tendance (régression linéaire sur le temps)
- Décomposition saisonnière (tendance + saisonnalité + résidus)
- Graphiques ACF / PACF avec explication pédagogique
- **Tests de stationnarité** :
  - ADF (H0 = non-stationnaire) : rejette si p < 0.05 → stationnaire
  - KPSS (H0 = stationnaire) : rejette si p < 0.05 → non-stationnaire
  - Conclusion croisée des deux tests
- **Suggestion automatique de l'ordre ARIMA** (p, d, q) via analyse des lags significatifs dans ACF/PACF

---

## 6. Étape 4 — Cible & Variables

### Ce que fait l'étape
L'utilisateur désigne la variable à prédire (cible) et sélectionne les variables explicatives (features).

### Branche Régression / Classification

1. **Sélection de la cible** : selectbox parmi toutes les colonnes
2. **Analyse de la cible** :
   - Distribution (histogramme ou barres selon le type)
   - Type, nombre d'uniques, % de manquants
   - **Classification** : vérification du déséquilibre de classes (ratio max/min > 10 → alerte)
3. **Corrélations cible-features** (si cible numérique) :
   - Barres horizontales de corrélation
   - Code couleur : 🟢 > 0.3 — 🟠 > 0.1 — ⚪ faible
4. **Sélection des features** :
   - Recommandation automatique basée sur la corrélation avec la cible :
     - |corr| > 0.7 : 🟢 Très utile
     - 0.4–0.7 : 🟢 Utile
     - 0.15–0.4 : 🟡 Potentiellement utile
     - < 0.15 : 🟠 Peu utile
   - Exclusion automatique des colonnes constantes (🔴) et à > 50% NaN (🔴)
   - Multi-sélection avec les features recommandées pré-cochées
5. **Garde-fou** : minimum 1 feature sélectionnée (`MIN_FEATURES = 2`, incluant la cible)

### Branche Série temporelle

1. **Sélection de la colonne date** (auto-détectée ou manuelle)
2. **Sélection de la colonne valeur** (numérique, une seule)
3. **Aperçu** : nombre de points, fréquence détectée, plage de dates, graphique interactif
4. **Info** : explication d'ARIMA (univarié) vs Mode Horizon (multivariable avec lags)

---

## 7. Étape 5 — Nettoyage

### Ce que fait l'étape
Traitement des valeurs manquantes, doublons et outliers, dans cet ordre.

### Branche Régression / Classification

#### Sous-étape 1 : Valeurs manquantes
Pour chaque colonne contenant des NaN :

| Stratégie | Description |
|-----------|-------------|
| `drop_column` | Supprimer la colonne entière |
| `drop_rows` | Supprimer les lignes contenant un NaN dans cette colonne |
| `mean` | Imputer par la moyenne (numérique) |
| `median` | Imputer par la médiane (numérique) |
| `mode` | Imputer par la valeur la plus fréquente (catégoriel) |
| `fixed` | Imputer par une valeur fixe saisie par l'utilisateur |
| `indicator` | Créer une colonne binaire `{col}_missing` + imputer (médiane/mode) |

**Recommandation automatique** (cf. [section 14](#14-logique-des-recommandations-automatiques)) pré-sélectionnée.

Aperçu avant/après affiché.

#### Sous-étape 2 : Doublons
- Détection sur l'ensemble des colonnes ou un sous-ensemble choisi
- Stratégie : Supprimer / Garder le premier / Garder le dernier
- Aperçu des lignes en double

#### Sous-étape 3 : Outliers
Détection par **méthode IQR** (Inter-Quartile Range) :
- Q1, Q3 calculés
- Bornes = [Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]
- `OUTLIER_IQR_FACTOR = 1.5`

Pour chaque colonne avec outliers :

| Stratégie | Description |
|-----------|-------------|
| `keep` | Ne rien faire |
| `drop` | Supprimer les lignes contenant des outliers |
| `cap` | Écrêter aux percentiles 1 et 99 (`CAPPING_LOWER = 1%`, `CAPPING_UPPER = 99%`) |
| `log` | Transformation log1p (gère les négatifs par décalage) |

### Branche Série temporelle

Sous-étapes spécifiques dans cet ordre :
1. **Doublons de date** : détection et suppression des dates en double
2. **Continuité & Gaps** : identification des dates manquantes dans la séquence
   - Gap défini comme > 1.5× le delta médian entre observations
   - Identification des segments contigus (gap > 5× médian)
   - Recommandations automatiques : dédoublonner (haute priorité), interpoler (basse), couper la série (haute si > 10% manquant)
3. **Interpolation** : linéaire / forward-fill / backward-fill par colonne
4. **Valeurs aberrantes** : IQR en respectant l'ordre temporel
5. **Validation** : confirmation de toutes les actions

### Garde-fous
- Chaque sous-étape est verrouillée tant que la précédente n'est pas validée
- `MIN_ROWS_AFTER_CLEANING = 50` : le nettoyage ne doit pas réduire le jeu sous 50 lignes

---

## 8. Étape 6 — Transformation

### Ce que fait l'étape
Encodage des variables catégorielles, normalisation des variables numériques, et ingénierie de features.

### Branche Régression / Classification

#### Sous-étape 1 : Encodage

| Méthode | Quand | Description |
|---------|-------|-------------|
| One-Hot | ≤ 20 modalités (`MAX_ONEHOT_CARDINALITY`) | Crée une colonne binaire par modalité |
| Label | > 20 modalités ou variable ordinale | Encode en entier (0, 1, 2, ...) |
| Ordinal | Ordre défini par l'utilisateur | Encode selon l'ordre saisi (ex: "bas < moyen < haut") |
| Target Encoding | Optionnel | Remplace chaque modalité par la moyenne de la cible |
| Drop | Inutile | Supprime la colonne |

Recommandation automatique pré-sélectionnée pour chaque colonne.

#### Sous-étape 2 : Scaling (normalisation)

| Méthode | Quand recommandé | Description |
|---------|-------------------|-------------|
| StandardScaler | Défaut | Centrage (moyenne=0) et réduction (écart-type=1) |
| MinMaxScaler | Bornes connues | Mise à l'échelle [0, 1] |
| RobustScaler | Présence d'outliers | Utilise médiane et IQR au lieu de moyenne/écart-type |

Recommandé automatiquement si le ratio max/min des plages de valeurs > 100.

#### Sous-étape 3 : Feature Engineering

| Opération | Description |
|-----------|-------------|
| Combiner | Somme, différence, ratio, produit de 2 colonnes |
| Transformer | log, sqrt, carré |
| Discrétiser | Découper une variable continue en N bins (quantile ou uniforme) |
| Supprimer | Retirer une colonne |
| Interactions | Termes croisés entre features |

### Branche Série temporelle

#### Sous-étape 1 : Analyse & Recommandations
- Détection de saisonnalité (force > 0.1 = saisonnalité présente)
  - Méthode : décomposition saisonnière, ratio variance saisonnière / (saisonnière + résiduelle)
  - Fallback : détection de pics dans l'ACF
  - Recommandation ARIMA vs SARIMA
- Recommandation de transformations :
  - Skewness > 1 → log recommandé
  - Ratio de variance 1ère/2ème moitié > 4 → log ou Box-Cox
  - Tendance non-linéaire → polynôme ou feature engineering

#### Sous-étape 2 : Transformations
- Différenciation (d=1 ou d=2 pour rendre stationnaire)
- Transformation log (avec inversion automatique pour les prévisions)
- Différenciation saisonnière
- Features polynomiales (extraction de tendance)

#### Sous-étape 3 : Scaling
Même options que Régression/Classification.

#### Sous-étape 4 : Mode Prédiction Horizon
Mode optionnel qui transforme le problème de série temporelle en **problème de régression supervisée** :
- L'utilisateur choisit un horizon (t+N jours)
- Création de la cible décalée : `{target}_t+N` (valeur future à prédire)
- Création de features :
  - **Lags** : valeurs passées (lag_1, lag_2, lag_3, ...)
  - **Rolling** : moyennes et écarts-types glissants (rolling_mean_7, rolling_std_7, ...)
  - **Deltas** : variations par rapport à N périodes avant
  - **Encodage saisonnier** : sin/cos du jour de l'année (capture la circularité : 31 déc ≈ 1 jan)
  - **Features temporelles** : année, mois, jour, jour de semaine, trimestre, week-end
  - **Autres variables numériques** sélectionnées par l'utilisateur
- Permet ensuite d'utiliser les modèles de régression classiques (RF, GB, etc.) au lieu d'ARIMA

#### Sous-étape 5 : Validation
Récapitulatif de toutes les transformations, vérification de la forme des données.

---

## 9. Étape 7 — Modélisation

### Ce que fait l'étape
Split train/test, sélection et entraînement de modèles, comparaison des résultats.

### Branche Régression / Classification

#### 1. Split des données

| Paramètre | Options | Défaut |
|-----------|---------|--------|
| Taille du test | 10% à 40% (slider) | 20% (`DEFAULT_TEST_SIZE`) |
| Méthode de split | Aléatoire / Chronologique | Aléatoire |

- **Aléatoire** : `train_test_split` avec `random_state=42`
- **Chronologique** : les premières (1-test_size)% lignes pour le train, le reste pour le test. Pas de shuffle. Adapté aux données ordonnées dans le temps.

#### 2. Sélection des modèles
Multi-sélection parmi les modèles disponibles (cf. [section 15](#15-modèles-disponibles-et-hyperparamètres)), avec pré-sélection des modèles recommandés à l'étape 3.

Options supplémentaires :
- **Cross-validation** : checkbox + nombre de folds (3 à 10, défaut 5)
- **Pondération des classes** (classification uniquement) : `class_weight='balanced'` si déséquilibre détecté

#### 3. Entraînement
- Tous les modèles sélectionnés sont entraînés séquentiellement avec barre de progression
- Métriques calculées :
  - **Régression** : R² (train + test), RMSE, MAE, % d'écart train/test
  - **Classification** : Accuracy (train + test), F1-Score (weighted), AUC-ROC (si `predict_proba` disponible), % d'écart
  - Si cross-validation : moyenne et écart-type des scores CV

#### 4. Résultats
- Tableau comparatif trié par score test décroissant
- Meilleur modèle mis en avant (🏆)
- **Statut par modèle** :
  - ✅ si `test_score ≥ 0.60` ET `écart ≤ 10%`
  - ⚠️ si `test_score ≥ 0.50`
  - ❌ sinon
- **Alerte overfitting** : si écart train/test > 10% (`DEFAULT_MAX_OVERFIT_PCT`)
- Graphiques Réel vs Prédit pour le top 3 (régression uniquement)

#### 5. Validation
L'utilisateur choisit le modèle à conserver → sélection par dropdown + bouton de validation.

### Branche Série temporelle (ARIMA / SARIMA)

#### 1. Configuration
- Détection automatique de saisonnalité
- Choix : ARIMA (non saisonnier) ou SARIMA (saisonnier)
- **Ordre ARIMA** (p, d, q) :
  - Suggestion automatique via analyse ACF/PACF
  - Override manuel par 3 sliders
  - p = ordre auto-régressif, d = différenciation, q = moyenne mobile
- **Ordre SARIMA** (P, D, Q, m) :
  - 4 sliders supplémentaires
  - m = période saisonnière (ex: 12 pour mensuel, 52 pour hebdomadaire)
  - Avertissement si m > 52 (calcul intensif)
- **Ratio train** : 60% à 90% (défaut 80%)

#### 2. Entraînement
- Split chronologique (train = début, test = fin)
- Métriques : MAE, RMSE, MAPE, AIC, BIC
- Graphique : données d'entraînement + données de test + prévisions avec intervalles de confiance

#### 3. Grid Search ARIMA (optionnel)
- Test systématique : p ∈ [0,3], d ∈ [0,2], q ∈ [0,3] → 48 combinaisons
- Classement par AIC
- L'utilisateur peut adopter le meilleur ordre trouvé

---

## 10. Étape 8 — Évaluation

### Ce que fait l'étape
Évaluation approfondie du modèle retenu avec diagnostics visuels et métriques.

### Branche Régression

| Onglet | Contenu | Ce qu'on vérifie |
|--------|---------|------------------|
| Réel vs Prédit | Scatter (points sur la diagonale = bon) | Biais systématique, sous/sur-estimation |
| Résidus | Scatter résidus vs prédits | Patron → non-linéarité manquée. Cloud aléatoire autour de 0 → OK |
| Distribution résidus | Histogramme + KDE | Normalité (cloche centrée sur 0) |
| Top erreurs | 10 plus grosses erreurs | Cas particuliers, données aberrantes restantes |
| Importance features | Barres (top 15) | Quelles variables pèsent le plus |
| Permutation importance | Shuffle de chaque feature → impact sur le score | Importance modèle-agnostique |
| Diagnostic | QQ-plot + Shapiro-Wilk (p > 0.05 = normal) + courbe d'apprentissage | Normalité des résidus, sur/sous-apprentissage |

**Commentaire automatique des résidus** : analyse du centrage, de la symétrie (skewness) et de la kurtosis pour générer des recommandations textuelles.

**Rapport de régression** : pour les modèles linéaires, extraction des coefficients (bruts et dé-scalés si scaler appliqué), formule du modèle.

### Branche Classification

| Onglet | Contenu | Ce qu'on vérifie |
|--------|---------|------------------|
| Matrice de confusion | 2 vues : comptages + proportions normalisées | Erreurs par classe (FP, FN) |
| Courbe ROC | TPR vs FPR avec AUC (binaire uniquement) | Pouvoir discriminant |
| Precision-Recall | Précision vs Rappel avec AP (binaire uniquement) | Performance sur la classe rare |
| Erreurs | 10 exemples mal classés | Comprendre les confusions |
| Importance features | Idem régression | — |
| Permutation importance | Idem régression | — |
| Diagnostic | Courbe d'apprentissage + courbe de calibration (binaire) | Fiabilité des probabilités prédites |

**Rapport de classification** : précision, rappel, F1 par classe (via `classification_report`).

### Branche Série temporelle

- **Métriques** : MAE, RMSE, MAPE, AIC, MASE (Mean Absolute Scaled Error)
- **Graphique train/test/forecast** avec intervalles de confiance
- **Walk-forward validation** :
  - Re-entraîne le modèle à chaque fold (simule un déploiement réel)
  - Paramètres : nombre de folds (3–10), gap entre train et test (évite la fuite de données)
  - Résultats : MAE moyen ± écart-type, graphiques par fold

---

## 11. Étape 9 — Optimisation & Prédiction

### Ce que fait l'étape
Trois volets : optimisation des hyperparamètres, prédiction sur nouvelles données, déploiement.

### Onglet 1 : Optimisation des hyperparamètres

| Paramètre | Options | Défaut |
|-----------|---------|--------|
| Méthode | Grid Search (exhaustif) / Random Search (échantillonné) | Grid |
| Itérations (Random) | 10 à 200 | 50 |
| Folds CV | 3 à 10 | 5 |
| Grille | Éditable par l'utilisateur (cf. [section 15](#15-modèles-disponibles-et-hyperparamètres)) | Grille par défaut |

- Affichage du nombre total de combinaisons
- Résultats : score avant, meilleur score CV, score test réel, meilleurs paramètres, top 10 configs
- **Garde-fou** : adoption uniquement si le score test s'améliore. Seuil minimum d'amélioration : 1% (`MIN_IMPROVEMENT_PCT`)
- Si le score empire → l'utilisateur peut garder le modèle original

### Onglet 2 : Prédiction

#### Mode Upload CSV
- L'utilisateur charge un CSV avec les mêmes colonnes que les features d'entraînement
- Le pipeline complet est rejoué automatiquement (encodage, scaling, feature engineering) via `replay_pipeline()`
- Résultat : DataFrame original + colonne de prédictions (+ probabilités si classification binaire)
- Téléchargement du CSV résultat

#### Mode Saisie manuelle
- Un champ de saisie par feature :
  - Numérique → `number_input` avec min/max issus des données d'entraînement
  - Catégoriel ≤ 50 modalités → `selectbox`
  - Catégoriel > 50 modalités → `text_input`
- Résultat : prédiction unique (+ probabilités par classe si classification)

### Onglet 3 : Déploiement API

Export d'un package complet (.zip) contenant :

| Fichier | Rôle |
|---------|------|
| `model.joblib` | Modèle sérialisé |
| `pipeline.json` | Pipeline de transformation (pour reproductibilité) |
| `scaler.pkl` | Objet scaler entraîné |
| `encoders.pkl` | Encodeurs entraînés |
| `template.csv` | CSV vide avec les colonnes attendues |
| `appscript.js` | Code Google Apps Script pour intégration Google Sheets |
| `Dockerfile` | Conteneurisation de l'API |
| `requirements_api.txt` | Dépendances Python de l'API |
| `README_DEPLOY.md` | Guide de déploiement pas à pas |

**Intégration Google Sheets** : le code Apps Script crée un menu "ML Studio" dans Google Sheets. L'utilisateur remplit les données dans la feuille, clique sur "Prédire", et les prédictions sont auto-remplies via appel API.

---

## 12. Branchement par type de problème

```
                    ┌─────────────────────────┐
                    │   Étapes 0–2 communes    │
                    │  (Chargement, Typage,    │
                    │   Consolidation)         │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        Régression      Classification    Série temporelle
              │                │                │
     ┌────────┴────────────────┴───┐     ┌──────┴──────┐
     │  Étape 3 : Diagnostic       │     │  + Onglet   │
     │  (mêmes onglets 1-5)        │     │  TS dédié   │
     └─────────────┬───────────────┘     │  (ACF/PACF, │
                   │                     │  stationn.) │
              ┌────┴────┐               └──────┬──────┘
              │         │                      │
     Étape 4: Cible     │              Étape 4: Date
     + features (corr)  │              + 1 valeur
              │         │                      │
     Étape 5: Manquants │              Étape 5: Doublons
     Doublons, Outliers │              Gaps, Interpolation
              │         │              Outliers TS
              │         │                      │
     Étape 6: Encoding  │              Étape 6: Transforms
     Scaling, Feat. Eng. │              Scaling, Horizon
              │         │                      │
     Étape 7: Modèles   │              Étape 7: ARIMA
     sklearn (10 ou 7)  │              ou SARIMA
              │         │              + Grid Search
              │         │                      │
     Étape 8: Résidus   │              Étape 8: Walk-
     Importance, ROC,   │              forward, MASE
     Confusion          │              Conf. intervals
              │         │                      │
     Étape 9: GridSearch│              Étape 9: Forecast
     Prédiction, API    │              futur, API
              └────┬────┘                      │
                   └───────────────────────────┘
```

### Différences clés par type

| Aspect | Régression | Classification | Série temporelle |
|--------|-----------|---------------|-----------------|
| Cible | 1 colonne numérique | 1 colonne catégorielle | 1 colonne numérique + 1 colonne date |
| Features | ≥ 1, sélection par corrélation | ≥ 1, sélection par corrélation | Aucune (ARIMA univarié) ou lags/rolling (mode Horizon) |
| Nettoyage | Manquants → Doublons → Outliers | Idem | Doublons date → Gaps → Interpolation → Outliers TS |
| Modèles | 10 modèles sklearn | 7 modèles sklearn | ARIMA / SARIMA (statsmodels) |
| Métriques principales | R², RMSE, MAE | Accuracy, F1, AUC-ROC | MAE, RMSE, MAPE, AIC |
| Évaluation spécifique | Résidus, QQ-plot | Confusion, ROC, Precision-Recall | Walk-forward, intervalles de confiance |
| Split | Aléatoire ou chronologique | Aléatoire ou chronologique | Toujours chronologique |
| Optimisation | GridSearch / RandomSearch sklearn | Idem | Grid search sur (p,d,q) |

---

## 13. Tableau complet des seuils et constantes

| Constante | Valeur | Étape | Rôle | Bloquant ? |
|-----------|--------|-------|------|------------|
| `MIN_ROWS` | 50 | 0 | Nombre minimum de lignes par fichier | Oui |
| `MIN_COLS` | 2 | 0 | Nombre minimum de colonnes par fichier | Oui |
| `MAX_FILES` | 3 | 0 | Nombre maximum de fichiers uploadés | Oui |
| `MAX_NAN_AFTER_CONVERSION_PCT` | 20% | 1 | % max de NaN créés par conversion de type | Non (alerte) |
| `MAX_JOIN_LOSS_PCT` | 30% | 2 | % max de lignes perdues lors d'une jointure | Non (alerte) |
| `DEFAULT_QUALITY_THRESHOLD` | 60 | 3 | Score qualité minimum sur 100 | Non (alerte) |
| `CORRELATION_THRESHOLD` | 0.80 | 3 | Seuil d'alerte multicolinéarité | Non (alerte) |
| `QUASI_CONSTANT_PCT` | 95% | 3 | % pour détecter une colonne quasi-constante | Non (alerte) |
| `CLASS_IMBALANCE_RATIO` | 10 | 4 | Ratio max/min entre classes | Non (alerte) |
| `MIN_ROWS_AFTER_CLEANING` | 50 | 5 | Lignes minimum après nettoyage | Oui |
| `MIN_FEATURES` | 2 | 4 | Variables minimum (cible incluse) | Oui |
| `OUTLIER_IQR_FACTOR` | 1.5 | 5 | Facteur multiplicateur IQR | — |
| `CAPPING_LOWER_PERCENTILE` | 1% | 5 | Percentile bas pour écrêtage | — |
| `CAPPING_UPPER_PERCENTILE` | 99% | 5 | Percentile haut pour écrêtage | — |
| `MAX_ONEHOT_CARDINALITY` | 20 | 6 | Cardinalité max pour One-Hot encoding | — |
| `DEFAULT_TEST_SIZE` | 20% | 7 | Taille du jeu de test | — |
| `DEFAULT_RANDOM_STATE` | 42 | 7 | Graine aléatoire (reproductibilité) | — |
| `DEFAULT_CV_FOLDS` | 5 | 7 | Nombre de folds cross-validation | — |
| `DEFAULT_MIN_SCORE` | 0.60 | 7 | Score minimum pour statut ✅ | Non (alerte) |
| `DEFAULT_MAX_OVERFIT_PCT` | 10% | 7 | Écart max train/test avant alerte overfitting | Non (alerte) |
| `OPTIMIZATION_DEFAULT_ITERATIONS` | 50 | 9 | Itérations par défaut en Random Search | — |
| `MIN_IMPROVEMENT_PCT` | 1% | 9 | Amélioration minimum pour adopter le modèle optimisé | Non (info) |

---

## 14. Logique des recommandations automatiques

L'application génère des recommandations pré-sélectionnées à chaque étape pour guider l'utilisateur non-codeur. Voici la logique sous-jacente :

### Stratégie pour les valeurs manquantes

| Condition | Stratégie recommandée | Confiance |
|-----------|----------------------|-----------|
| % NaN > 60% | Supprimer la colonne | 🟢 Haute |
| % NaN > 40% | Supprimer la colonne | 🟠 Moyenne |
| Numérique, skewness > 2 | Médiane | 🟢 Haute |
| Numérique, distribution normale | Moyenne | 🟢 Haute |
| Catégoriel | Mode (valeur la plus fréquente) | 🟢 Haute |

### Stratégie pour les outliers

| Condition | Stratégie recommandée | Confiance |
|-----------|----------------------|-----------|
| % outliers > 10% | Écrêtage (cap) | 🟢 Haute |
| 2% < % outliers ≤ 10% | Écrêtage (cap) | 🟢 Haute |
| % outliers ≤ 2% | Garder (keep) | 🟢 Haute |

### Stratégie d'encodage

| Condition | Méthode recommandée | Confiance |
|-----------|---------------------|-----------|
| 2 modalités | Label (0/1) | 🟢 Haute |
| 3 à 10 modalités | One-Hot | 🟢 Haute |
| 11 à 20 modalités | Target Encoding | 🟡 Moyenne |
| > 20 modalités | Supprimer | 🔴 Haute |

### Recommandation de normalisation

| Condition | Recommandation |
|-----------|---------------|
| Ratio (max plage / min plage) > 100 | StandardScaler nécessaire |
| Ratio ≤ 100 | Scaling optionnel |

### Recommandation de features

| Corrélation avec la cible | Évaluation |
|---------------------------|------------|
| |corr| > 0.7 | 🟢 Très utile |
| 0.4 – 0.7 | 🟢 Utile |
| 0.15 – 0.4 | 🟡 Potentiellement utile |
| < 0.15 | 🟠 Peu utile |
| Colonne constante | 🔴 Exclure |
| > 50% NaN | 🔴 Exclure |

### Recommandation de modèles
Basée sur le diagnostic (étape 3), les modèles sont suggérés en fonction du type de problème, de la taille du jeu, de la présence d'outliers, et de la linéarité apparente des corrélations.

---

## 15. Modèles disponibles et hyperparamètres

### Modèles de Régression (10)

| Modèle | Bibliothèque | Hyperparamètres optimisables (grille par défaut) |
|--------|-------------|--------------------------------------------------|
| Régression Linéaire | sklearn | — (pas d'hyperparamètre) |
| Polynomiale degré 2 | sklearn (Pipeline) | — |
| Polynomiale degré 3 | sklearn (Pipeline) | — |
| Ridge | sklearn | `alpha`: [0.01, 0.1, 1, 10, 100] |
| Lasso | sklearn | `alpha`: [0.001, 0.01, 0.1, 1, 10] |
| ElasticNet | sklearn | `alpha`: [0.01, 0.1, 1], `l1_ratio`: [0.2, 0.5, 0.8] |
| Arbre de décision | sklearn | `max_depth`: [3, 5, 7, 10, 15, None] |
| Random Forest | sklearn | `n_estimators`: [50, 100, 200], `max_depth`: [5, 10, 15, None] |
| Gradient Boosting | sklearn | `n_estimators`: [50, 100, 200], `learning_rate`: [0.01, 0.1, 0.2], `max_depth`: [3, 5, 7] |
| SVR | sklearn | `C`: [0.1, 1, 10], `kernel`: ["rbf", "linear"] |

### Modèles de Classification (7)

| Modèle | Bibliothèque | Hyperparamètres optimisables (grille par défaut) |
|--------|-------------|--------------------------------------------------|
| Régression Logistique | sklearn | `C`: [0.01, 0.1, 1, 10, 100] |
| KNN | sklearn | `n_neighbors`: [3, 5, 7, 11, 15] |
| Arbre de décision | sklearn | `max_depth`: [3, 5, 7, 10, 15, None] |
| Random Forest | sklearn | (idem régression) |
| Gradient Boosting | sklearn | (idem régression) |
| SVM | sklearn | `C`: [0.1, 1, 10], `kernel`: ["rbf", "linear"] |
| Naive Bayes | sklearn | — |

### Modèles de Série temporelle (2)

| Modèle | Bibliothèque | Paramètres |
|--------|-------------|-----------|
| ARIMA(p,d,q) | statsmodels | p ∈ [0–5], d ∈ [0–2], q ∈ [0–5] (auto-suggéré via ACF/PACF) |
| SARIMA(p,d,q)(P,D,Q,m) | statsmodels | Idem + P, D, Q ∈ [0–2], m = période saisonnière |

**Note** : tous les modèles sklearn utilisent `random_state=42` pour la reproductibilité.

---

## 16. Persistance et versionnement

### Structure de sauvegarde

```
data/saves/{nom_projet}/
├── project.json                                    # Métadonnées complètes
├── save_history.json                               # Journal d'audit
├── {projet}_etape{N}_{suffixe}_v{V}.csv           # CSVs versionnés
└── models/
    ├── best_model.pkl                              # Modèle retenu
    ├── scaler.pkl                                  # Scaler entraîné
    ├── encoders.pkl                                # Encodeurs
    ├── best_result.pkl                             # Métriques du meilleur modèle
    └── all_results.pkl                             # Résultats de tous les modèles
```

### Versionnement
- Chaque sauvegarde d'étape incrémente automatiquement le numéro de version (v1, v2, v3...)
- L'utilisateur peut consulter l'historique des versions de chaque étape
- Restauration possible à n'importe quelle version

### Métadonnées sauvegardées dans `project.json`
Nom du projet, type de problème, colonne cible, features, score qualité, stratégies de nettoyage/encoding/scaling, nom du meilleur modèle, taille du test, graine aléatoire, colonnes datetime (si TS), paramètres spécifiques TS.

### Rétrocompatibilité
Support de l'ancien format de sauvegarde (meta/dataframes/*.parquet).

---

## 17. Cas d'usage non couvert : Détection d'anomalies dans les séries

### Besoin exprimé
> À la suite d'une campagne de relevé de compteurs, identifier les compteurs présentant des profils anormaux.

### Pourquoi ce n'est pas couvert actuellement
L'application traite les séries temporelles en mode **prévision** (ARIMA : prédire le futur d'UNE série). Le besoin ici est de **comparer N séries entre elles** pour trouver celles qui dévient du comportement normal. C'est un problème fondamentalement différent.

### Approches recommandées pour l'ajout

#### Approche 1 : Clustering de profils (la plus accessible pour des non-codeurs)
- **Principe** : regrouper les compteurs ayant des profils de consommation similaires, puis identifier ceux qui n'appartiennent à aucun groupe ou qui sont dans un cluster très petit
- **Méthode** :
  1. Extraire des features par compteur (consommation moyenne, écart-type, tendance, saisonnalité, min, max, nb de pics)
  2. Normaliser les features
  3. Appliquer K-Means ou DBSCAN
  4. Les compteurs dans des clusters de taille 1 ou dans le "bruit" (DBSCAN) sont anomaux
- **Avantage** : interprétable, l'utilisateur voit les groupes et comprend pourquoi un compteur est anormal
- **Modèles à ajouter** : K-Means (déjà envisageable via Clustering), DBSCAN, éventuellement HDBSCAN

#### Approche 2 : Isolation Forest sur features agrégées
- **Principe** : algorithme spécialisé dans la détection d'anomalies. Isole les observations "faciles à séparer" (= anormales)
- **Méthode** :
  1. Même extraction de features que l'approche 1
  2. Appliquer Isolation Forest → score d'anomalie par compteur
  3. Seuil configurable (ex : top 5% les plus anormaux)
- **Avantage** : pas besoin de définir le nombre de clusters, robuste aux distributions non-standard

#### Approche 3 : Distance DTW (Dynamic Time Warping)
- **Principe** : comparer les formes des séries temporelles entre elles, même si elles sont légèrement décalées dans le temps
- **Méthode** :
  1. Calculer la matrice de distance DTW entre toutes les paires de compteurs
  2. Identifier les compteurs les plus éloignés de la médiane (ou du centroïde)
- **Avantage** : capte les anomalies de forme (ex : pic de consommation décalé)
- **Inconvénient** : coûteux en calcul si beaucoup de compteurs (O(n²))

#### Approche 4 : Prédiction + écart (déjà partiellement possible)
- **Principe** : entraîner un modèle sur les compteurs "normaux", puis mesurer l'écart entre prédiction et réalité
- **Méthode** :
  1. Entraîner ARIMA/SARIMA sur un profil de référence (ou la moyenne des compteurs)
  2. Appliquer à chaque compteur
  3. Les compteurs avec un MAPE très élevé sont anormaux
- **Avantage** : utilise déjà les outils présents dans l'appli
- **Inconvénient** : suppose qu'on connaît le profil "normal"

#### Recommandation pour l'intégration dans ML Studio
L'approche la plus cohérente avec la philosophie de l'appli (non-codeurs, guidé pas à pas) serait :
1. Ajouter un **4ème type de problème** : "Détection d'anomalies" (ou l'intégrer comme sous-type de "Série temporelle")
2. L'utilisateur charge un CSV avec une colonne identifiant le compteur, une colonne date, une colonne valeur
3. L'appli **pivote** automatiquement (un compteur = une colonne ou une série)
4. **Extraction automatique de features** par compteur (moyenne, écart-type, tendance, saisonnalité, nb de NaN, amplitude, etc.)
5. **Choix de la méthode** : Isolation Forest (recommandé par défaut) ou DBSCAN ou K-Means
6. **Résultat visuel** :
   - Score d'anomalie par compteur
   - Graphique des profils colorés (normal en gris, anomaux en rouge)
   - Tableau exportable des compteurs anomaux avec raison
7. **Seuil configurable** par l'utilisateur (% d'anomalies attendues ou score minimum)

---

## 18. Points d'attention et limites

### Points soumis à validation du data scientist

#### Choix méthodologiques
1. **Score qualité (étape 3)** : la pondération (complétude 40pts, constantes 15pts, etc.) est-elle pertinente ? Le score reflète-t-il bien la qualité réelle des données pour un objectif ML ?
2. **Seuil IQR = 1.5** : c'est le standard, mais pour certaines distributions (log-normale, Pareto), cela peut flag beaucoup de valeurs. Faut-il proposer un facteur ajustable ?
3. **Recommandation d'encoding** : la règle "> 20 modalités → supprimer" est-elle trop agressive ? Des alternatives comme le hashing ou l'embedding existent.
4. **Corrélation de Pearson uniquement** : ne capture pas les relations non-linéaires. Faut-il ajouter Spearman ou le Mutual Information Score pour la sélection de features ?
5. **Target Encoding** : risque de fuite de données (data leakage) si appliqué avant le split train/test. Est-ce correctement géré ?
6. **Seuil d'overfitting à 10%** : est-ce suffisant ? Pour certains problèmes, 10% d'écart est déjà problématique.

#### Complétude du pipeline
7. **Absence de validation croisée stratifiée** : pour la classification avec classes déséquilibrées, le split devrait être stratifié. Est-ce le cas ?
8. **Pas de gestion explicite du data leakage** : les transformations (encoding, scaling) sont-elles appliquées après le split ou avant ? Si avant → fuite.
9. **Pas de SHAP** : l'importance par permutation est présente mais SHAP (SHapley Additive exPlanations) donnerait des explications plus riches. Pertinent pour le public cible ?
10. **Pas de XGBoost / LightGBM / CatBoost** : les modèles boosting de sklearn sont présents mais les bibliothèques spécialisées sont souvent plus performantes.
11. **Pas de gestion du multi-label** : la classification est mono-label uniquement.
12. **Série temporelle limitée à ARIMA/SARIMA** : pas de Prophet, pas de modèles DL (LSTM). Le mode Horizon avec régression sklearn compense-t-il ?

#### Robustesse des garde-fous
13. **MIN_ROWS = 50** : suffisant ? Certains modèles (RF, GB) peuvent avoir besoin de plus. Faut-il un seuil adaptatif selon le modèle ?
14. **Pas de limite sur le nombre de colonnes** : un OneHot sur 20 modalités × 10 colonnes = 200 features. Risque de curse of dimensionality pour les petits jeux.
15. **Pas de détection de fuite temporelle** : si l'utilisateur choisit un split aléatoire sur des données temporelles, le modèle verra le futur → scores artificiellement hauts.
16. **Walk-forward** : le gap=0 par défaut ne protège pas contre les lags créés en feature engineering. Le gap devrait-il être ≥ au plus grand lag utilisé ?

#### Pour le cas d'usage compteurs
17. **Format d'entrée** : l'appli attend un tableau plat (lignes × colonnes). Pour N compteurs avec des relevés temporels, le format naturel est long (compteur, date, valeur). L'appli ne gère pas le pivot.
18. **Volume** : si campagne de relevés = 10 000 compteurs × 365 jours → 3.6M lignes. L'appli a-t-elle été testée à cette échelle ?

---

> **Pour le relecteur** : ce document couvre l'intégralité du code source tel qu'il existe au 16 mars 2026. Toute question ou demande de précision peut être adressée en commentaire de ce fichier.
