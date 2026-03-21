# 🤖 Pipeline ML — Guide interactif

Application web interactive avec **Streamlit** qui guide un utilisateur non codeur à travers toutes les étapes d'un projet Data Science / Machine Learning, de l'import des données brutes jusqu'à la validation d'un modèle de régression ou de classification.

## Prérequis

- **Python 3.8+** (recommandé : 3.10 ou 3.11)
- pip

## Installation

```bash
# 1. Cloner ou télécharger le projet
cd Webapp_Processus_LM

# 2. Créer un environnement virtuel
python -m venv env

# 3. Activer l'environnement
# Windows :
env\Scripts\activate
# macOS / Linux :
source env/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

## Structure du projet

```
Webapp_Processus_LM/
│
├── app.py                        ← Point d'entrée Streamlit
├── config.py                     ← Paramètres globaux et seuils
├── requirements.txt              ← Dépendances Python
├── README.md                     ← Ce fichier
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            ← Chargement et typage des données
│   ├── consolidation.py          ← Jointures et agrégations
│   ├── audit.py                  ← Analyse statistique et EDA
│   ├── preprocessing.py          ← Nettoyage, normalisation, encodage
│   ├── feature_engineering.py    ← Création et sélection de variables
│   ├── models.py                 ← Entraînement des modèles
│   ├── evaluation.py             ← Métriques et diagnostics
│   └── validators.py             ← Points de validation
│
├── data/
│   ├── raw/                      ← Données brutes (jamais modifiées)
│   ├── processed/                ← Données nettoyées
│   └── output/                   ← Résultats et prédictions
│
└── models/                       ← Modèles sauvegardés (.pkl)
```

## 🤖 GitHub Copilot — Agent ML Spécialisé

Un agent Copilot **spécialisé pour ML Studio** a été configuré pour accélérer le développement ML et garantir la qualité du code. Il s'active automatiquement quand vous codez dans `src/` et `modules/`.

### ✨ Fonctionnalités

- **Auto-activation** — Chargement automatique des best practices ML quand vous éditez `src/models.py`, `modules/m4_entrainement.py`, etc.
- **Pattern guidance** — Suggestions intelligentes basées sur les 500+ lignes de patterns validés du projet
- **Best practices embarquées** — Hyperparameter grids, data validation rules, error handling patterns
- **Code review** — Prompt `/ml-review` pour vérifier completeness et quality gates

### 📚 Ressources

- **Fichier d'instructions** : [copilot-instructions.md](copilot-instructions.md) — Guide complet (12 sections)
- **Prompt de review** : [`/.github/prompts/ml-code-review.prompt.md`](.github/prompts/ml-code-review.prompt.md) (accès via `/ml-review` en chat)

### 🎯 Usage Exemples

#### 1. Auto-activation (pas de commande nécessaire)
Ouvrez `src/models.py` et demandez :
```
Write a train_multiple() function that compares 5 regression models
```
→ Copilot applique automatiquement patterns de config.py, hyperparameter grids, et validation rules

#### 2. Code Review avec prompt
En chat VS Code, tapez :
```
/ml-review Review my feature_engineering() function for data leakage and multicollinearity checks
```
→ Retour structuré avec checklist best practices, issues, et recommendations

#### 3. Hyperparameter validation
```
/ml-review Is this Random Forest grid appropriate? 
{
  "n_estimators": [50, 100, 200, 500],
  "max_depth": [3, 5, 10, 15, 20],
}
```
→ Copilot suggère RandomizedSearchCV, compare à DEFAULT_PARAM_GRIDS de config.py

### 📖 Couverture Copilot

Le guide Copilot couvre :
- ✅ Model instantiation patterns (`get_model()` registry)
- ✅ Training logic (`train_model()` + cross-validation)
- ✅ Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- ✅ Feature engineering (combine, transform, discretize)
- ✅ Quality validation rules (config.py thresholds)
- ✅ Common pitfalls (data leakage, overfitting, session state)
- ✅ Metrics interpretation (R², RMSE, F1, AUC-ROC)

### ⚙️ Configuration

Les instructions Copilot sont configurées pour :
- Charger automatiquement quand vous éditez `src/**/*.py` ou `modules/**/*.py`
- Inclure best practices, patterns validés, et seuils de qualité du projet
- Supporter les principales commandes de review et réfactoring

📌 **Pas de configuration supplémentaire nécessaire** — utilisation directe en chat VS Code.

---

## Étapes de l'application

| # | Étape | Description |
|---|-------|-------------|
| 0 | **Configuration** | Nom du projet, type de problème, seuils de validation |
| 1 | **Chargement** | Import de 1 à 3 fichiers CSV ou Excel |
| 2 | **Typage** | Vérification et conversion des types de colonnes |
| 3 | **Consolidation** | Jointures et agrégations entre fichiers |
| 4 | **Audit EDA** | Qualité des données, statistiques, corrélations |
| 5 | **Préparation** | Nettoyage, normalisation, encodage, sélection cible |
| 6 | **Graphiques** | Visualisations exploratoires interactives |
| 7 | **Modélisation** | Entraînement et comparaison de modèles |
| 8 | **Optimisation** | Recherche des meilleurs hyperparamètres |
| 9 | **Variables** | Modification et ajout de features |
| 10 | **Résidus** | Analyse des erreurs du modèle |
| 11 | **Validation** | Rapport final et exports (CSV, HTML, .pkl) |
| 12 | **Prédiction** | Application du modèle sur de nouvelles données |

## Formats de fichiers acceptés

- **CSV** (`.csv`) — séparateurs : virgule, point-virgule, tabulation
- **Excel** (`.xlsx`, `.xls`)
- Encodages supportés : UTF-8, latin-1, ISO-8859-1, cp1252

## Modèles disponibles

### Régression
Régression Linéaire, Ridge, Lasso, ElasticNet, Arbre de décision, Random Forest, Gradient Boosting, SVR

### Classification
Régression Logistique, KNN, Arbre de décision, Random Forest, Gradient Boosting, SVM, Naive Bayes

## Licence

Projet à usage personnel et pédagogique.
