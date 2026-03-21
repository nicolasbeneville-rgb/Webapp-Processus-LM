# -*- coding: utf-8 -*-
"""
config.py — Paramètres globaux et seuils de validation du pipeline ML.

Ce fichier centralise toutes les constantes et valeurs par défaut
utilisées à travers l'application.
"""

# ═══════════════════════════════════════════════════════════
# 1. MÉTADONNÉES DE L'APPLICATION
# ═══════════════════════════════════════════════════════════
APP_TITLE = "� ML Studio — Pipeline interactif"
APP_ICON = "🧪"
APP_LAYOUT = "wide"

# ═══════════════════════════════════════════════════════════
# 2. TYPES DE PROBLÈME SUPPORTÉS
# ═══════════════════════════════════════════════════════════
PROBLEM_TYPES = [
    "Régression",
    "Classification",
    "Série temporelle",
    "Détection d'anomalies",
]

# ═══════════════════════════════════════════════════════════
# 3. SEUILS DE VALIDATION PAR DÉFAUT
# ═══════════════════════════════════════════════════════════

# Étape 1 — Chargement
MIN_ROWS = 50                       # Nombre minimum de lignes par fichier
MIN_COLS = 2                        # Nombre minimum de colonnes par fichier

# Étape 2 — Typage
MAX_NAN_AFTER_CONVERSION_PCT = 20   # % max de NaN apparus après conversion

# Étape 3 — Consolidation
MAX_JOIN_LOSS_PCT = 30              # % max de lignes perdues lors d'une jointure

# Étape 4 — Audit
DEFAULT_QUALITY_THRESHOLD = 60      # Score qualité minimum sur 100
CORRELATION_THRESHOLD = 0.80        # Seuil de multicolinéarité
QUASI_CONSTANT_PCT = 95             # % pour détection quasi-constante
CLASS_IMBALANCE_RATIO = 10          # Ratio max entre classes (classification)

# Étape 5 — Préparation
MIN_ROWS_AFTER_CLEANING = 50       # Lignes minimum après nettoyage
MIN_FEATURES = 2                    # Variables explicatives minimum
OUTLIER_IQR_FACTOR = 1.5           # Facteur IQR pour détection des outliers
CAPPING_LOWER_PERCENTILE = 1       # Percentile bas pour capping (%)
CAPPING_UPPER_PERCENTILE = 99      # Percentile haut pour capping (%)
MAX_ONEHOT_CARDINALITY = 20        # Cardinalité max pour One-Hot Encoding

# Étape 7 — Modélisation
DEFAULT_TEST_SIZE = 0.20           # Taille du jeu de test par défaut
DEFAULT_RANDOM_STATE = 42          # Graine aléatoire par défaut
DEFAULT_CV_FOLDS = 5               # Nombre de folds cross-validation
DEFAULT_MIN_SCORE = 0.60           # Score minimum attendu (R² ou Accuracy)
DEFAULT_MAX_OVERFIT_PCT = 10       # Écart max train/test (%)

# Étape 8 — Optimisation
OPTIMIZATION_DEFAULT_ITERATIONS = 50  # Itérations par défaut Random Search
MIN_IMPROVEMENT_PCT = 1.0             # Seuil minimum d'amélioration (%)

# ═══════════════════════════════════════════════════════════
# 4. OPTIONS DE CHARGEMENT DES FICHIERS
# ═══════════════════════════════════════════════════════════
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]
SEPARATORS = {
    "Virgule (,)": ",",
    "Point-virgule (;)": ";",
    "Tabulation": "\t",
}
ENCODINGS = ["utf-8", "latin-1", "ISO-8859-1", "cp1252"]
MAX_FILES = 3  # Nombre maximum de fichiers uploadés simultanément

# ═══════════════════════════════════════════════════════════
# 5. OPTIONS DE TYPAGE
# ═══════════════════════════════════════════════════════════
TARGET_TYPES = [
    "Numérique (float)",
    "Entier (int)",
    "Texte (string)",
    "Catégoriel (category)",
    "Booléen (bool)",
    "Date (datetime)",
    "À supprimer",
]

# Motifs pour détection automatique des booléens
BOOLEAN_PATTERNS = {
    "oui": True, "non": False,
    "o": True, "n": False,
    "yes": True, "no": False,
    "y": True, "n": False,
    "true": True, "false": False,
    "vrai": True, "faux": False,
    "1": True, "0": False,
    "actif": True, "inactif": False,
}

# ═══════════════════════════════════════════════════════════
# 6. OPTIONS DE JOINTURE
# ═══════════════════════════════════════════════════════════
JOIN_TYPES = {
    "INNER": "Garde uniquement les lignes présentes dans les deux tables.",
    "LEFT": "Garde toutes les lignes de la table de gauche.",
    "RIGHT": "Garde toutes les lignes de la table de droite.",
    "OUTER": "Garde toutes les lignes des deux tables.",
}
AGGREGATION_FUNCTIONS = ["sum", "mean", "median", "count", "min", "max"]

# ═══════════════════════════════════════════════════════════
# 7. MODÈLES DISPONIBLES
# ═══════════════════════════════════════════════════════════
REGRESSION_MODELS = [
    "Régression Linéaire",
    "Régression Polynomiale (degré 2)",
    "Régression Polynomiale (degré 3)",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Arbre de décision",
    "Random Forest",
    "Gradient Boosting",
    "SVR",
]

CLASSIFICATION_MODELS = [
    "Régression Logistique",
    "KNN",
    "Arbre de décision",
    "Random Forest",
    "Gradient Boosting",
    "SVM",
    "Naive Bayes",
]

ANOMALY_MODELS = [
    "Isolation Forest",
]

# ═══════════════════════════════════════════════════════════
# 8. MÉTRIQUES
# ═══════════════════════════════════════════════════════════
REGRESSION_METRICS = ["R²", "RMSE", "MAE"]
CLASSIFICATION_METRICS = ["Accuracy", "F1-Score", "AUC-ROC"]

# ═══════════════════════════════════════════════════════════
# 9. CHEMINS PAR DÉFAUT
# ═══════════════════════════════════════════════════════════
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_OUTPUT_DIR = "data/output"
MODELS_DIR = "models"
SAVE_DIR = "data/saves"

# ═══════════════════════════════════════════════════════════
# 10. NAVIGATION — LISTE DES ÉTAPES
# ═══════════════════════════════════════════════════════════
STEPS = [
    "0 — Démarrage",
    "1 — Typage",
    "2 — Consolidation",
    "3 — Diagnostic",
    "4 — Cible & Variables",
    "5 — Nettoyage",
    "6 — Transformation",
    "7 — Modélisation",
    "8 — Évaluation",
    "9 — Optimisation & Prédiction",
]
