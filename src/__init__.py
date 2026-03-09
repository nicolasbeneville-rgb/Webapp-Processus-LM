# -*- coding: utf-8 -*-
"""
Package src — Modules métier du pipeline ML interactif.

Modules :
    - data_loader          : chargement et typage des données
    - consolidation        : jointures et agrégations multi-bases
    - audit                : analyse statistique et EDA
    - preprocessing        : nettoyage, normalisation, encodage
    - feature_engineering  : création et sélection de variables
    - models               : entraînement des modèles
    - evaluation           : métriques et diagnostics
    - validators           : points de validation systématiques
"""

from . import (
    data_loader,
    consolidation,
    audit,
    preprocessing,
    feature_engineering,
    models,
    evaluation,
    validators,
    guide,
    persistence,
)

__all__ = [
    "data_loader",
    "consolidation",
    "audit",
    "preprocessing",
    "feature_engineering",
    "models",
    "evaluation",
    "validators",
    "guide",
    "persistence",
]
