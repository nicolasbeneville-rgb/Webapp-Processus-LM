# -*- coding: utf-8 -*-
"""
projet_manager.py — Gestion des projets (création, sauvegarde, chargement).

Chaque projet est un dossier dans projets/ avec :
    - rapport.json : état complet du projet
    - data_raw.csv : données brutes
    - data_cleaned.csv : données nettoyées
    - model_*.pkl : modèles entraînés
    - predictions.csv : prédictions
"""

import os
import json
import shutil
import zipfile
import io
from datetime import datetime

import pandas as pd
import joblib

PROJETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "projets")


def _safe_name(name: str) -> str:
    """Transforme un nom de projet en nom de dossier sécurisé."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def creer_projet(nom: str) -> dict:
    """Crée un nouveau projet et retourne le rapport initial."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier = f"{timestamp}_{_safe_name(nom)}"
    chemin = os.path.join(PROJETS_DIR, dossier)
    os.makedirs(chemin, exist_ok=True)

    rapport = {
        "nom": nom,
        "dossier": dossier,
        "chemin": chemin,
        "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "etape_courante": 0,
        "parcours": None,
        "type_ml": None,
        "colonnes_features": [],
        "colonne_cible": None,
        "nettoyage": {},
        "modeles_entraines": [],
        "meilleur_modele": None,
        "metriques": {},
        "diagnostic": {},
        "historique": [],
    }
    sauvegarder_rapport(rapport)
    return rapport


def sauvegarder_rapport(rapport: dict):
    """Sauvegarde le rapport.json dans le dossier du projet."""
    chemin = rapport.get("chemin", "")
    if not chemin:
        return
    path = os.path.join(chemin, "rapport.json")

    # Convertir les types non sérialisables
    rapport_clean = _nettoyer_pour_json(rapport)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rapport_clean, f, ensure_ascii=False, indent=2, default=str)


def _nettoyer_pour_json(obj):
    """Nettoie récursivement un objet pour la sérialisation JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _nettoyer_pour_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_nettoyer_pour_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(obj)
    return obj


def charger_rapport(chemin_projet: str) -> dict:
    """Charge le rapport.json d'un projet."""
    path = os.path.join(chemin_projet, "rapport.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lister_projets() -> list:
    """Liste tous les projets existants."""
    if not os.path.isdir(PROJETS_DIR):
        return []
    projets = []
    for dossier in sorted(os.listdir(PROJETS_DIR), reverse=True):
        chemin = os.path.join(PROJETS_DIR, dossier)
        if not os.path.isdir(chemin):
            continue
        rapport_path = os.path.join(chemin, "rapport.json")
        if os.path.isfile(rapport_path):
            try:
                with open(rapport_path, "r", encoding="utf-8") as f:
                    rapport = json.load(f)
                rapport["chemin"] = chemin
                rapport["dossier"] = dossier
                projets.append(rapport)
            except (json.JSONDecodeError, KeyError):
                projets.append({"nom": dossier, "chemin": chemin, "dossier": dossier,
                                "etape_courante": 0})
        else:
            projets.append({"nom": dossier, "chemin": chemin, "dossier": dossier,
                            "etape_courante": 0})
    return projets


def supprimer_projet(chemin_projet: str):
    """Supprime un projet et tout son contenu."""
    if os.path.isdir(chemin_projet):
        shutil.rmtree(chemin_projet)


def sauvegarder_csv(rapport: dict, df: pd.DataFrame, nom_fichier: str):
    """Sauvegarde un DataFrame CSV dans le dossier du projet."""
    chemin = rapport.get("chemin", "")
    if not chemin:
        return
    path = os.path.join(chemin, nom_fichier)
    df.to_csv(path, index=False)


def charger_csv(chemin_projet: str, nom_fichier: str) -> pd.DataFrame:
    """Charge un CSV depuis le dossier du projet."""
    path = os.path.join(chemin_projet, nom_fichier)
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def sauvegarder_modele(rapport: dict, model, nom: str):
    """Sauvegarde un modèle entraîné dans le dossier du projet."""
    chemin = rapport.get("chemin", "")
    if not chemin:
        return
    safe = _safe_name(nom)
    path = os.path.join(chemin, f"model_{safe}.pkl")
    joblib.dump(model, path)


def charger_modele(chemin_projet: str, nom: str):
    """Charge un modèle depuis le dossier du projet."""
    safe = _safe_name(nom)
    path = os.path.join(chemin_projet, f"model_{safe}.pkl")
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


def sauvegarder_objet(rapport: dict, obj, nom_fichier: str):
    """Sauvegarde un objet Python (scaler, encoder...) avec joblib."""
    chemin = rapport.get("chemin", "")
    if not chemin:
        return
    path = os.path.join(chemin, nom_fichier)
    joblib.dump(obj, path)


def charger_objet(chemin_projet: str, nom_fichier: str):
    """Charge un objet joblib depuis le dossier du projet."""
    path = os.path.join(chemin_projet, nom_fichier)
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


def ajouter_historique(rapport: dict, action: str):
    """Ajoute une entrée à l'historique du projet."""
    if "historique" not in rapport:
        rapport["historique"] = []
    rapport["historique"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
    })


def lister_fichiers_projet(chemin_projet: str) -> list:
    """Liste tous les fichiers d'un projet avec leurs métadonnées."""
    if not os.path.isdir(chemin_projet):
        return []
    fichiers = []
    for f in os.listdir(chemin_projet):
        path = os.path.join(chemin_projet, f)
        if os.path.isfile(path):
            taille = os.path.getsize(path)
            ext = os.path.splitext(f)[1]
            fichiers.append({
                "nom": f,
                "chemin": path,
                "taille": taille,
                "extension": ext,
                "categorie": "modele" if f.startswith("model_") else
                             "donnees" if ext == ".csv" else
                             "meta" if ext == ".json" else "autre",
            })
    return fichiers


def exporter_projet_zip(chemin_projet: str) -> bytes:
    """Crée un ZIP en mémoire du dossier projet complet. Retourne les bytes."""
    buf = io.BytesIO()
    dossier_nom = os.path.basename(chemin_projet)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(chemin_projet):
            for fname in files:
                full_path = os.path.join(root, fname)
                arcname = os.path.join(dossier_nom,
                                       os.path.relpath(full_path, chemin_projet))
                zf.write(full_path, arcname)
    return buf.getvalue()
