# -*- coding: utf-8 -*-
"""
projet_manager.py — Gestion des projets (création, sauvegarde, chargement, export portable).

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
import pickle
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


# ═══════════════════════════════════════════════════════════
# EXPORT / IMPORT PORTABLE (.mlproject)
# Fonctionne sans accès disque — basé sur session_state.
# ═══════════════════════════════════════════════════════════

def exporter_projet_portable(session_state) -> bytes:
    """Exporte l'état complet du projet depuis session_state en bytes (.mlproject).

    Le fichier .mlproject est un pickle contenant toutes les données
    nécessaires pour restaurer le projet : données, modèle, encodeurs,
    scaler, paramètres, progression.
    """
    rapport = session_state.get("rapport", {})
    best = session_state.get("meilleur_modele", {})

    export = {
        # Métadonnées
        "format_version": 1,
        "nom": rapport.get("nom", "Projet"),
        "date_export": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "problem_type": session_state.get("problem_type", ""),
        "target_col": session_state.get("target_col", ""),
        "feature_cols": session_state.get("feature_cols", []),
        "feature_cols_used": session_state.get("feature_cols_used", []),
        "etape_courante": rapport.get("etape_courante", 0),
        "rapport": {k: v for k, v in rapport.items() if k != "chemin"},

        # Données
        "df_courant": session_state.get("df_courant"),
        "X_train": session_state.get("X_train"),
        "X_test": session_state.get("X_test"),
        "y_train": session_state.get("y_train"),
        "y_test": session_state.get("y_test"),

        # Pipeline de transformation
        "encoders": session_state.get("encoders", {}),
        "scaler": session_state.get("scaler"),
        "scaled_columns": session_state.get("scaled_columns", []),
        "fe_operations": session_state.get("fe_operations", []),

        # Modèle
        "meilleur_modele": best if best else None,
        "resultats_modeles": session_state.get("resultats_modeles"),
        "opt_result": session_state.get("opt_result"),

        # Drapeaux de progression
        "flags": {
            key: session_state.get(key, False)
            for key in [
                "chargement_done", "typage_done", "consolidation_done",
                "diagnostic_done", "cible_done",
                "nettoyage_done", "manquantes_done", "doublons_done",
                "outliers_done", "encoding_done", "scaling_done",
                "transformation_done", "entrainement_done", "evaluation_done",
            ]
        },

        # Série temporelle
        "ts_datetime_col": session_state.get("ts_datetime_col"),
        "ts_value_col": session_state.get("ts_value_col"),
        "ts_series": session_state.get("ts_series"),
    }

    buf = io.BytesIO()
    pickle.dump(export, buf)
    return buf.getvalue()


def importer_projet_portable(file_bytes, session_state):
    """Restaure un projet depuis un fichier .mlproject dans session_state.

    Retourne le nom du projet importé.
    """
    data = pickle.loads(file_bytes)

    # Métadonnées
    session_state["problem_type"] = data.get("problem_type", "")
    session_state["target_col"] = data.get("target_col", "")
    session_state["feature_cols"] = data.get("feature_cols", [])
    session_state["feature_cols_used"] = data.get("feature_cols_used", [])

    # Rapport (sans chemin local)
    rapport = data.get("rapport", {})
    rapport["chemin"] = ""  # Pas de chemin local sur le cloud
    session_state["rapport"] = rapport
    session_state["etape_courante"] = data.get("etape_courante", 0)
    session_state["projet_charge"] = True

    # Données
    if data.get("df_courant") is not None:
        session_state["df_courant"] = data["df_courant"]
        nom_fichier = data.get("nom", "data") + ".csv"
        session_state["raw_dataframes"] = {nom_fichier: data["df_courant"].copy()}
        session_state["typed_dataframes"] = {nom_fichier: data["df_courant"].copy()}

    if data.get("X_train") is not None:
        session_state["X_train"] = data["X_train"]
        session_state["X_test"] = data["X_test"]
        session_state["y_train"] = data["y_train"]
        session_state["y_test"] = data["y_test"]

    # Pipeline
    session_state["encoders"] = data.get("encoders", {})
    session_state["scaler"] = data.get("scaler")
    session_state["scaled_columns"] = data.get("scaled_columns", [])
    session_state["fe_operations"] = data.get("fe_operations", [])

    # Modèle
    if data.get("meilleur_modele"):
        session_state["meilleur_modele"] = data["meilleur_modele"]
    if data.get("resultats_modeles"):
        session_state["resultats_modeles"] = data["resultats_modeles"]
    if data.get("opt_result"):
        session_state["opt_result"] = data["opt_result"]

    # Drapeaux de progression
    for flag, value in data.get("flags", {}).items():
        session_state[flag] = value

    # Série temporelle
    if data.get("ts_datetime_col"):
        session_state["ts_datetime_col"] = data["ts_datetime_col"]
    if data.get("ts_value_col"):
        session_state["ts_value_col"] = data["ts_value_col"]
    if data.get("ts_series") is not None:
        session_state["ts_series"] = data["ts_series"]

    return data.get("nom", "Projet")
