# -*- coding: utf-8 -*-
"""
persistence.py — Sauvegarde versionnée par étape et rechargement du projet.

Chaque projet a son propre dossier. À chaque étape, un CSV versionné est créé :
    data/saves/<nom_projet>/
        project.json                          # Config et état du projet
        save_history.json                     # Historique des sauvegardes
        <nom>_etape0_<fichier>_v1.csv         # Fichiers bruts (étape 0)
        <nom>_etape1_<fichier>_v1.csv         # Fichiers typés (étape 1)
        <nom>_etape2_v1.csv                   # Données consolidées (étape 2)
        <nom>_etape4_v1.csv                   # Données préparées (étape 4)
        <nom>_etape6_X_train_v1.csv           # Splits (étape 6)
        models/
            best_model.pkl
            scaler.pkl
            encoders.pkl
"""

import os
import re
import json
import joblib
import pandas as pd
from datetime import datetime


SAVE_DIR = "data/saves"

STEP_LABELS = {
    0: "Données brutes",
    1: "Données typées",
    2: "Données consolidées",
    3: "Audit",
    4: "Données préparées",
    6: "Modélisation (splits)",
    7: "Optimisation",
}


# ═══════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════

def _safe_name(name: str) -> str:
    """Nettoie un nom pour l'utiliser dans un chemin de fichier."""
    return re.sub(r'_+', '_',
        re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    ).strip('_')


def _safe_filename(name: str) -> str:
    """Nettoie un nom de fichier (supprime extensions, caractères spéciaux)."""
    name = re.sub(r'\.(csv|xlsx?|tsv|parquet)(\s|$)', '', name, flags=re.IGNORECASE)
    return _safe_name(name)


def _project_dir(project_name: str) -> str:
    """Retourne le chemin du dossier de sauvegarde pour un projet."""
    return os.path.join(SAVE_DIR, _safe_name(project_name))


def _next_version(project_dir: str, prefix: str) -> int:
    """Trouve le prochain numéro de version pour un préfixe donné."""
    if not os.path.isdir(project_dir):
        return 1
    pattern = re.compile(re.escape(prefix) + r'_v(\d+)\.csv$')
    versions = []
    for f in os.listdir(project_dir):
        m = pattern.match(f)
        if m:
            versions.append(int(m.group(1)))
    return max(versions, default=0) + 1


# ═══════════════════════════════════════════════════════════
# SAUVEGARDE DES CSV VERSIONNÉS
# ═══════════════════════════════════════════════════════════

def save_step_csv(project_name: str, step: int, data, label: str = "") -> list:
    """
    Sauvegarde un ou plusieurs CSV versionnés pour une étape.

    Args:
        project_name: Nom du projet.
        step: Numéro de l'étape (0, 1, 2, 4, 6, ...).
        data: DataFrame unique ou dict {nom_fichier: DataFrame}.
        label: Description de la sauvegarde.

    Returns:
        Liste des chemins sauvegardés.
    """
    pdir = _project_dir(project_name)
    os.makedirs(pdir, exist_ok=True)
    sname = _safe_name(project_name)

    saved = []
    if isinstance(data, dict):
        for fname, df in data.items():
            sfname = _safe_filename(fname)
            prefix = f"{sname}_etape{step}_{sfname}"
            version = _next_version(pdir, prefix)
            filename = f"{prefix}_v{version}.csv"
            filepath = os.path.join(pdir, filename)
            df.to_csv(filepath, index=False)
            saved.append(filepath)
    elif isinstance(data, pd.DataFrame):
        prefix = f"{sname}_etape{step}"
        version = _next_version(pdir, prefix)
        filename = f"{prefix}_v{version}.csv"
        filepath = os.path.join(pdir, filename)
        data.to_csv(filepath, index=False)
        saved.append(filepath)

    if saved:
        _append_history(pdir, step, label, saved)
    return saved


def get_step_versions(project_name: str, step: int) -> list:
    """
    Liste toutes les versions CSV d'une étape.

    Returns:
        Liste de dicts {filename, filepath, version, suffix, size, modified}.
    """
    pdir = _project_dir(project_name)
    sname = _safe_name(project_name)
    prefix = f"{sname}_etape{step}"

    if not os.path.isdir(pdir):
        return []

    pattern = re.compile(re.escape(prefix) + r'(?:_(.+?))?_v(\d+)\.csv$')
    versions = []
    for f in sorted(os.listdir(pdir)):
        m = pattern.match(f)
        if m:
            filepath = os.path.join(pdir, f)
            versions.append({
                "filename": f,
                "filepath": filepath,
                "suffix": m.group(1) or "",
                "version": int(m.group(2)),
                "size": os.path.getsize(filepath),
                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)),
            })
    return sorted(versions, key=lambda x: (x["version"], x["suffix"]))


def load_step_csv(project_name: str, step: int, version: int = None):
    """
    Charge le(s) CSV d'une étape (dernière version par défaut).

    Returns:
        DataFrame si fichier unique, dict {suffix: DataFrame} si multi-fichiers.
        None si aucun fichier trouvé.
    """
    all_versions = get_step_versions(project_name, step)
    if not all_versions:
        return None

    if version is not None:
        matching = [v for v in all_versions if v["version"] == version]
    else:
        max_v = max(v["version"] for v in all_versions)
        matching = [v for v in all_versions if v["version"] == max_v]

    if not matching:
        return None

    if len(matching) == 1 and not matching[0]["suffix"]:
        return pd.read_csv(matching[0]["filepath"])
    else:
        return {v["suffix"] or f"fichier_{i}": pd.read_csv(v["filepath"])
                for i, v in enumerate(matching)}


# ═══════════════════════════════════════════════════════════
# MÉTADONNÉES DU PROJET
# ═══════════════════════════════════════════════════════════

def save_project_meta(project_name: str, session_state):
    """Sauvegarde les métadonnées du projet (config, flags, colonnes, etc.)."""
    pdir = _project_dir(project_name)
    os.makedirs(pdir, exist_ok=True)

    meta = {}
    simple_keys = [
        "project_name", "problem_type", "max_missing_pct", "min_score",
        "max_overfit_pct", "project_configured", "files_loaded",
        "typing_done", "consolidation_done", "audit_done",
        "preparation_done", "modeling_done", "optimization_done",
        "target_col", "feature_cols", "quality_score",
        "best_model_name", "scaled_columns", "modification_history",
        "test_size", "random_state", "use_chrono_split", "datetime_col",
        "missing_strategies", "outlier_strategies", "encoding_strategies",
        "normalization_method", "ts_created_cols", "fe_operations",
    ]
    for key in simple_keys:
        val = session_state.get(key)
        if val is not None:
            meta[key] = val

    anomalies = session_state.get("anomalies")
    if anomalies:
        meta["anomalies"] = {
            k: (list(v) if isinstance(v, (set, frozenset)) else v)
            for k, v in anomalies.items()
        }

    # Stocker les noms de fichiers originaux
    raw_dfs = session_state.get("raw_dataframes")
    if raw_dfs:
        meta["original_filenames"] = list(raw_dfs.keys())

    with open(os.path.join(pdir, "project.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def load_project_meta(project_name: str) -> dict:
    """Charge les métadonnées d'un projet (nouveau ou ancien format)."""
    pdir = _project_dir(project_name)
    # Nouveau format : project.json à la racine
    meta_path = os.path.join(pdir, "project.json")
    # Ancien format : meta/project.json
    if not os.path.isfile(meta_path):
        meta_path = os.path.join(pdir, "meta", "project.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════
# MODÈLES ET PRÉPROCESSEURS
# ═══════════════════════════════════════════════════════════

def save_models(project_name: str, session_state):
    """Sauvegarde les modèles et préprocesseurs (joblib)."""
    pdir = _project_dir(project_name)
    model_dir = os.path.join(pdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    for key, fname in [("best_model", "best_model.pkl"),
                        ("scaler", "scaler.pkl"),
                        ("encoders", "encoders.pkl")]:
        obj = session_state.get(key)
        if obj is not None:
            joblib.dump(obj, os.path.join(model_dir, fname))

    best_result = session_state.get("best_model_result")
    if best_result:
        result_copy = {k: v for k, v in best_result.items() if k != "model"}
        joblib.dump(result_copy, os.path.join(model_dir, "best_result.pkl"))

    model_results = session_state.get("model_results")
    if model_results:
        results_copy = [{k: v for k, v in r.items() if k != "model"}
                        for r in model_results]
        joblib.dump(results_copy, os.path.join(model_dir, "all_results.pkl"))


def load_models(project_name: str) -> dict:
    """Charge les modèles et préprocesseurs."""
    pdir = _project_dir(project_name)
    model_dir = os.path.join(pdir, "models")
    state = {}

    for fname, key in [("best_model.pkl", "best_model"),
                        ("scaler.pkl", "scaler"),
                        ("encoders.pkl", "encoders"),
                        ("all_results.pkl", "model_results")]:
        path = os.path.join(model_dir, fname)
        if os.path.isfile(path):
            state[key] = joblib.load(path)

    result_path = os.path.join(model_dir, "best_result.pkl")
    if os.path.isfile(result_path):
        best_result = joblib.load(result_path)
        if "best_model" in state:
            best_result["model"] = state["best_model"]
        state["best_model_result"] = best_result

    return state


# ═══════════════════════════════════════════════════════════
# HISTORIQUE
# ═══════════════════════════════════════════════════════════

def _append_history(project_dir: str, step: int, label: str, files: list):
    """Ajoute une entrée à l'historique des sauvegardes."""
    history_path = os.path.join(project_dir, "save_history.json")
    history = []
    if os.path.isfile(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step,
        "label": label,
        "files": [os.path.basename(f) for f in files],
    })

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# FONCTIONS DE HAUT NIVEAU
# ═══════════════════════════════════════════════════════════

def save_project_state(session_state, label: str = "",
                       step: int = None, data=None) -> str:
    """
    Sauvegarde l'état du projet :
      - Toujours : métadonnées (project.json)
      - Si step+data fournis : CSV versionné pour l'étape
      - Si modèles présents : fichiers .pkl
    """
    project_name = session_state.get("project_name", "projet_sans_nom")

    save_project_meta(project_name, session_state)

    if step is not None and data is not None:
        save_step_csv(project_name, step, data, label=label)

    if (session_state.get("best_model") is not None
            or session_state.get("scaler") is not None):
        save_models(project_name, session_state)

    return _project_dir(project_name)


def load_project_state(project_name: str) -> dict:
    """
    Charge l'état complet d'un projet sauvegardé.
    Retourne un dict à injecter dans st.session_state.
    """
    pdir = _project_dir(project_name)
    if not os.path.isdir(pdir):
        raise FileNotFoundError(
            f"Aucun projet sauvegardé sous « {project_name} ».")

    # 1. Métadonnées
    state = load_project_meta(project_name)
    original_filenames = state.pop("original_filenames", None)

    # 2. DataFrames — essayer le nouveau format (CSV versionnés), sinon ancien format
    def _reconstruct_dict(data, filenames):
        """Reconstruit un dict {nom_original: df}."""
        if isinstance(data, pd.DataFrame):
            fname = (filenames[0] if filenames else "fichier.csv")
            return {fname: data}
        elif isinstance(data, dict):
            if filenames and len(filenames) == len(data):
                return dict(zip(filenames, data.values()))
            return data
        return None

    # Détection du format : ancien = dossier dataframes/ présent
    _old_df_dir = os.path.join(pdir, "dataframes")
    _is_old_format = os.path.isdir(_old_df_dir) and not any(
        f.endswith(".csv") and "_etape" in f for f in os.listdir(pdir)
    )

    if _is_old_format:
        # ── ANCIEN FORMAT (dataframes/*.csv ou *.parquet) ──
        def _load_df_legacy(path_base):
            for ext in [".parquet", ".csv"]:
                p = path_base + ext
                if os.path.isfile(p):
                    return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            return None

        # raw_dataframes
        raw_dfs = {}
        for fname in os.listdir(_old_df_dir):
            if fname.startswith("raw__") and (fname.endswith(".parquet") or fname.endswith(".csv")):
                ext_len = 8 if fname.endswith(".parquet") else 4
                original_name = fname[5:-(ext_len)]
                if original_name not in raw_dfs:
                    df = _load_df_legacy(os.path.join(_old_df_dir, f"raw__{original_name}"))
                    if df is not None:
                        raw_dfs[original_name] = df
        if raw_dfs:
            state["raw_dataframes"] = raw_dfs

        # typed_dataframes
        typed_dfs = {}
        for fname in os.listdir(_old_df_dir):
            if fname.startswith("typed__") and (fname.endswith(".parquet") or fname.endswith(".csv")):
                ext_len = 8 if fname.endswith(".parquet") else 4
                original_name = fname[7:-(ext_len)]
                if original_name not in typed_dfs:
                    df = _load_df_legacy(os.path.join(_old_df_dir, f"typed__{original_name}"))
                    if df is not None:
                        typed_dfs[original_name] = df
        if typed_dfs:
            state["typed_dataframes"] = typed_dfs

        # consolidated_df
        df = _load_df_legacy(os.path.join(_old_df_dir, "consolidated"))
        if df is not None:
            state["consolidated_df"] = df

        # prepared_df
        df = _load_df_legacy(os.path.join(_old_df_dir, "prepared"))
        if df is not None:
            state["prepared_df"] = df

        # Train/test splits
        for key in ["X_train", "X_test"]:
            df = _load_df_legacy(os.path.join(_old_df_dir, key))
            if df is not None:
                state[key] = df.values
        for key in ["y_train", "y_test"]:
            df = _load_df_legacy(os.path.join(_old_df_dir, key))
            if df is not None:
                state[key] = df.iloc[:, 0].values

        # Préprocesseurs (ancien format)
        _old_prep_dir = os.path.join(pdir, "preprocessors")
        for fname, key in [("scaler.pkl", "scaler"), ("encoders.pkl", "encoders")]:
            path = os.path.join(_old_prep_dir, fname)
            if os.path.isfile(path):
                state[key] = joblib.load(path)

    else:
        # ── NOUVEAU FORMAT (CSV versionnés) ──
        # Étape 0 : raw_dataframes
        raw = load_step_csv(project_name, 0)
        if raw is not None:
            result = _reconstruct_dict(raw, original_filenames)
            if result:
                state["raw_dataframes"] = result

        # Étape 1 : typed_dataframes
        typed = load_step_csv(project_name, 1)
        if typed is not None:
            result = _reconstruct_dict(typed, original_filenames)
            if result:
                state["typed_dataframes"] = result

        # Étape 2 : consolidated_df
        consolidated = load_step_csv(project_name, 2)
        if consolidated is not None:
            if isinstance(consolidated, pd.DataFrame):
                state["consolidated_df"] = consolidated
            else:
                state["consolidated_df"] = list(consolidated.values())[0]

        # Étape 4 : prepared_df
        prepared = load_step_csv(project_name, 4)
        if prepared is not None:
            if isinstance(prepared, pd.DataFrame):
                state["prepared_df"] = prepared
            else:
                state["prepared_df"] = list(prepared.values())[0]

        # Étape 6 : splits train/test
        splits = load_step_csv(project_name, 6)
        if splits is not None and isinstance(splits, dict):
            for key in ["X_train", "X_test"]:
                if key in splits:
                    state[key] = splits[key].values
            for key in ["y_train", "y_test"]:
                if key in splits:
                    state[key] = splits[key].iloc[:, 0].values

    # 3. Modèles
    state.update(load_models(project_name))

    return state


def list_saved_projects() -> list:
    """Liste tous les projets sauvegardés."""
    projects = []
    if not os.path.isdir(SAVE_DIR):
        return projects

    for folder in sorted(os.listdir(SAVE_DIR)):
        project_path = os.path.join(SAVE_DIR, folder)
        if not os.path.isdir(project_path):
            continue

        # Chercher project.json (nouveau format : racine, ancien : meta/)
        meta_path = os.path.join(project_path, "project.json")
        if not os.path.isfile(meta_path):
            meta_path = os.path.join(project_path, "meta", "project.json")

        name = folder
        last_save = ""
        steps_done = 0
        last_label = ""
        save_history = []

        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                name = meta.get("project_name", folder)
                step_flags = [
                    "project_configured", "files_loaded", "typing_done",
                    "consolidation_done", "audit_done", "preparation_done",
                    "modeling_done", "optimization_done",
                ]
                steps_done = sum(1 for k in step_flags if meta.get(k))
            except (json.JSONDecodeError, OSError):
                pass

        # Historique (nouveau format : racine, ancien : meta/)
        history_path = os.path.join(project_path, "save_history.json")
        if not os.path.isfile(history_path):
            history_path = os.path.join(project_path, "meta", "save_history.json")
        if os.path.isfile(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    save_history = json.load(f)
                if save_history:
                    last_save = save_history[-1].get("timestamp", "")
                    last_label = save_history[-1].get("label", "")
            except (json.JSONDecodeError, OSError):
                pass

        projects.append({
            "name": name,
            "folder": folder,
            "last_save": last_save,
            "steps_done": steps_done,
            "last_label": last_label,
            "save_history": save_history,
        })

    return projects


def list_project_files(project_name: str) -> list:
    """
    Liste tous les fichiers d'un projet, organisés par catégorie.

    Returns:
        Liste de dicts {name, path, size, category, step}.
    """
    pdir = _project_dir(project_name)
    if not os.path.isdir(pdir):
        return []

    sname = _safe_name(project_name)
    step_pattern = re.compile(re.escape(sname) + r'_etape(\d+)')
    files = []

    # Fichiers racine du projet
    for f in sorted(os.listdir(pdir)):
        fpath = os.path.join(pdir, f)
        if not os.path.isfile(fpath):
            continue
        if f.endswith(".csv"):
            m = step_pattern.match(f)
            step_num = int(m.group(1)) if m else -1
            files.append({
                "name": f, "path": fpath,
                "size": os.path.getsize(fpath),
                "category": "data", "step": step_num,
            })
        elif f.endswith(".json"):
            files.append({
                "name": f, "path": fpath,
                "size": os.path.getsize(fpath),
                "category": "meta", "step": -1,
            })

    # Sous-dossier models/
    model_dir = os.path.join(pdir, "models")
    if os.path.isdir(model_dir):
        for f in sorted(os.listdir(model_dir)):
            fpath = os.path.join(model_dir, f)
            if os.path.isfile(fpath):
                files.append({
                    "name": f, "path": fpath,
                    "size": os.path.getsize(fpath),
                    "category": "model", "step": -1,
                })

    return files


def delete_project(project_name: str):
    """Supprime un projet sauvegardé."""
    import shutil
    project_dir = _project_dir(project_name)
    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir)
