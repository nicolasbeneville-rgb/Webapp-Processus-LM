# -*- coding: utf-8 -*-
"""
api_server.py — API FastAPI pour servir les modèles ML Studio.

Endpoints :
  GET  /              → Info serveur
  GET  /models        → Liste des modèles disponibles
  POST /predict       → Prédiction (un modèle, données brutes)
  GET  /health        → Healthcheck
  GET  /docs          → Documentation Swagger auto-générée

Déploiement :
  - Local : python api_server.py
  - Cloud Run : via Dockerfile
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
PORT = int(os.environ.get("PORT", 8080))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-studio-api")

app = FastAPI(
    title="ML Studio API",
    description="API de prédiction pour les modèles entraînés avec ML Studio",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# Cache des modèles chargés
# ═══════════════════════════════════════════════════════════

_loaded_models = {}


def _load_model(model_name: str) -> dict:
    """Charge un modèle et son pipeline depuis le dossier models/."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    model_dir = Path(MODELS_DIR) / model_name
    if not model_dir.is_dir():
        raise HTTPException(404, f"Modèle '{model_name}' non trouvé. "
                            f"Dossiers disponibles : {_list_model_names()}")

    # Pipeline JSON (obligatoire)
    pipeline_path = model_dir / "pipeline.json"
    if not pipeline_path.exists():
        raise HTTPException(500, f"pipeline.json manquant pour '{model_name}'")
    with open(pipeline_path, "r", encoding="utf-8") as f:
        pipeline = json.load(f)

    # Modèle joblib (obligatoire)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise HTTPException(500, f"model.joblib manquant pour '{model_name}'")
    model = joblib.load(model_path)

    # Scaler (optionnel)
    scaler = None
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Encoders (optionnel)
    encoders = {}
    encoders_path = model_dir / "encoders.pkl"
    if encoders_path.exists():
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)

    loaded = {
        "model": model,
        "pipeline": pipeline,
        "scaler": scaler,
        "encoders": encoders,
    }
    _loaded_models[model_name] = loaded
    logger.info(f"Modèle '{model_name}' chargé avec succès")
    return loaded


def _list_model_names() -> list:
    """Liste les noms de modèles disponibles."""
    models_path = Path(MODELS_DIR)
    if not models_path.is_dir():
        return []
    return [d.name for d in models_path.iterdir()
            if d.is_dir() and (d / "pipeline.json").exists()]


# ═══════════════════════════════════════════════════════════
# Replay des transformations
# ═══════════════════════════════════════════════════════════

def _replay_ts_transforms(df: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    """Rejoue les transformations de séries temporelles sur les données brutes."""
    dt_col = pipeline.get("datetime_col", "")
    if dt_col and dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.sort_values(dt_col).reset_index(drop=True)

    for t in pipeline.get("ts_transforms", []):
        ttype = t.get("type")
        col = t.get("col", "")

        if col and col not in df.columns:
            continue

        if ttype == "lag":
            lag = t["lag"]
            df[t["output"]] = df[col].shift(lag)

        elif ttype == "delta":
            delta = t["delta"]
            df[t["output"]] = df[col] - df[col].shift(delta)

        elif ttype == "lead":
            horizon = t["horizon"]
            agg = t.get("agg", "mean")
            if agg == "mean":
                df[t["output"]] = df[col].shift(-horizon).rolling(horizon).mean()
            elif agg == "sum":
                df[t["output"]] = df[col].shift(-horizon).rolling(horizon).sum()
            elif agg == "max":
                df[t["output"]] = df[col].shift(-horizon).rolling(horizon).max()
            else:
                df[t["output"]] = df[col].shift(-horizon)

        elif ttype == "rolling_mean":
            w = t["window"]
            df[t["output"]] = df[col].rolling(w).mean()

        elif ttype == "rolling_std":
            w = t["window"]
            df[t["output"]] = df[col].rolling(w).std()

        elif ttype == "seasonal_encoding":
            dt = t.get("datetime_col", dt_col)
            if dt and dt in df.columns:
                doy = pd.to_datetime(df[dt]).dt.dayofyear
                df["saison_sin"] = np.sin(2 * np.pi * doy / 365.25)
                df["saison_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def _replay_fe_operations(df: pd.DataFrame, operations: list) -> pd.DataFrame:
    """Rejoue les opérations de feature engineering classiques."""
    for op in operations:
        try:
            op_type = op.get("type")
            if op_type == "combine":
                col_a, col_b = op["col_a"], op["col_b"]
                operation, new_col = op["operation"], op["new_col"]
                if col_a in df.columns and col_b in df.columns:
                    ops = {
                        "sum": df[col_a] + df[col_b],
                        "diff": df[col_a] - df[col_b],
                        "product": df[col_a] * df[col_b],
                        "ratio": df[col_a] / df[col_b].replace(0, np.nan),
                    }
                    if operation in ops:
                        df[new_col] = ops[operation]

            elif op_type == "derive":
                col, func, new_col = op["col"], op["func"], op["new_col"]
                if col in df.columns:
                    min_val = op.get("min_val", 0)
                    funcs = {
                        "square": df[col] ** 2,
                        "sqrt": np.sqrt(df[col].clip(lower=0)),
                        "log": np.log1p(df[col] - min_val) if min_val <= 0 else np.log1p(df[col]),
                        "inv": 1.0 / df[col].replace(0, np.nan),
                        "abs": df[col].abs(),
                    }
                    if func in funcs:
                        df[new_col] = funcs[func]

            elif op_type == "transform_inplace":
                col, func = op["col"], op["func"]
                if col in df.columns:
                    if func == "log":
                        min_val = op.get("min_val", 1)
                        df[col] = np.log1p(df[col] - min_val) if min_val <= 0 else np.log1p(df[col])
                    elif func == "sqrt":
                        df[col] = np.sqrt(df[col].clip(lower=0))
                    elif func == "square":
                        df[col] = df[col] ** 2
        except Exception:
            continue
    return df


def _replay_encoding(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Rejoue l'encodage catégoriel."""
    for col, enc_info in encoders.items():
        if col not in df.columns:
            continue
        enc_type = enc_info.get("type")
        if enc_type == "onehot":
            categories = enc_info.get("categories", [])
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            for cat_col in categories:
                if cat_col not in df.columns:
                    df[cat_col] = 0
        elif enc_type == "label":
            encoder = enc_info.get("encoder")
            if encoder:
                known = set(encoder.classes_)
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in known else -1)
                df[col] = pd.to_numeric(df[col], errors="coerce")
        elif enc_type == "target":
            mapping = enc_info.get("mapping", {})
            if mapping:
                fallback = np.mean(list(mapping.values()))
                df[col] = df[col].map(mapping).fillna(fallback)
    return df


def _replay_scaling(df: pd.DataFrame, scaler, scaled_columns: list) -> pd.DataFrame:
    """Rejoue le scaling."""
    if scaler and scaled_columns:
        cols = [c for c in scaled_columns if c in df.columns]
        if cols:
            df[cols] = scaler.transform(df[cols])
    return df


def _transform_and_predict(data_rows: list, loaded: dict) -> dict:
    """Pipeline complet : données brutes → transformation → prédiction.

    Gère 2 cas :
    1. Données complètes → prédiction directe (régression/classification)
    2. Historique + prévisions futures (lignes avec cible vide)
       → prédiction itérative jour par jour (série temporelle)
    """
    pipeline = loaded["pipeline"]
    model = loaded["model"]
    horizon = pipeline.get("ts_horizon")
    target_original = pipeline.get("target_col_original", "")
    dt_col = pipeline.get("datetime_col", "")
    lead_cols = pipeline.get("colonnes_prevision", [])

    # Créer le DataFrame
    df = pd.DataFrame(data_rows)

    # Convertir les types numériques (sauf date)
    for col in df.columns:
        if col != dt_col:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    if dt_col and dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.sort_values(dt_col).reset_index(drop=True)

    # Détecter les lignes futures (cible vide) vs historiques
    has_future_rows = False
    future_indices = []
    if target_original and target_original in df.columns and horizon and lead_cols:
        # Lignes où la cible est vide/NaN = lignes futures à prédire
        target_series = pd.to_numeric(df[target_original], errors="coerce")
        future_mask = target_series.isna()
        future_indices = df.index[future_mask].tolist()
        has_future_rows = len(future_indices) > 0

    # ═══════════════════════════════════════════════════════
    # Mode itératif : prédiction jour par jour avec prévision météo
    # ═══════════════════════════════════════════════════════
    if has_future_rows:
        predictions = []
        dates = []

        for idx in future_indices:
            # Recalculer les features sur tout le DataFrame à chaque itération
            # (car les lags utilisent les prédictions précédentes)
            df_work = df.copy()

            # 1. Transformations TS
            df_work = _replay_ts_transforms(df_work, pipeline)

            # 2. FE classique
            if pipeline.get("fe_operations"):
                df_work = _replay_fe_operations(df_work, pipeline["fe_operations"])

            # 3. Encodage
            if loaded.get("encoders"):
                df_work = _replay_encoding(df_work, loaded["encoders"])

            # 4. Scaling
            if loaded.get("scaler"):
                df_work = _replay_scaling(df_work, loaded["scaler"],
                                         pipeline.get("scaled_columns", []))

            # 5. Extraire les features pour cette ligne
            feature_cols = pipeline.get("feature_cols_model", [])
            missing = [c for c in feature_cols if c not in df_work.columns]
            if missing:
                return {"error": f"Colonnes manquantes : {missing}"}

            row_features = df_work.loc[idx, feature_cols]

            # Vérifier que les features sont complètes
            if row_features.isna().any():
                nan_cols = [c for c in feature_cols if pd.isna(row_features[c])]
                # Si c'est juste des NaN de bord (pas assez d'historique), on skip
                logger.warning(f"Ligne {idx}: features NaN: {nan_cols}")
                continue

            X_row = pd.DataFrame([row_features], columns=feature_cols)

            # 6. Prédire
            pred = float(model.predict(X_row)[0])

            # 7. Inverser le log si nécessaire
            if pipeline.get("log_applied"):
                pred = float(np.expm1(pred))

            predictions.append(pred)
            if dt_col and dt_col in df.columns:
                d = df.loc[idx, dt_col]
                dates.append(str(d.date()) if hasattr(d, "date") else str(d))
            else:
                dates.append(str(idx))

            # 8. Injecter la prédiction dans le DataFrame pour les lags suivants
            df.loc[idx, target_original] = pred

        return {
            "predictions": predictions,
            "dates": dates,
            "prediction": predictions[-1] if predictions else None,
            "n_predictions": len(predictions),
            "horizon": horizon,
            "model": pipeline.get("nom"),
            "mode": "iteratif",
            "features_used": pipeline.get("feature_cols_model", []),
        }

    # ═══════════════════════════════════════════════════════
    # Mode standard : prédiction directe
    # ═══════════════════════════════════════════════════════

    # 1. Transformations TS
    if pipeline.get("ts_transforms"):
        df = _replay_ts_transforms(df, pipeline)

    # 2. Feature engineering classique
    if pipeline.get("fe_operations"):
        df = _replay_fe_operations(df, pipeline["fe_operations"])

    # 3. Encodage catégoriel
    if loaded.get("encoders"):
        df = _replay_encoding(df, loaded["encoders"])

    # 4. Scaling
    if loaded.get("scaler"):
        df = _replay_scaling(df, loaded["scaler"],
                             pipeline.get("scaled_columns", []))

    # 5. Sélectionner les features
    feature_cols = pipeline.get("feature_cols_model", [])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Colonnes manquantes après transformation : {missing}. "
                f"Colonnes disponibles : {df.columns.tolist()}"}

    df_features = df[feature_cols].copy()
    valid_mask = df_features.notna().all(axis=1)

    if not valid_mask.any():
        return {"error": "Pas assez de données pour calculer les features. "
                f"Fournissez au moins {pipeline.get('historique_requis', 10)} lignes."}

    X = df_features[valid_mask]

    # 6. Prédire
    predictions = model.predict(X).tolist()

    # 7. Inverser le log
    if pipeline.get("log_applied"):
        predictions = [float(np.expm1(p)) for p in predictions]

    return {
        "predictions": predictions,
        "prediction": predictions[-1] if predictions else None,
        "n_predictions": len(predictions),
        "horizon": horizon,
        "model": pipeline.get("nom"),
        "mode": "standard",
        "features_used": feature_cols,
    }


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    model: str
    data: list[dict]
    batch: bool = False


@app.get("/")
def root():
    models = _list_model_names()
    return {
        "service": "ML Studio API",
        "version": "1.0.0",
        "models_available": len(models),
        "models": models,
        "endpoints": ["/predict", "/models", "/health", "/docs"],
    }


@app.get("/models")
def list_models():
    models = []
    for name in _list_model_names():
        try:
            loaded = _load_model(name)
            p = loaded["pipeline"]
            models.append({
                "nom": p.get("nom", name),
                "id": name,
                "problem_type": p.get("problem_type", ""),
                "target": p.get("target_col", ""),
                "colonnes_brutes": p.get("colonnes_brutes", []),
                "colonnes_prevision": p.get("colonnes_prevision", []),
                "historique_requis": p.get("historique_requis", 1),
                "horizon": p.get("ts_horizon"),
                "date_export": p.get("date_export", ""),
            })
        except Exception as e:
            models.append({"nom": name, "error": str(e)})
    return models


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        loaded = _load_model(request.model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur chargement modèle : {e}")

    result = _transform_and_predict(request.data, loaded)

    if "error" in result:
        raise HTTPException(400, result["error"])

    if request.batch:
        return {
            "predictions": result["predictions"],
            "dates": result.get("dates"),
            "n_predictions": result["n_predictions"],
            "model": result["model"],
            "mode": result.get("mode", "standard"),
        }
    else:
        return {
            "prediction": result["prediction"],
            "predictions": result["predictions"],
            "dates": result.get("dates"),
            "horizon": result.get("horizon"),
            "model": result["model"],
            "mode": result.get("mode", "standard"),
            "n_predictions": result["n_predictions"],
        }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(_loaded_models)}


# ═══════════════════════════════════════════════════════════
# Lancement
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info(f"Démarrage ML Studio API sur le port {PORT}")
    logger.info(f"Modèles dans : {MODELS_DIR}")
    logger.info(f"Modèles détectés : {_list_model_names()}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
