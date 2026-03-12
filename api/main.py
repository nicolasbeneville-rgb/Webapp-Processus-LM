# -*- coding: utf-8 -*-
"""
API de prédiction ML Studio — FastAPI pour Google Cloud Run.

Charge les modèles exportés depuis ML Studio et expose des endpoints REST.
Déployable sur Google Cloud Run.

Usage local :
    uvicorn api.main:app --reload --port 8080
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "api/models"))

app = FastAPI(
    title="ML Studio — API de prédiction",
    description="API REST pour utiliser les modèles entraînés dans ML Studio.",
    version="1.0.0",
)

# CORS ouvert (nécessaire pour AppScript / navigateur)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# Cache des modèles chargés
# ═══════════════════════════════════════════════════════════
_loaded_models: dict = {}


def _load_model(model_id: str) -> dict:
    """Charge un modèle et ses métadonnées depuis le dossier models/."""
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    model_dir = MODELS_DIR / model_id
    if not model_dir.is_dir():
        raise HTTPException(status_code=404,
                            detail=f"Modèle '{model_id}' introuvable.")

    # Charger les métadonnées
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=500,
                            detail=f"metadata.json manquant pour '{model_id}'.")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Charger le modèle
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=500,
                            detail=f"model.joblib manquant pour '{model_id}'.")
    model = joblib.load(model_path)

    # Charger le scaler (optionnel)
    scaler = None
    scaler_path = model_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    entry = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
    }
    _loaded_models[model_id] = entry
    return entry


# ═══════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    model: str
    features: dict[str, Any]


class PredictBatchRequest(BaseModel):
    model: str
    data: list[dict[str, Any]]


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════
@app.get("/")
def root():
    """Page d'accueil."""
    return {
        "service": "ML Studio — API de prédiction",
        "version": "1.0.0",
        "endpoints": {
            "GET /models": "Liste des modèles disponibles",
            "GET /models/{model_id}": "Détails d'un modèle",
            "POST /predict": "Prédiction unitaire",
            "POST /predict/batch": "Prédiction par lot",
        },
    }


@app.get("/models")
def list_models():
    """Liste tous les modèles disponibles."""
    if not MODELS_DIR.exists():
        return {"models": []}

    models = []
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir() and (d / "metadata.json").exists():
            with open(d / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            models.append({
                "id": d.name,
                "name": meta.get("project_name", d.name),
                "model_type": meta.get("model_name", "?"),
                "problem_type": meta.get("problem_type", "?"),
                "features": meta.get("feature_names", []),
                "score": meta.get("test_score"),
            })
    return {"models": models}


@app.get("/models/{model_id}")
def model_info(model_id: str):
    """Détails d'un modèle : features attendues, types, score, etc."""
    entry = _load_model(model_id)
    meta = entry["metadata"]
    return {
        "id": model_id,
        "name": meta.get("project_name", model_id),
        "model_name": meta.get("model_name", "?"),
        "problem_type": meta.get("problem_type", "?"),
        "test_score": meta.get("test_score"),
        "features": meta.get("feature_names", []),
        "feature_types": meta.get("feature_types", {}),
        "target": meta.get("target_name", "?"),
        "has_scaler": entry["scaler"] is not None,
        "created_at": meta.get("created_at"),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Prédiction unitaire. Envoie un dict de features, reçoit la prédiction."""
    entry = _load_model(req.model)
    model = entry["model"]
    scaler = entry["scaler"]
    meta = entry["metadata"]
    feature_names = meta.get("feature_names", [])

    # Valider les features
    missing = [f for f in feature_names if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Features manquantes : {missing}. "
                   f"Attendues : {feature_names}")

    # Construire le DataFrame
    df = pd.DataFrame([{f: req.features[f] for f in feature_names}])

    # Appliquer le scaler si présent
    X = df.values
    if scaler is not None:
        X = scaler.transform(X)

    # Prédire
    prediction = model.predict(X)
    result = {"prediction": _to_serializable(prediction[0])}

    # Probabilités si classification
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        result["probabilities"] = {
            str(c): round(float(p), 4)
            for c, p in zip(model.classes_, probas)
        }

    return result


@app.post("/predict/batch")
def predict_batch(req: PredictBatchRequest):
    """Prédiction par lot. Envoie une liste de dicts, reçoit les prédictions."""
    entry = _load_model(req.model)
    model = entry["model"]
    scaler = entry["scaler"]
    meta = entry["metadata"]
    feature_names = meta.get("feature_names", [])

    if not req.data:
        raise HTTPException(status_code=422, detail="data est vide.")

    # Construire le DataFrame
    df = pd.DataFrame(req.data)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Features manquantes : {missing}")

    X = df[feature_names].values
    if scaler is not None:
        X = scaler.transform(X)

    predictions = model.predict(X)
    results = [_to_serializable(p) for p in predictions]

    response = {"predictions": results, "count": len(results)}

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        response["probabilities"] = [
            {str(c): round(float(p), 4) for c, p in zip(model.classes_, row)}
            for row in probas
        ]

    return response


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def _to_serializable(val):
    """Convertit les types numpy en types Python natifs pour JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val
