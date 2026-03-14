# -*- coding: utf-8 -*-
"""
export_api.py — Export d'un modèle ML Studio pour déploiement API.

Génère un package complet :
  - model.joblib          : modèle entraîné
  - pipeline.json         : recette de transformation (lags, rolling, etc.)
  - scaler.pkl            : scaler sklearn (si utilisé)
  - encoders.pkl          : encodeurs catégoriels (si utilisés)
  - template.csv          : trame vide avec colonnes brutes attendues
  - appscript.js          : code Google Apps Script prêt à copier
  - README_DEPLOY.md      : guide de déploiement Cloud Run
"""

import io
import json
import os
import zipfile
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime


def build_pipeline_json(session_state: dict) -> dict:
    """Construit le pipeline.json décrivant toute la chaîne de transformation."""
    rapport = session_state.get("rapport", {})
    nettoyage = rapport.get("nettoyage", {})
    ts_horizon = nettoyage.get("ts_horizon", {})

    # Colonnes brutes d'origine (avant toute transformation)
    raw_cols = []
    if ts_horizon:
        # En mode TS horizon, les colonnes brutes sont celles d'avant les features
        dt_col = session_state.get("ts_datetime_col", "")
        val_col = session_state.get("ts_value_col", "")
        # Reconstituer les colonnes brutes à partir des lags/rolling/leads
        all_source_cols = set()
        if dt_col:
            all_source_cols.add(dt_col)
        if val_col:
            all_source_cols.add(val_col)
        for col in ts_horizon.get("lag_cols", []):
            all_source_cols.add(col)
        for col in ts_horizon.get("rolling_cols", []):
            all_source_cols.add(col)
        for col in ts_horizon.get("lead_cols", []):
            all_source_cols.add(col)
        delta_col = ts_horizon.get("delta_col")
        if delta_col:
            all_source_cols.add(delta_col)
        raw_cols = sorted(all_source_cols)
    else:
        # Mode classique : colonnes features + cible
        feature_cols = session_state.get("feature_cols", [])
        target = session_state.get("target_col", "")
        raw_cols = list(set(feature_cols + ([target] if target else [])))

    # Transformations TS
    ts_transforms = []
    if ts_horizon:
        horizon = ts_horizon.get("horizon", 1)
        # Lags
        for col in ts_horizon.get("lag_cols", []):
            for lag in ts_horizon.get("lags", []):
                ts_transforms.append({
                    "type": "lag", "col": col, "lag": lag,
                    "output": f"{col}_lag{lag}"
                })
        # Deltas
        delta_col = ts_horizon.get("delta_col")
        if delta_col:
            for d in ts_horizon.get("deltas", []):
                ts_transforms.append({
                    "type": "delta", "col": delta_col, "delta": d,
                    "output": f"{delta_col}_delta{d}"
                })
        # Leads
        lead_agg = ts_horizon.get("lead_agg", "mean")
        for col in ts_horizon.get("lead_cols", []):
            ts_transforms.append({
                "type": "lead", "col": col, "horizon": horizon,
                "agg": lead_agg,
                "output": f"{col}_lead{horizon}_{lead_agg}"
            })
        # Rolling
        for col in ts_horizon.get("rolling_cols", []):
            for w in ts_horizon.get("rolling_windows", []):
                ts_transforms.append({
                    "type": "rolling_mean", "col": col, "window": w,
                    "output": f"{col}_rmean{w}"
                })
                ts_transforms.append({
                    "type": "rolling_std", "col": col, "window": w,
                    "output": f"{col}_rstd{w}"
                })
        # Seasonal encoding
        if ts_horizon.get("seasonal_encoding"):
            dt_col = session_state.get("ts_datetime_col", "")
            ts_transforms.append({
                "type": "seasonal_encoding", "datetime_col": dt_col,
                "outputs": ["saison_sin", "saison_cos"]
            })

    # FE classiques
    fe_operations = session_state.get("fe_operations", [])

    # Scaling info
    scaled_columns = session_state.get("scaled_columns", [])
    scaler_type = ""
    scaler = session_state.get("scaler")
    if scaler:
        scaler_type = type(scaler).__name__

    # Colonnes finales attendues par le modèle
    feature_cols_used = session_state.get("feature_cols_used",
                                          session_state.get("feature_cols", []))

    # Historique requis (plus grand lag/rolling)
    max_lookback = 1
    if ts_horizon:
        all_lags = ts_horizon.get("lags", [])
        all_windows = ts_horizon.get("rolling_windows", [])
        all_deltas = ts_horizon.get("deltas", [])
        max_lookback = max(
            max(all_lags, default=0),
            max(all_windows, default=0),
            max(all_deltas, default=0),
            ts_horizon.get("horizon", 1)
        ) + 2  # marge de sécurité

    # Colonnes de prévision future (lead_cols = variables exogènes à fournir)
    lead_cols = ts_horizon.get("lead_cols", []) if ts_horizon else []

    pipeline = {
        "format_version": 2,
        "nom": rapport.get("nom", "modele"),
        "date_export": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "problem_type": session_state.get("problem_type", ""),
        "target_col": session_state.get("target_col", ""),
        "target_col_original": session_state.get("ts_value_col", ""),
        "datetime_col": session_state.get("ts_datetime_col", ""),
        "colonnes_brutes": raw_cols,
        "colonnes_prevision": lead_cols,  # colonnes à fournir pour le futur
        "feature_cols_model": feature_cols_used,
        "ts_horizon": ts_horizon.get("horizon") if ts_horizon else None,
        "ts_transforms": ts_transforms,
        "fe_operations": fe_operations,
        "scaler_type": scaler_type,
        "scaled_columns": scaled_columns,
        "historique_requis": max_lookback,
        "log_applied": bool(session_state.get("ts_log_applied")),
        "log_applied_cols": session_state.get("ts_log_applied_cols", []),
    }
    return pipeline


def generate_template_csv(pipeline: dict) -> str:
    """Génère un CSV template avec section historique + prévision future.

    Si le modèle utilise des lead features (prévisions météo futures),
    le template inclut des lignes futures avec les colonnes exogènes
    à remplir et la cible vide.
    """
    cols = pipeline.get("colonnes_brutes", [])
    n_history = max(pipeline.get("historique_requis", 10), 5)
    horizon = pipeline.get("ts_horizon")
    lead_cols = pipeline.get("colonnes_prevision", [])
    target_original = pipeline.get("target_col_original", "")
    dt_col = pipeline.get("datetime_col", "")

    # Générer des dates exemples
    from datetime import datetime, timedelta
    today = datetime.now().date()

    rows = []

    # Lignes historiques (toutes les colonnes remplies avec des exemples)
    for i in range(n_history):
        row = {}
        d = today - timedelta(days=n_history - i)
        for c in cols:
            if c == dt_col:
                row[c] = d.strftime("%Y-%m-%d")
            elif c == target_original:
                row[c] = f"<mesure_{c}>"
            else:
                row[c] = f"<valeur_{c}>"
        rows.append(row)

    # Lignes futures (si horizon + lead_cols)
    if horizon and lead_cols:
        for i in range(1, horizon + 1):
            row = {}
            d = today + timedelta(days=i)
            for c in cols:
                if c == dt_col:
                    row[c] = d.strftime("%Y-%m-%d")
                elif c in lead_cols:
                    row[c] = f"<prevision_{c}>"
                elif c == target_original:
                    row[c] = ""  # vide = à prédire
                else:
                    row[c] = ""  # pas nécessaire pour le futur
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def generate_appscript(pipeline: dict, api_url: str = "https://YOUR-API-URL.run.app") -> str:
    """Génère le code Google Apps Script pour appeler l'API."""
    model_name = pipeline.get("nom", "modele").replace(" ", "_").lower()
    raw_cols = pipeline.get("colonnes_brutes", [])
    n_history = pipeline.get("historique_requis", 14)
    target = pipeline.get("target_col_original") or pipeline.get("target_col", "")

    lead_cols = pipeline.get("colonnes_prevision", [])
    horizon = pipeline.get("ts_horizon")
    has_leads = bool(lead_cols and horizon)

    lead_info = ""
    if has_leads:
        lead_info = (
            f"\n * PRÉVISION FUTURE :"
            f"\n * Pour prédire dans le futur, ajoutez des lignes avec :"
            f"\n *   - La date future"
            f"\n *   - Les prévisions pour : {', '.join(lead_cols)}"
            f"\n *   - La colonne {target} vide (c'est ce que l'API va prédire)"
            f"\n * L'API prédit jour par jour en utilisant ses propres prédictions"
            f"\n * comme historique pour les jours suivants."
        )

    code = f'''// ═══════════════════════════════════════════════════════
// Code AppScript généré par ML Studio
// Modèle : {pipeline.get("nom", "modele")}
// Date : {pipeline.get("date_export", "")}
//
// COLONNES REQUISES : {", ".join(raw_cols)}
// HISTORIQUE MINIMUM : {n_history} lignes{f"""
// COLONNES DE PRÉVISION FUTURE : {", ".join(lead_cols)}""" if has_leads else ""}
// ═══════════════════════════════════════════════════════

const API_URL = "{api_url}";
const MODEL_NAME = "{model_name}";

/**
 * Prédit la valeur de {target}.
 *{lead_info}
 * Utilisation :
 *   1. Remplissez le tableau avec les {n_history}+ dernières mesures
 *{f'   2. Ajoutez les lignes futures avec les prévisions météo ({", ".join(lead_cols)})' if has_leads else '   2. Le modèle prédit la dernière ligne'}
 *   3. Menu ⚗️ ML Studio > Prédire
 */
function predire() {{
  const sheet = SpreadsheetApp.getActiveSheet();
  const data = sheet.getDataRange().getValues();
  const headers = data[0];
  const allRows = data.slice(1);

  // Construire le payload avec TOUTES les lignes (historique + prévisions)
  const rows = allRows.map(row => {{
    const obj = {{}};
    headers.forEach((h, i) => {{
      // Convertir les dates en string ISO
      if (row[i] instanceof Date) {{
        obj[h] = row[i].toISOString().split("T")[0];
      }} else {{
        obj[h] = (row[i] === "" || row[i] === null) ? null : row[i];
      }}
    }});
    return obj;
  }});

  const payload = {{
    "model": MODEL_NAME,
    "data": rows
  }};

  const options = {{
    "method": "post",
    "contentType": "application/json",
    "payload": JSON.stringify(payload),
    "muteHttpExceptions": true
  }};

  try {{
    const response = UrlFetchApp.fetch(API_URL + "/predict", options);
    const result = JSON.parse(response.getContentText());

    if (result.detail) {{
      SpreadsheetApp.getUi().alert("Erreur : " + result.detail);
      return;
    }}

    const predictions = result.predictions || [result.prediction];
    const dates = result.dates || [];
    const mode = result.mode || "standard";

    if (mode === "iteratif" && predictions.length > 1) {{
      // Mode prévision : écrire les prédictions dans les cellules vides
      const targetCol = headers.indexOf("{target}");
      if (targetCol >= 0) {{
        let written = 0;
        for (let i = 0; i < allRows.length; i++) {{
          const val = allRows[i][targetCol];
          if (val === "" || val === null || val === undefined) {{
            if (written < predictions.length) {{
              const cell = sheet.getRange(i + 2, targetCol + 1);
              cell.setValue(Math.round(predictions[written] * 10000) / 10000);
              cell.setBackground("#E8F5E9");
              cell.setFontStyle("italic");
              cell.setNote("Prédit par ML Studio le " +
                           new Date().toLocaleDateString());
              written++;
            }}
          }}
        }}
        SpreadsheetApp.getUi().alert(
          "Prédiction ML Studio",
          written + " prédictions écrites dans la colonne {target} !\\n" +
          "Mode : itératif (jour par jour)",
          SpreadsheetApp.getUi().ButtonSet.OK
        );
      }}
    }} else {{
      // Mode simple : afficher la dernière prédiction
      const prediction = predictions[predictions.length - 1];
      SpreadsheetApp.getUi().alert(
        "Prédiction ML Studio",
        "{target} prédit : " + prediction.toFixed(4) +
        (result.horizon ? " (horizon t+" + result.horizon + ")" : ""),
        SpreadsheetApp.getUi().ButtonSet.OK
      );
    }}

  }} catch (e) {{
    SpreadsheetApp.getUi().alert("Erreur de connexion : " + e.message);
  }}
}}


/**
 * Liste les modèles disponibles sur l'API.
 */
function listerModeles() {{
  try {{
    const response = UrlFetchApp.fetch(API_URL + "/models");
    const models = JSON.parse(response.getContentText());

    let msg = "Modèles disponibles :\\n\\n";
    models.forEach(m => {{
      msg += "• " + m.nom + " (" + m.problem_type + ")";
      if (m.colonnes_prevision && m.colonnes_prevision.length > 0) {{
        msg += "\\n  Prévisions requises : " + m.colonnes_prevision.join(", ");
      }}
      msg += "\\n";
    }});

    SpreadsheetApp.getUi().alert("ML Studio - Modèles", msg,
                                  SpreadsheetApp.getUi().ButtonSet.OK);
  }} catch (e) {{
    SpreadsheetApp.getUi().alert("Erreur : " + e.message);
  }}
}}


/**
 * Ajoute le menu ML Studio dans Google Sheets.
 */
function onOpen() {{
  SpreadsheetApp.getUi()
    .createMenu("⚗️ ML Studio")
    .addItem("🔮 Prédire", "predire")
    .addSeparator()
    .addItem("📋 Lister les modèles", "listerModeles")
    .addToUi();
}}
'''
    return code


def generate_deploy_readme(pipeline: dict) -> str:
    """Génère le guide de déploiement Cloud Run."""
    model_name = pipeline.get("nom", "modele").replace(" ", "_").lower()

    return f"""# Déployer l'API ML Studio sur Google Cloud Run

## Prérequis
- Un compte Google Cloud (gratuit : 300$ de crédits)
- Google Cloud CLI installé (`gcloud`)

## Étapes

### 1. Préparer le projet
```bash
# Créer un dossier pour l'API
mkdir ml-studio-api && cd ml-studio-api

# Copier les fichiers exportés (model.joblib, pipeline.json, etc.)
# dans un dossier models/{model_name}/
mkdir -p models/{model_name}
cp /chemin/vers/export/* models/{model_name}/
```

### 2. Copier les fichiers serveur
Copier `api_server.py`, `requirements_api.txt` et `Dockerfile` dans le dossier.

### 3. Tester en local
```bash
pip install -r requirements_api.txt
python api_server.py
# Ouvrir http://localhost:8080/docs pour tester
```

### 4. Déployer sur Cloud Run
```bash
# Se connecter
gcloud auth login
gcloud config set project VOTRE_PROJET_ID

# Construire et déployer
gcloud run deploy ml-studio-api \\
  --source . \\
  --region europe-west1 \\
  --allow-unauthenticated \\
  --memory 512Mi \\
  --timeout 60
```

### 5. Configurer AppScript
1. Ouvrir Google Sheets
2. Extensions > Apps Script
3. Coller le contenu de `appscript.js`
4. Remplacer `YOUR-API-URL` par l'URL Cloud Run
5. Sauvegarder et autoriser

### 6. Utiliser
- Menu ⚗️ ML Studio apparaît dans la barre de menu
- Remplir le tableau avec vos données brutes
- Cliquer sur Prédire !

## Coût
Google Cloud Run est gratuit jusqu'à :
- 2 millions de requêtes/mois
- 360 000 Go-secondes de mémoire
- 180 000 vCPU-secondes

Pour un usage normal, c'est **gratuit**.
"""


def export_model_package(session_state: dict) -> bytes:
    """Exporte un ZIP complet contenant tout le nécessaire pour l'API.

    Returns:
        bytes du fichier ZIP
    """
    pipeline = build_pipeline_json(session_state)
    model_name = pipeline["nom"].replace(" ", "_").lower()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. pipeline.json
        zf.writestr(f"{model_name}/pipeline.json",
                     json.dumps(pipeline, indent=2, ensure_ascii=False))

        # 2. model.joblib
        best = session_state.get("meilleur_modele", {})
        if best and best.get("model"):
            model_buf = io.BytesIO()
            joblib.dump(best["model"], model_buf)
            zf.writestr(f"{model_name}/model.joblib", model_buf.getvalue())

        # 3. scaler.pkl
        scaler = session_state.get("scaler")
        if scaler is not None:
            scaler_buf = io.BytesIO()
            pickle.dump(scaler, scaler_buf)
            zf.writestr(f"{model_name}/scaler.pkl", scaler_buf.getvalue())

        # 4. encoders.pkl
        encoders = session_state.get("encoders")
        if encoders:
            enc_buf = io.BytesIO()
            pickle.dump(encoders, enc_buf)
            zf.writestr(f"{model_name}/encoders.pkl", enc_buf.getvalue())

        # 5. template.csv
        template = generate_template_csv(pipeline)
        zf.writestr(f"{model_name}/template.csv", template)

        # 6. appscript.js
        appscript = generate_appscript(pipeline)
        zf.writestr(f"{model_name}/appscript.js", appscript)

        # 7. README
        readme = generate_deploy_readme(pipeline)
        zf.writestr(f"{model_name}/README_DEPLOY.md", readme)

        # 8. api_server.py (copie du serveur FastAPI)
        api_server_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "api", "api_server.py")
        if os.path.exists(api_server_path):
            with open(api_server_path, "r", encoding="utf-8") as f:
                zf.writestr("api_server.py", f.read())

        # 9. requirements_api.txt
        api_req_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "api", "requirements_api.txt")
        if os.path.exists(api_req_path):
            with open(api_req_path, "r", encoding="utf-8") as f:
                zf.writestr("requirements_api.txt", f.read())

        # 10. Dockerfile
        api_docker_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "api", "Dockerfile")
        if os.path.exists(api_docker_path):
            with open(api_docker_path, "r", encoding="utf-8") as f:
                zf.writestr("Dockerfile", f.read())

    return buf.getvalue()
