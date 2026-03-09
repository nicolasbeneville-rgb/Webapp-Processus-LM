# -*- coding: utf-8 -*-
"""
model_utils.py — Fonctions utilitaires pour la modélisation.

Wraps autour de src/models.py et src/evaluation.py pour fournir
des interfaces simplifiées utilisées par les modules.
"""

import numpy as np
import pandas as pd


def comparer_modeles(resultats: list, problem_type: str) -> pd.DataFrame:
    """Crée un tableau comparatif des modèles pour affichage."""
    rows = []
    for r in resultats:
        row = {
            "Modèle": r.get("name", "?"),
            "Score test": f"{r.get('test_score', 0):.4f}",
            "Score train": f"{r.get('train_score', 0):.4f}",
        }
        if problem_type == "Régression":
            row["RMSE"] = f"{r.get('rmse', 0):.4f}" if r.get("rmse") else "—"
            row["MAE"] = f"{r.get('mae', 0):.4f}" if r.get("mae") else "—"
        else:
            row["F1"] = f"{r.get('f1', 0):.4f}" if r.get("f1") else "—"

        ecart = abs(r.get("train_score", 0) - r.get("test_score", 0))
        row["Écart train/test"] = f"{ecart:.4f}"
        row["Temps (s)"] = f"{r.get('training_time', 0):.1f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def replay_pipeline(new_df: pd.DataFrame, session: dict) -> tuple:
    """Rejoue tout le pipeline (FE, encodage, scaling) sur de nouvelles données.

    Args:
        new_df: DataFrame avec les mêmes colonnes brutes
        session: dict contenant encoders, scaler, scaled_columns, fe_operations

    Returns:
        (DataFrame transformé, liste de descriptions)
    """
    transforms = []

    # 1. Feature engineering
    fe_ops = session.get("fe_operations", [])
    for op in fe_ops:
        try:
            op_type = op.get("type")
            if op_type == "combine":
                col_a, col_b = op["col_a"], op["col_b"]
                operation, new_col = op["operation"], op["new_col"]
                if col_a in new_df.columns and col_b in new_df.columns:
                    ops = {"sum": lambda: new_df[col_a] + new_df[col_b],
                           "diff": lambda: new_df[col_a] - new_df[col_b],
                           "product": lambda: new_df[col_a] * new_df[col_b],
                           "ratio": lambda: new_df[col_a] / new_df[col_b].replace(0, np.nan)}
                    if operation in ops:
                        new_df[new_col] = ops[operation]()
                        transforms.append(f"Combinaison : {new_col}")

            elif op_type == "derive":
                col, func, new_col = op["col"], op["func"], op["new_col"]
                if col in new_df.columns:
                    min_val = op.get("min_val", 0)
                    funcs = {
                        "square": lambda: new_df[col] ** 2,
                        "sqrt": lambda: np.sqrt(new_df[col] - min_val) if min_val < 0 else np.sqrt(new_df[col]),
                        "log": lambda: np.log1p(new_df[col] - min_val) if min_val <= 0 else np.log1p(new_df[col]),
                        "inv": lambda: 1.0 / new_df[col].replace(0, np.nan),
                        "abs": lambda: new_df[col].abs(),
                    }
                    if func in funcs:
                        new_df[new_col] = funcs[func]()
                        transforms.append(f"Dérivée : {new_col}")

            elif op_type == "transform_inplace":
                col, func = op["col"], op["func"]
                if col in new_df.columns:
                    min_val = op.get("min_val", 1)
                    if func == "log":
                        new_df[col] = np.log1p(new_df[col] - min_val) if min_val <= 0 else np.log1p(new_df[col])
                    elif func == "sqrt":
                        new_df[col] = np.sqrt(new_df[col].clip(lower=0))
                    elif func == "square":
                        new_df[col] = new_df[col] ** 2
                    transforms.append(f"Transformation : {col} ({func})")

            elif op_type == "discretize":
                col, n_bins = op["col"], op["n_bins"]
                if col in new_df.columns:
                    new_df[f"{col}_bin"] = pd.qcut(new_df[col], q=n_bins, labels=False, duplicates="drop")
                    transforms.append(f"Découpage : {col}")
        except Exception:
            continue

    # 2. Encodage
    encoders = session.get("encoders", {})
    for col, enc_info in encoders.items():
        if col not in new_df.columns:
            continue
        enc_type = enc_info.get("type")
        if enc_type == "onehot":
            categories = enc_info.get("categories", [])
            dummies = pd.get_dummies(new_df[col], prefix=col, dtype=int)
            new_df = pd.concat([new_df.drop(columns=[col]), dummies], axis=1)
            for cat_col in categories:
                if cat_col not in new_df.columns:
                    new_df[cat_col] = 0
            transforms.append(f"One-Hot : {col}")
        elif enc_type == "label":
            encoder = enc_info.get("encoder")
            if encoder:
                known = set(encoder.classes_)
                mask = new_df[col].notna()
                new_df.loc[mask, col] = new_df.loc[mask, col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in known else -1)
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
                transforms.append(f"Label Encoding : {col}")
        elif enc_type == "target":
            mapping = enc_info.get("mapping", {})
            if mapping:
                fallback = np.mean(list(mapping.values()))
                new_df[col] = new_df[col].map(mapping).fillna(fallback)
                transforms.append(f"Target Encoding : {col}")

    # 3. Scaling
    scaler = session.get("scaler")
    scaled_columns = session.get("scaled_columns", [])
    if scaler and scaled_columns:
        cols = [c for c in scaled_columns if c in new_df.columns]
        if cols:
            new_df[cols] = scaler.transform(new_df[cols])
            transforms.append(f"Scaling : {len(cols)} colonnes")

    return new_df, transforms
