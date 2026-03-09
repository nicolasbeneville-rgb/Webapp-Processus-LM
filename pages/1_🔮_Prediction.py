# -*- coding: utf-8 -*-
"""
Page autonome de prédiction — utilisable indépendamment du pipeline ML Studio.

Permet à un utilisateur de :
  1. Charger un modèle (.pkl / .joblib) exporté depuis ML Studio
  2. Charger ses propres données (CSV / Excel)
  3. Obtenir les prédictions + télécharger le résultat

URL directe : http://<host>:8501/Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔮 Prédiction — ML Studio",
    page_icon="🔮",
    layout="wide",
)

# ── CSS cohérent avec l'app principale ──
st.markdown("""
<style>
:root {
    --primary: #4F5BD5;
    --sidebar-bg: #1B2A4A;
    --sidebar-text: #E2E8F0;
    --title-red: #8B1A1A;
    --surface: #FFFFFF;
    --bg: #F8F9FC;
    --border: #E5E7EB;
    --radius: 10px;
}
.stApp { background-color: var(--bg); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2A4A 0%, #162240 100%) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] h1 {
    color: #E74C3C !important;
    font-weight: 800 !important;
}
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🔮 Prédiction")
    st.markdown("---")
    st.markdown("""
**Mode autonome** — cette page fonctionne
indépendamment du pipeline ML Studio.

**Comment ça marche :**
1. Chargez un modèle `.pkl` ou `.joblib`
2. Chargez vos données (CSV / Excel)
3. Lancez la prédiction

> Le modèle doit avoir été entraîné sur des
> colonnes **identiques** à celles de vos données.
""")

    st.markdown("---")
    st.caption("🔗 [← Retour au Pipeline ML Studio](/)")


# ═══════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ═══════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="color:#8B1A1A; margin-bottom:4px;">🔮 Prédiction autonome</h1>',
    unsafe_allow_html=True,
)
st.caption("Chargez un modèle exporté et vos données pour obtenir des prédictions instantanées.")

# ── Étape 1 : Charger le modèle ──
st.markdown("### 1. Charger le modèle")

model_file = st.file_uploader(
    "Modèle (.pkl ou .joblib)",
    type=["pkl", "joblib"],
    key="pred_model_upload",
    help="Fichier modèle exporté depuis ML Studio ou scikit-learn",
)

model = None
if model_file is not None:
    try:
        model = joblib.load(model_file)
        model_type = type(model).__name__

        c1, c2 = st.columns(2)
        c1.metric("Modèle chargé", model_type)

        # Détection des features attendues
        expected_features = None
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            c2.metric("Features attendues", len(expected_features))
            with st.expander("📋 Colonnes attendues par le modèle"):
                st.write(expected_features)
        elif hasattr(model, "n_features_in_"):
            c2.metric("Nb features", model.n_features_in_)
        else:
            c2.metric("Features", "inconnues")

        st.success(f"✅ Modèle **{model_type}** prêt.")
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle : {e}")
        model = None

st.markdown("---")

# ── Étape 2 : Charger les données ──
st.markdown("### 2. Charger les données")

data_file = st.file_uploader(
    "Données (CSV ou Excel)",
    type=["csv", "xlsx", "xls"],
    key="pred_data_upload",
    help="Fichier avec les mêmes colonnes que celles utilisées pour l'entraînement",
)

df_input = None
if data_file is not None:
    try:
        if data_file.name.endswith((".xlsx", ".xls")):
            df_input = pd.read_excel(data_file)
        else:
            # Détection automatique du séparateur
            sample = data_file.read(4096).decode("utf-8", errors="replace")
            data_file.seek(0)
            if sample.count(";") > sample.count(","):
                sep = ";"
            else:
                sep = ","
            df_input = pd.read_csv(data_file, sep=sep)

        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes", df_input.shape[0])
        c2.metric("Colonnes", df_input.shape[1])
        c3.metric("Taille", f"{df_input.memory_usage(deep=True).sum() / 1024:.0f} Ko")

        st.dataframe(df_input.head(10), use_container_width=True)

        # Vérification de compatibilité
        if model is not None and expected_features is not None:
            missing = [c for c in expected_features if c not in df_input.columns]
            extra = [c for c in df_input.columns if c not in expected_features]
            if missing:
                st.warning(f"⚠️ Colonnes manquantes : {', '.join(missing)}")
            if extra:
                st.info(f"ℹ️ Colonnes ignorées (pas dans le modèle) : {', '.join(extra)}")
            if not missing:
                st.success("✅ Toutes les colonnes attendues sont présentes.")

    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        df_input = None

st.markdown("---")

# ── Étape 3 : Prédiction ──
st.markdown("### 3. Lancer la prédiction")

if model is None:
    st.info("⬆️ Chargez d'abord un modèle ci-dessus.")
elif df_input is None:
    st.info("⬆️ Chargez d'abord vos données ci-dessus.")
else:
    # Déterminer les colonnes à utiliser
    if expected_features is not None:
        available = [c for c in expected_features if c in df_input.columns]
        missing = [c for c in expected_features if c not in df_input.columns]
    else:
        available = df_input.columns.tolist()
        missing = []

    if missing:
        st.error(f"❌ Impossible de prédire — colonnes manquantes : {', '.join(missing)}")
    else:
        if st.button("🔮 Lancer la prédiction", type="primary", key="pred_go"):
            try:
                X_new = df_input[available]

                # Prédictions
                predictions = model.predict(X_new)
                df_result = df_input.copy()
                df_result["🎯 Prédiction"] = predictions

                # Probabilités si classification
                if hasattr(model, "predict_proba"):
                    try:
                        probas = model.predict_proba(X_new)
                        if probas.shape[1] == 2:
                            df_result["📊 Probabilité"] = probas[:, 1]
                        else:
                            for i in range(probas.shape[1]):
                                df_result[f"P(classe_{i})"] = probas[:, i]
                    except Exception:
                        pass

                st.success(f"✅ **{len(predictions)} prédictions** générées !")

                # Statistiques rapides
                c1, c2, c3 = st.columns(3)
                c1.metric("Prédictions", len(predictions))
                if np.issubdtype(type(predictions[0]), np.number):
                    c2.metric("Moyenne", f"{np.mean(predictions):.4f}")
                    c3.metric("Écart-type", f"{np.std(predictions):.4f}")
                else:
                    unique, counts = np.unique(predictions, return_counts=True)
                    top_class = unique[np.argmax(counts)]
                    c2.metric("Classe majoritaire", str(top_class))
                    c3.metric("Classes distinctes", len(unique))

                # Affichage du résultat
                st.dataframe(df_result, use_container_width=True)

                # Téléchargement
                csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Télécharger les résultats (CSV)",
                    csv_bytes,
                    "predictions.csv",
                    "text/csv",
                    key="dl_pred_standalone",
                )

            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")
                with st.expander("🔍 Détails de l'erreur"):
                    st.code(str(e))
