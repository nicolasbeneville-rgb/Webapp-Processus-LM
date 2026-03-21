# -*- coding: utf-8 -*-
"""
timeseries.py — Module complet d'analyse et de modélisation de séries temporelles.

Fournit : tests de stationnarité, décomposition saisonnière, ACF/PACF,
moyennes mobiles, modèles ARIMA, validation walk-forward, visualisations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ═══════════════════════════════════════════════════════════════════
# 1. DÉTECTION ET PRÉPARATION
# ═══════════════════════════════════════════════════════════════════

def detect_datetime_column(df: pd.DataFrame) -> str | None:
    """Détecte la colonne datetime la plus probable dans un DataFrame."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # Tenter la conversion (format européen DD/MM puis ISO)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                sample = df[col].dropna().head(20)
                parsed = pd.to_datetime(sample, dayfirst=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return col
            except (ValueError, TypeError):
                continue
    return None


def prepare_timeseries(df: pd.DataFrame, datetime_col: str,
                       target_col: str, freq: str = None) -> pd.Series:
    """Prépare une série temporelle : tri, index datetime, fréquence.

    Returns:
        pd.Series avec index DatetimeIndex, triée et sans doublons.
    """
    ts = df[[datetime_col, target_col]].copy()
    ts[datetime_col] = pd.to_datetime(ts[datetime_col], dayfirst=True, errors="coerce")
    ts = ts.sort_values(datetime_col).drop_duplicates(subset=datetime_col)
    ts = ts.set_index(datetime_col)[target_col]
    ts = ts.dropna()

    if freq:
        ts = ts.asfreq(freq)
    elif ts.index.inferred_freq:
        ts = ts.asfreq(ts.index.inferred_freq)

    return ts


def detect_frequency(ts: pd.Series) -> dict:
    """Détecte la fréquence d'une série temporelle.

    Returns:
        Dict avec freq (str), label (str), median_delta (timedelta).
    """
    if len(ts) < 3:
        return {"freq": None, "label": "Inconnue", "median_delta": None}

    deltas = pd.Series(ts.index).diff().dropna()
    median_delta = deltas.median()

    freq_map = {
        pd.Timedelta(hours=1): ("h", "Horaire"),
        pd.Timedelta(days=1): ("D", "Journalier"),
        pd.Timedelta(days=7): ("W", "Hebdomadaire"),
        pd.Timedelta(days=30): ("MS", "Mensuel"),
        pd.Timedelta(days=91): ("QS", "Trimestriel"),
        pd.Timedelta(days=365): ("YS", "Annuel"),
    }

    best_freq = None
    best_label = "Irrégulier"
    best_diff = float("inf")

    for delta, (f, label) in freq_map.items():
        diff = abs((median_delta - delta).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_freq = f
            best_label = label

    return {"freq": best_freq, "label": best_label, "median_delta": median_delta}


# ═══════════════════════════════════════════════════════════════════
# 2. TESTS DE STATIONNARITÉ
# ═══════════════════════════════════════════════════════════════════

def test_stationarity(ts: pd.Series) -> dict:
    """Exécute les tests ADF et KPSS sur la série.

    Returns:
        Dict avec résultats ADF, KPSS, et conclusion en français.
    """
    ts_clean = ts.dropna()

    # Test ADF (H0 = non-stationnaire)
    adf_result = adfuller(ts_clean, autolag="AIC")
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    adf_stationary = adf_pvalue < 0.05

    # Test KPSS (H0 = stationnaire)
    try:
        kpss_result = kpss(ts_clean, regression="c", nlags="auto")
        kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
        kpss_stationary = kpss_pvalue > 0.05
    except Exception:
        kpss_stat, kpss_pvalue = None, None
        kpss_stationary = None

    # Conclusion
    if adf_stationary and kpss_stationary:
        conclusion = "La série est stationnaire (ADF et KPSS concordent)."
        is_stationary = True
    elif not adf_stationary and (kpss_stationary is False or kpss_stationary is None):
        conclusion = ("La série n'est PAS stationnaire. "
                       "Une différenciation est recommandée.")
        is_stationary = False
    else:
        conclusion = ("Résultats ambigus entre ADF et KPSS. "
                       "La série pourrait avoir une tendance ou une rupture.")
        is_stationary = False

    return {
        "adf_statistic": round(adf_stat, 4),
        "adf_pvalue": round(adf_pvalue, 4),
        "adf_stationary": adf_stationary,
        "kpss_statistic": round(kpss_stat, 4) if kpss_stat else None,
        "kpss_pvalue": round(kpss_pvalue, 4) if kpss_pvalue else None,
        "kpss_stationary": kpss_stationary,
        "is_stationary": is_stationary,
        "conclusion": conclusion,
    }


def make_stationary(ts: pd.Series, max_diffs: int = 2) -> tuple:
    """Différencie la série jusqu'à la rendre stationnaire.

    Returns:
        Tuple (série différenciée, nombre de différenciations, résultats tests).
    """
    current = ts.dropna().copy()
    for d in range(max_diffs + 1):
        result = test_stationarity(current)
        if result["is_stationary"]:
            return current, d, result
        if d < max_diffs:
            current = current.diff().dropna()
    return current, max_diffs, result


# ═══════════════════════════════════════════════════════════════════
# 3. ACF / PACF
# ═══════════════════════════════════════════════════════════════════

def plot_acf_pacf(ts: pd.Series, lags: int = 40,
                  title: str = "Autocorrélation") -> plt.Figure:
    """Trace les graphiques ACF et PACF côte à côte."""
    ts_clean = ts.dropna()
    n = len(ts_clean)
    lags = min(lags, n // 2 - 1)
    if lags < 1:
        lags = 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # ACF
    acf_vals = acf(ts_clean, nlags=lags, fft=True)
    ax1.bar(range(len(acf_vals)), acf_vals, width=0.3, color="steelblue")
    conf = 1.96 / np.sqrt(n)
    ax1.axhline(conf, ls="--", color="gray", alpha=0.7)
    ax1.axhline(-conf, ls="--", color="gray", alpha=0.7)
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_title("ACF (Autocorrélation)")
    ax1.set_xlabel("Lag")

    # PACF
    try:
        pacf_vals = pacf(ts_clean, nlags=lags)
        ax2.bar(range(len(pacf_vals)), pacf_vals, width=0.3, color="coral")
        ax2.axhline(conf, ls="--", color="gray", alpha=0.7)
        ax2.axhline(-conf, ls="--", color="gray", alpha=0.7)
        ax2.axhline(0, color="black", lw=0.5)
    except Exception:
        ax2.text(0.5, 0.5, "PACF indisponible", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title("PACF (Autocorrélation partielle)")
    ax2.set_xlabel("Lag")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def suggest_arima_order(ts: pd.Series, max_lags: int = 20) -> tuple:
    """Suggère un ordre (p, d, q) basé sur ACF/PACF.

    Heuristique simple :
    - d = nombre de différenciations nécessaires
    - p = dernier lag significatif dans PACF
    - q = dernier lag significatif dans ACF
    """
    _, d, _ = make_stationary(ts, max_diffs=2)

    ts_diff = ts.dropna().copy()
    for _ in range(d):
        ts_diff = ts_diff.diff().dropna()

    n = len(ts_diff)
    max_lags = min(max_lags, n // 2 - 1)
    if max_lags < 1:
        return (1, d, 1)

    conf = 1.96 / np.sqrt(n)

    # ACF pour q
    acf_vals = acf(ts_diff, nlags=max_lags, fft=True)
    q = 0
    for i in range(1, len(acf_vals)):
        if abs(acf_vals[i]) > conf:
            q = i

    # PACF pour p
    try:
        pacf_vals = pacf(ts_diff, nlags=max_lags)
        p = 0
        for i in range(1, len(pacf_vals)):
            if abs(pacf_vals[i]) > conf:
                p = i
    except Exception:
        p = 1

    # Limiter
    p = min(p, 5) or 1
    q = min(q, 5) or 1

    return (p, d, q)


# ═══════════════════════════════════════════════════════════════════
# 4. DÉCOMPOSITION SAISONNIÈRE
# ═══════════════════════════════════════════════════════════════════

def decompose_series(ts: pd.Series, model: str = "additive",
                     period: int = None) -> dict:
    """Décompose la série en tendance, saisonnalité et résidu.

    Returns:
        Dict avec components (DecomposeResult) et figure matplotlib.
    """
    ts_clean = ts.dropna()

    if period is None:
        freq_info = detect_frequency(ts_clean)
        period_map = {"Horaire": 24, "Journalier": 7, "Hebdomadaire": 52,
                      "Mensuel": 12, "Trimestriel": 4, "Annuel": 1}
        period = period_map.get(freq_info["label"], max(2, len(ts_clean) // 4))
        period = min(period, len(ts_clean) // 2)

    if period < 2:
        period = 2

    result = seasonal_decompose(ts_clean, model=model, period=period,
                                 extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        ("Données originales", result.observed, "steelblue"),
        ("Tendance", result.trend, "darkorange"),
        ("Saisonnalité", result.seasonal, "green"),
        ("Résidus", result.resid, "red"),
    ]
    for ax, (title, data, color) in zip(axes, components):
        ax.plot(data, color=color, linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Décomposition {model}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    return {"result": result, "figure": fig, "period": period}


# ═══════════════════════════════════════════════════════════════════
# 5. MOYENNES MOBILES ET TENDANCE
# ═══════════════════════════════════════════════════════════════════

def plot_moving_averages(ts: pd.Series,
                          windows: list = None) -> plt.Figure:
    """Trace la série avec ses moyennes mobiles."""
    ts_clean = ts.dropna()
    if windows is None:
        n = len(ts_clean)
        windows = [w for w in [7, 14, 30, 90] if w < n // 2]
        if not windows:
            windows = [max(2, n // 4)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts_clean, label="Données", alpha=0.5, linewidth=0.8)

    colors = ["darkorange", "green", "red", "purple"]
    for i, w in enumerate(windows):
        ma = ts_clean.rolling(window=w, center=True).mean()
        color = colors[i % len(colors)]
        ax.plot(ma, label=f"MA {w}", linewidth=2, color=color)

    ax.set_title("Moyennes mobiles", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_trend_analysis(ts: pd.Series) -> plt.Figure:
    """Analyse de tendance avec régression linéaire."""
    ts_clean = ts.dropna()
    x = np.arange(len(ts_clean))
    y = ts_clean.values.astype(float)

    # Régression linéaire
    coeffs = np.polyfit(x, y, 1)
    trend_line = np.polyval(coeffs, x)
    slope = coeffs[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts_clean.index, y, alpha=0.5, label="Données", linewidth=0.8)
    ax.plot(ts_clean.index, trend_line, "r--", linewidth=2,
            label=f"Tendance (pente = {slope:.4f})")

    direction = "haussière" if slope > 0 else "baissière"
    ax.set_title(f"Analyse de tendance — Tendance {direction}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# 6. MODÉLISATION ARIMA
# ═══════════════════════════════════════════════════════════════════

def fit_arima(ts: pd.Series, order: tuple = None,
              train_ratio: float = 0.8) -> dict:
    """Entraîne un modèle ARIMA et retourne les résultats.

    Returns:
        Dict avec model, forecast, metrics, train/test, figure.
    """
    ts_clean = ts.dropna()
    n = len(ts_clean)
    split = int(n * train_ratio)

    train = ts_clean.iloc[:split]
    test = ts_clean.iloc[split:]

    if order is None:
        order = suggest_arima_order(train)

    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
    except Exception as e:
        return {"error": str(e), "order": order}

    # Prévisions sur le jeu de test
    forecast = fitted.forecast(steps=len(test))

    # Métriques
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test.replace(0, np.nan))) * 100

    # Figure
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train.values, label="Train", color="steelblue")
    ax.plot(test.index, test.values, label="Test (réel)", color="green")
    ax.plot(test.index, forecast.values, label="Prévision", color="red",
            linestyle="--")
    ax.axvline(x=test.index[0], color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"ARIMA{order} — MAE={mae:.2f}, RMSE={rmse:.2f}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return {
        "model": fitted,
        "order": order,
        "train": train,
        "test": test,
        "forecast": forecast,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "aic": round(fitted.aic, 2),
        "bic": round(fitted.bic, 2),
        "figure": fig,
    }


def arima_grid_search(ts: pd.Series, p_range: range = range(0, 4),
                       d_range: range = range(0, 3),
                       q_range: range = range(0, 4),
                       train_ratio: float = 0.8) -> list:
    """Recherche le meilleur ordre ARIMA par AIC.

    Returns:
        Liste triée de dicts {order, aic, mae, rmse} (meilleur en premier).
    """
    ts_clean = ts.dropna()
    n = len(ts_clean)
    split = int(n * train_ratio)
    train = ts_clean.iloc[:split]
    test = ts_clean.iloc[split:]

    results = []
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=len(test))
                    mae = mean_absolute_error(test, forecast)
                    rmse = np.sqrt(mean_squared_error(test, forecast))
                    results.append({
                        "order": (p, d, q),
                        "aic": round(fitted.aic, 2),
                        "bic": round(fitted.bic, 2),
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                    })
                except Exception:
                    continue

    results.sort(key=lambda x: x["aic"])
    return results


# ═══════════════════════════════════════════════════════════════════
# 6b. MODÉLISATION SARIMA (composante saisonnière)
# ═══════════════════════════════════════════════════════════════════

def fit_sarima(ts: pd.Series, order: tuple = None,
               seasonal_order: tuple = None,
               train_ratio: float = 0.8) -> dict:
    """Entraîne un modèle SARIMA et retourne les résultats.

    Args:
        ts: Série temporelle.
        order: (p, d, q) pour la composante non-saisonnière.
        seasonal_order: (P, D, Q, m) pour la composante saisonnière.
        train_ratio: Proportion de données d'entraînement.

    Returns:
        Dict avec model, forecast, métriques, figure.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    ts_clean = ts.dropna()
    n = len(ts_clean)
    split = int(n * train_ratio)

    train = ts_clean.iloc[:split]
    test = ts_clean.iloc[split:]

    if order is None:
        order = suggest_arima_order(train)
    if seasonal_order is None:
        seasonal_order = (1, 1, 1, 12)

    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
    except Exception as e:
        return {"error": str(e), "order": order, "seasonal_order": seasonal_order}

    forecast = fitted.forecast(steps=len(test))

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test.replace(0, np.nan))) * 100

    # Figure
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train.values, label="Train", color="steelblue")
    ax.plot(test.index, test.values, label="Test (réel)", color="green")
    ax.plot(test.index, forecast.values, label="Prévision", color="red",
            linestyle="--")
    ax.axvline(x=test.index[0], color="gray", linestyle=":", alpha=0.7)
    ax.set_title(
        f"SARIMA{order}×{seasonal_order} — MAE={mae:.2f}, RMSE={rmse:.2f}",
        fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return {
        "model": fitted,
        "order": order,
        "seasonal_order": seasonal_order,
        "train": train,
        "test": test,
        "forecast": forecast,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "aic": round(fitted.aic, 2),
        "bic": round(fitted.bic, 2),
        "figure": fig,
        "is_sarima": True,
    }


# ═══════════════════════════════════════════════════════════════════
# 7. VALIDATION WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════

def walk_forward_validation(ts: pd.Series, order: tuple,
                             n_splits: int = 5,
                             min_train_size: int = 30,
                             gap: int = 0) -> dict:
    """Validation walk-forward : re-entraîne le modèle à chaque pas.

    Args:
        ts: Série temporelle.
        order: Ordre ARIMA (p, d, q).
        n_splits: Nombre de folds.
        min_train_size: Taille minimale d'entraînement.
        gap: Nombre de points à exclure entre train et test
             pour éviter les fuites de données.

    Returns:
        Dict avec scores par fold, score moyen, et figure.
    """
    ts_clean = ts.dropna()
    n = len(ts_clean)

    if n < min_train_size + n_splits + gap:
        return {"error": f"Pas assez de données ({n} points) pour {n_splits} folds avec gap={gap}."}

    step_size = (n - min_train_size - gap) // n_splits
    if step_size < 1:
        step_size = 1

    folds = []
    actuals = []
    predictions = []

    for i in range(n_splits):
        train_end = min_train_size + i * step_size
        test_start = train_end + gap
        test_end = min(test_start + step_size, n)
        if train_end >= n or test_start >= n:
            break

        train = ts_clean.iloc[:train_end]
        test = ts_clean.iloc[test_start:test_end]
        if len(test) == 0:
            break

        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(test))
            mae = mean_absolute_error(test, forecast)

            folds.append({
                "fold": i + 1,
                "train_size": len(train),
                "test_size": len(test),
                "mae": round(mae, 4),
            })
            actuals.extend(test.values.tolist())
            predictions.extend(forecast.values.tolist())
        except Exception:
            continue

    if not folds:
        return {"error": "Aucun fold n'a convergé."}

    mae_scores = [f["mae"] for f in folds]

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar([f"Fold {f['fold']}" for f in folds], mae_scores,
            color="steelblue")
    ax1.axhline(np.mean(mae_scores), ls="--", color="red",
                label=f"Moyenne = {np.mean(mae_scores):.4f}")
    ax1.set_title("MAE par fold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(actuals, predictions, alpha=0.5, s=20)
    lims = [min(min(actuals), min(predictions)),
            max(max(actuals), max(predictions))]
    ax2.plot(lims, lims, "r--", alpha=0.7)
    ax2.set_xlabel("Réel")
    ax2.set_ylabel("Prédit")
    ax2.set_title("Réel vs Prédit (tous folds)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Walk-Forward Validation — ARIMA{order}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    return {
        "folds": folds,
        "mean_mae": round(np.mean(mae_scores), 4),
        "std_mae": round(np.std(mae_scores), 4),
        "figure": fig,
    }


# ═══════════════════════════════════════════════════════════════════
# 8. PRÉVISION FUTURE
# ═══════════════════════════════════════════════════════════════════

def forecast_future(ts: pd.Series, order: tuple,
                     steps: int = 30) -> dict:
    """Entraîne ARIMA sur toute la série et prédit le futur.

    Returns:
        Dict avec forecast (Series), confidence intervals, et figure.
    """
    ts_clean = ts.dropna()

    try:
        model = ARIMA(ts_clean, order=order)
        fitted = model.fit()
    except Exception as e:
        return {"error": str(e)}

    forecast_result = fitted.get_forecast(steps=steps)
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Créer index futur
    if ts_clean.index.inferred_freq:
        future_idx = pd.date_range(
            start=ts_clean.index[-1],
            periods=steps + 1,
            freq=ts_clean.index.inferred_freq
        )[1:]
    else:
        # Estimation basée sur le delta médian
        deltas = pd.Series(ts_clean.index).diff().dropna()
        delta = deltas.median()
        future_idx = [ts_clean.index[-1] + delta * (i + 1) for i in range(steps)]

    forecast_values.index = future_idx
    conf_int.index = future_idx

    # Figure
    fig, ax = plt.subplots(figsize=(14, 5))

    # Dernières observations
    last_n = min(len(ts_clean), steps * 3)
    recent = ts_clean.iloc[-last_n:]
    ax.plot(recent.index, recent.values, label="Historique", color="steelblue")

    ax.plot(forecast_values.index, forecast_values.values,
            label="Prévision", color="red", linewidth=2)
    ax.fill_between(conf_int.index,
                     conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     alpha=0.2, color="red", label="Intervalle de confiance 95%")

    ax.axvline(x=ts_clean.index[-1], color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"Prévision ARIMA{order} — {steps} pas",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    return {
        "forecast": forecast_values,
        "conf_int": conf_int,
        "model": fitted,
        "figure": fig,
    }


# ═══════════════════════════════════════════════════════════════════
# 9. VISUALISATIONS GÉNÉRALES
# ═══════════════════════════════════════════════════════════════════

def plot_timeseries(ts: pd.Series, title: str = "Série temporelle") -> plt.Figure:
    """Trace la série temporelle avec statistiques de base."""
    ts_clean = ts.dropna()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts_clean, color="steelblue", linewidth=0.8)
    ax.axhline(ts_clean.mean(), ls="--", color="red", alpha=0.5,
               label=f"Moyenne = {ts_clean.mean():.2f}")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_timeseries_interactive(ts: pd.Series, title: str = "Série temporelle"):
    """Version Plotly interactive de la série temporelle (zoom, pan, hover)."""
    import plotly.graph_objects as go
    ts_clean = ts.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_clean.index, y=ts_clean.values,
        mode="lines", name=ts_clean.name or "Valeur",
        line=dict(color="#4F5BD5", width=1.2),
    ))
    mean_val = float(ts_clean.mean())
    fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Moyenne = {mean_val:.2f}")
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="Date", yaxis_title="Valeur",
        xaxis=dict(rangeslider=dict(visible=True)),
        height=450,
    )
    return fig


def plot_seasonal_boxplot(ts: pd.Series, period: str = "month") -> plt.Figure:
    """Boxplot saisonnier (par mois, jour de semaine, etc.)."""
    ts_clean = ts.dropna()
    df = pd.DataFrame({"value": ts_clean})

    if period == "month":
        df["period"] = ts_clean.index.month
        labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                  "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
        title = "Distribution par mois"
    elif period == "dayofweek":
        df["period"] = ts_clean.index.dayofweek
        labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        title = "Distribution par jour de la semaine"
    elif period == "quarter":
        df["period"] = ts_clean.index.quarter
        labels = ["T1", "T2", "T3", "T4"]
        title = "Distribution par trimestre"
    elif period == "hour":
        df["period"] = ts_clean.index.hour
        labels = [str(h) for h in range(24)]
        title = "Distribution par heure"
    else:
        df["period"] = ts_clean.index.month
        labels = None
        title = "Distribution saisonnière"

    fig, ax = plt.subplots(figsize=(14, 5))
    groups = [group["value"].values for _, group in df.groupby("period")]
    positions = sorted(df["period"].unique())
    ax.boxplot(groups, positions=range(len(groups)))

    if labels and len(positions) <= len(labels):
        ax.set_xticklabels([labels[p] if p < len(labels) else str(p)
                            for p in positions])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# 10. RÉSUMÉ AUTOMATIQUE
# ═══════════════════════════════════════════════════════════════════

def auto_summary(ts: pd.Series) -> dict:
    """Génère un résumé automatique de la série temporelle."""
    ts_clean = ts.dropna()
    freq = detect_frequency(ts_clean)
    stationarity = test_stationarity(ts_clean)

    return {
        "n_points": len(ts_clean),
        "date_debut": str(ts_clean.index.min()),
        "date_fin": str(ts_clean.index.max()),
        "frequence": freq["label"],
        "moyenne": round(ts_clean.mean(), 4),
        "ecart_type": round(ts_clean.std(), 4),
        "min": round(ts_clean.min(), 4),
        "max": round(ts_clean.max(), 4),
        "stationnaire": stationarity["is_stationary"],
        "conclusion_stationnarite": stationarity["conclusion"],
        "nb_valeurs_manquantes": int(ts.isna().sum()),
    }


# ═══════════════════════════════════════════════════════════════════
# 11. ANALYSE DE CONTINUITÉ ET GAPS
# ═══════════════════════════════════════════════════════════════════

def analyze_ts_continuity(df: pd.DataFrame, datetime_col: str,
                           value_cols: list) -> dict:
    """Analyse la continuité temporelle de chaque colonne.

    Returns:
        Dict avec freq_info, gaps, duplicates, column_stats, et recommandations.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # ── Détection des doublons de date ──
    date_counts = df[datetime_col].value_counts()
    dup_dates = date_counts[date_counts > 1]
    duplicates = []
    for dt_val, count in dup_dates.items():
        rows = df[df[datetime_col] == dt_val]
        # Vérifier si les valeurs diffèrent entre les lignes dupliquées
        identical = rows.drop(columns=[datetime_col]).duplicated(keep=False).all()
        duplicates.append({
            "date": dt_val,
            "count": int(count),
            "identical": bool(identical),
        })

    # ── Fréquence globale (calculée sur dates dédoublonnées) ──
    df_unique = df.drop_duplicates(subset=datetime_col).sort_values(datetime_col).reset_index(drop=True)
    deltas = pd.Series(df_unique[datetime_col]).diff().dropna()
    median_delta = deltas.median()

    # ── Détecter les gaps (trous > 1.5x la fréquence médiane) ──
    threshold = median_delta * 1.5
    gap_mask = deltas > threshold
    gaps = []
    if gap_mask.any():
        for idx in gap_mask[gap_mask].index:
            gap_start = df_unique[datetime_col].iloc[idx - 1]
            gap_end = df_unique[datetime_col].iloc[idx]
            gap_duration = gap_end - gap_start
            n_missing = int(gap_duration / median_delta) - 1
            # Valeur juste avant et juste après le trou
            val_before = {}
            val_after = {}
            for vc in value_cols:
                if vc in df_unique.columns:
                    vb = df_unique.loc[df_unique[datetime_col] == gap_start, vc]
                    va = df_unique.loc[df_unique[datetime_col] == gap_end, vc]
                    val_before[vc] = float(vb.iloc[0]) if len(vb) > 0 and pd.notna(vb.iloc[0]) else None
                    val_after[vc] = float(va.iloc[0]) if len(va) > 0 and pd.notna(va.iloc[0]) else None
            gaps.append({
                "debut": gap_start,
                "fin": gap_end,
                "duree": gap_duration,
                "periodes_manquees": n_missing,
                "val_before": val_before,
                "val_after": val_after,
            })

    # ── Identifier les segments contigus (entre les gros gaps, > 5x médiane) ──
    big_threshold = median_delta * 5
    big_gap_indices = deltas[deltas > big_threshold].index.tolist()
    segments = []
    seg_boundaries = [0] + big_gap_indices + [len(df_unique)]
    for i in range(len(seg_boundaries) - 1):
        s_start = seg_boundaries[i]
        s_end = seg_boundaries[i + 1] - 1
        if s_end < s_start:
            continue
        seg_df = df_unique.iloc[s_start:s_end + 1]
        if len(seg_df) < 2:
            continue
        seg_deltas = pd.Series(seg_df[datetime_col]).diff().dropna()
        max_internal = 0
        if len(seg_deltas) > 0:
            max_internal_td = seg_deltas.max()
            max_internal = max(0, int(max_internal_td / median_delta) - 1)
        segments.append({
            "start": seg_df[datetime_col].iloc[0],
            "end": seg_df[datetime_col].iloc[-1],
            "n_points": len(seg_df),
            "max_internal_gap": max_internal,
        })

    # ── Stats par colonne ──
    column_stats = {}
    for col in value_cols:
        if col not in df.columns:
            continue
        col_data = df[col]
        n_total = len(col_data)
        n_missing = int(col_data.isna().sum())
        pct_missing = round(n_missing / max(n_total, 1) * 100, 1)

        is_na = col_data.isna()
        longest_na_streak = 0
        current_streak = 0
        for v in is_na:
            if v:
                current_streak += 1
                longest_na_streak = max(longest_na_streak, current_streak)
            else:
                current_streak = 0

        column_stats[col] = {
            "n_total": n_total,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "longest_na_streak": longest_na_streak,
            "first_valid": str(df.loc[col_data.first_valid_index(), datetime_col])
            if col_data.first_valid_index() is not None else None,
            "last_valid": str(df.loc[col_data.last_valid_index(), datetime_col])
            if col_data.last_valid_index() is not None else None,
        }

    # ── Recommandations ──
    recommendations = []
    if duplicates:
        n_ident = sum(1 for d in duplicates if d["identical"])
        n_diff = len(duplicates) - n_ident
        if n_ident > 0:
            recommendations.append({
                "type": "dedup",
                "message": f"🔴 **{n_ident} date(s) en double** (lignes identiques) → "
                           "supprimez les doublons (garder la 1ère occurrence).",
                "priority": "high",
            })
        if n_diff > 0:
            recommendations.append({
                "type": "dedup_conflict",
                "message": f"🔴 **{n_diff} date(s) en double avec valeurs différentes** → "
                           "choisissez une stratégie (garder la 1ère, la dernière, ou la moyenne).",
                "priority": "high",
            })

    if gaps:
        total_missing = sum(g["periodes_manquees"] for g in gaps)
        if total_missing > len(df) * 0.1:
            recommendations.append({
                "type": "cut",
                "message": f"⚠️ Beaucoup de trous ({total_missing} périodes). "
                           "Envisagez de couper la série pour garder un segment continu.",
                "priority": "high",
            })
        else:
            recommendations.append({
                "type": "interpolation",
                "message": f"💡 {len(gaps)} gap(s) détecté(s) ({total_missing} périodes). "
                           "L'interpolation (linéaire ou par propagation) est recommandée.",
                "priority": "medium",
            })

    for col, stats in column_stats.items():
        if stats["pct_missing"] > 30:
            recommendations.append({
                "type": "drop_column",
                "message": f"🔴 **{col}** : {stats['pct_missing']}% manquant → colonne peu fiable, "
                           "envisagez de la supprimer ou de n'utiliser que le segment exploitable.",
                "priority": "high",
                "column": col,
            })
        elif stats["longest_na_streak"] > 10:
            recommendations.append({
                "type": "cut",
                "message": f"🟠 **{col}** : {stats['longest_na_streak']} trous consécutifs → "
                           "interpoler sur de si longues séries est risqué. "
                           "Coupez la série pour garder un segment continu.",
                "priority": "medium",
                "column": col,
            })
        elif stats["n_missing"] > 0:
            recommendations.append({
                "type": "interpolation",
                "message": f"🟢 **{col}** : {stats['n_missing']} trous épars → "
                           "interpolation linéaire ou forward-fill recommandée.",
                "priority": "low",
                "column": col,
            })

    return {
        "median_delta": median_delta,
        "n_gaps": len(gaps),
        "gaps": gaps,
        "duplicates": duplicates,
        "n_duplicates": len(duplicates),
        "segments": segments,
        "column_stats": column_stats,
        "recommendations": recommendations,
        "total_points": len(df),
    }


def reindex_ts(df: pd.DataFrame, datetime_col: str,
               freq: str = None) -> pd.DataFrame:
    """Matérialise les dates manquantes en créant des lignes NaN.

    Crée un index régulier de date_min à date_max à la fréquence
    détectée (ou spécifiée) et réindexe le DataFrame dessus.

    Returns:
        DataFrame avec toutes les dates, les trous remplis de NaN.
    """
    result = df.copy()
    result[datetime_col] = pd.to_datetime(result[datetime_col])
    result = result.sort_values(datetime_col).reset_index(drop=True)

    date_min = result[datetime_col].min()
    date_max = result[datetime_col].max()

    if not freq:
        deltas = result[datetime_col].diff().dropna()
        median_delta = deltas.median()
        freq_map = [
            (pd.Timedelta(hours=1), "h"),
            (pd.Timedelta(days=1), "D"),
            (pd.Timedelta(days=7), "W-MON"),
            (pd.Timedelta(days=30), "MS"),
            (pd.Timedelta(days=91), "QS"),
            (pd.Timedelta(days=365), "YS"),
        ]
        best_freq = "D"
        best_diff = float("inf")
        for delta, f in freq_map:
            diff = abs((median_delta - delta).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_freq = f
        freq = best_freq

    full_range = pd.date_range(start=date_min, end=date_max, freq=freq)
    result = result.set_index(datetime_col)
    result = result.reindex(full_range)
    result.index.name = datetime_col
    result = result.reset_index()
    return result


def detect_seasonality(ts: pd.Series) -> dict:
    """Détecte la saisonnalité et retourne ses caractéristiques.

    Returns:
        Dict avec has_seasonality, period, strength, model_type, recommendations.
    """
    ts_clean = ts.dropna()
    if len(ts_clean) < 10:
        return {"has_seasonality": False, "period": None, "strength": 0}

    freq_info = detect_frequency(ts_clean)

    # Candidats de période par fréquence — du plus naturel au moins naturel.
    # Ex. : données journalières → tester annuel (365), mensuel (30), hebdo (7).
    candidates_map = {
        "Horaire":      [24, 168, 24*7],       # jour, semaine
        "Journalier":   [365, 30, 7],           # année, mois, semaine
        "Hebdomadaire": [52, 13, 4],            # année, trimestre, mois
        "Mensuel":      [12, 6, 4, 3],          # année, semestre, trimestre
        "Trimestriel":  [4],                    # année
        "Annuel":       [],                     # pas de cycle infra-annuel
    }
    candidates = candidates_map.get(freq_info["label"], [])

    # Filtrer les candidats réalisables (période < len/2 et >= 2)
    max_period = len(ts_clean) // 2
    candidates = [p for p in candidates if 2 <= p < max_period]

    # Évaluer chaque candidat par décomposition saisonnière
    best = {"period": None, "strength": 0.0, "decomp": None}
    for p in candidates:
        try:
            result = seasonal_decompose(ts_clean, model="additive", period=p,
                                         extrapolate_trend="freq")
            seasonal_var = np.var(result.seasonal.dropna())
            resid_var = np.var(result.resid.dropna())
            total_var = seasonal_var + resid_var
            s = seasonal_var / total_var if total_var > 0 else 0
            if s > best["strength"]:
                best = {"period": p, "strength": s, "decomp": result}
        except Exception:
            continue

    # Fallback ACF si aucun candidat n'a fonctionné
    if best["period"] is None or best["strength"] < 0.05:
        try:
            max_lags = min(len(ts_clean) // 2 - 1, 400)
            acf_vals = acf(ts_clean, nlags=max_lags, fft=True)
            conf = 1.96 / np.sqrt(len(ts_clean))
            peaks = []
            for i in range(2, len(acf_vals) - 1):
                if (acf_vals[i] > conf and
                        acf_vals[i] > acf_vals[i - 1] and
                        acf_vals[i] > acf_vals[i + 1]):
                    peaks.append((i, acf_vals[i]))
            if peaks:
                acf_period = peaks[0][0]
                if 2 <= acf_period < max_period:
                    try:
                        result = seasonal_decompose(ts_clean, model="additive",
                                                     period=acf_period,
                                                     extrapolate_trend="freq")
                        seasonal_var = np.var(result.seasonal.dropna())
                        resid_var = np.var(result.resid.dropna())
                        total_var = seasonal_var + resid_var
                        s = seasonal_var / total_var if total_var > 0 else 0
                        if s > best["strength"]:
                            best = {"period": acf_period, "strength": s, "decomp": result}
                    except Exception:
                        pass
        except Exception:
            pass

    period = best["period"]
    strength = best["strength"]

    if period is None or period < 2 or strength < 0.05:
        return {"has_seasonality": False, "period": None, "strength": 0,
                "recommendations": ["Pas de saisonnalité détectée → ARIMA classique suffit."]}

    has_seasonality = strength > 0.1

    recommendations = []
    if has_seasonality:
        recommendations.append(
            f"✅ Saisonnalité détectée (période={period}, force={strength:.1%})")
        recommendations.append(
            f"→ Utilisez **SARIMA** avec seasonal_order=({period}) au lieu d'ARIMA simple.")
        if strength > 0.5:
            recommendations.append(
                "→ La saisonnalité est forte. Un modèle sans composante saisonnière "
                "sera significativement moins bon.")
        # Vérifier si additive ou multiplicative
        try:
            result_add = best["decomp"]
            result_mult = seasonal_decompose(ts_clean, model="multiplicative",
                                              period=period, extrapolate_trend="freq")
            resid_cv_add = np.std(result_add.resid.dropna()) / abs(np.mean(result_add.resid.dropna()) + 1e-10)
            resid_cv_mult = np.std(result_mult.resid.dropna()) / abs(np.mean(result_mult.resid.dropna()) + 1e-10)
            if resid_cv_mult < resid_cv_add * 0.9:
                recommendations.append(
                    "→ Saisonnalité **multiplicative** détectée : la log-transformation "
                    "de la cible est recommandée pour stabiliser la variance.")
                model_type = "multiplicative"
            else:
                model_type = "additive"
        except Exception:
            model_type = "additive"
    else:
        recommendations.append("Pas de saisonnalité significative → ARIMA classique suffit.")
        model_type = "additive"

    return {
        "has_seasonality": has_seasonality,
        "period": period,
        "strength": round(strength, 4),
        "model_type": model_type,
        "recommendations": recommendations,
    }


def recommend_ts_transforms(ts: pd.Series) -> list:
    """Analyse la série et recommande des transformations mathématiques.

    Returns:
        Liste de dicts {transform, reason, priority, before_after_hint}.
    """
    ts_clean = ts.dropna()
    recommendations = []

    # 1. Log transform — si variance non constante ou forte asymétrie
    skewness = ts_clean.skew()
    if abs(skewness) > 1.0 and ts_clean.min() > 0:
        recommendations.append({
            "transform": "log",
            "reason": f"Distribution très asymétrique (skew={skewness:.2f}). "
                      "La transformation log stabilise la variance et "
                      "rend la série plus symétrique.",
            "priority": "high",
            "how": "np.log1p(série) → inverse avec np.expm1(prédictions)",
        })
    elif abs(skewness) > 0.5 and ts_clean.min() > 0:
        recommendations.append({
            "transform": "log",
            "reason": f"Asymétrie modérée (skew={skewness:.2f}). "
                      "Le log peut aider si la variance augmente avec le niveau.",
            "priority": "medium",
            "how": "np.log1p(série) → inverse avec np.expm1(prédictions)",
        })

    # 2. Variance non constante (hétéroscédasticité)
    n = len(ts_clean)
    if n > 20:
        first_half_var = ts_clean.iloc[:n // 2].var()
        second_half_var = ts_clean.iloc[n // 2:].var()
        ratio = max(first_half_var, second_half_var) / max(min(first_half_var, second_half_var), 1e-10)
        if ratio > 4:
            recommendations.append({
                "transform": "log",
                "reason": f"Variance non constante (ratio 1ère/2ème moitié : {ratio:.1f}x). "
                          "Le log ou Box-Cox stabilise la variance.",
                "priority": "high",
                "how": "np.log1p(série) pour stabiliser, ou scipy.stats.boxcox",
            })

    # 3. Tendance polynomiale — quand la tendance est non linéaire
    if n > 30:
        x = np.arange(n)
        coeffs1 = np.polyfit(x, ts_clean.values, 1)
        coeffs2 = np.polyfit(x, ts_clean.values, 2)
        y1 = np.polyval(coeffs1, x)
        y2 = np.polyval(coeffs2, x)
        ss_res1 = np.sum((ts_clean.values - y1) ** 2)
        ss_res2 = np.sum((ts_clean.values - y2) ** 2)
        improvement = (ss_res1 - ss_res2) / max(ss_res1, 1e-10)
        if improvement > 0.15 and abs(coeffs2[0]) > 1e-6:
            recommendations.append({
                "transform": "poly",
                "reason": f"La tendance est non linéaire (le polynôme degré 2 améliore "
                          f"de {improvement:.0%} l'ajustement). "
                          "La différenciation d'ordre 2 ou un terme² peut capturer cela.",
                "priority": "medium",
                "how": "Augmenter d dans ARIMA (d=2) ou créer une feature temporelle au carré.",
            })

    # 4. Interactions / produits croisés — pour ML multi-colonnes (pas ARIMA pur)
    recommendations.append({
        "transform": "interactions",
        "reason": "Si vous utilisez un ML classique (pas ARIMA), les produits croisés "
                  "entre features (ex: temp × humidité) peuvent capturer des effets combinés.",
        "priority": "info",
        "how": "Feature Engineering (étape 6c) → combine_columns avec 'product'.",
    })

    return recommendations
