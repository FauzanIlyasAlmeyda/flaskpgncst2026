import joblib
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ======================================================================
# DATA CLEANING
# ======================================================================

def clean_commodity_series(df: pd.DataFrame, commodity_column: str) -> pd.Series:
    cleaned = (
        df[commodity_column]
        .astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace(["-", "", "nan", "None"], np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ======================================================================
# METRICS
# ======================================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def mase(y_true, y_pred, insample) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)
    scale = np.mean(np.abs(np.diff(insample)))
    if scale <= 1e-12:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_prediction(y_true, y_pred, insample, model_name: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": mape_safe(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, insample),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def make_time_series_features(series: pd.Series, max_lag: int) -> pd.DataFrame:
    s = series.astype(float).copy()
    frame = pd.DataFrame(index=s.index)
    frame["target"] = s

    for lag in range(1, max_lag + 1):
        frame[f"lag_{lag}"] = s.shift(lag)

    shifted = s.shift(1)
    for window in [3, 5, 7, 10, 14, 21, 30]:
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std()
        frame[f"roll_min_{window}"] = shifted.rolling(window).min()
        frame[f"roll_max_{window}"] = shifted.rolling(window).max()

    frame["ewm_mean_5"] = shifted.ewm(span=5, adjust=False).mean()
    frame["ewm_mean_14"] = shifted.ewm(span=14, adjust=False).mean()
    frame["diff_1"] = s.shift(1) - s.shift(2)
    frame["diff_5"] = s.shift(1) - s.shift(6)

    idx = pd.DatetimeIndex(frame.index)
    frame["day_of_week"] = idx.dayofweek
    frame["day_of_month"] = idx.day
    frame["month"] = idx.month
    frame["quarter"] = idx.quarter
    frame["day_of_year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def valid_tscv(n_samples, requested_splits):
    n_splits = min(requested_splits, max(2, n_samples // 60))
    n_splits = min(n_splits, n_samples - 1)
    return TimeSeriesSplit(n_splits=n_splits)


# ======================================================================
# DESCRIPTIVE ANALYSIS
# ======================================================================

def descriptive_analysis(df: pd.DataFrame, date_column: str, commodity_column: str) -> dict:
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_column]).sort_values(date_column)
    df[commodity_column] = clean_commodity_series(df, commodity_column)
    df = df.dropna(subset=[commodity_column])

    harga = df[commodity_column]
    mean_val = float(harga.mean())
    std_val = float(harga.std())
    skew_val = float(stats.skew(harga))
    kurt_val = float(stats.kurtosis(harga, fisher=False))

    monthly = df.set_index(date_column)[commodity_column].resample("MS").mean()
    monthly_change = monthly.pct_change().mean() * 100

    return {
        "mean": mean_val,
        "median": float(harga.median()),
        "std": std_val,
        "min": float(harga.min()),
        "max": float(harga.max()),
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "monthly_change_pct": None if pd.isna(monthly_change) else float(monthly_change),
        "cv": float(std_val / mean_val),
        "price_history": [
            {"date": str(row[date_column].date()), "price": float(row[commodity_column])}
            for _, row in df[[date_column, commodity_column]].iterrows()
        ],
    }


# ======================================================================
# TRAINING
# ======================================================================

RF_ITERATIONS_BY_PROFILE = {"Fast": 15, "Balanced": 24, "Thorough": 40}
MAX_LAG = 30
RANDOM_STATE = 42


def train_model(
    df: pd.DataFrame,
    date_column: str,
    commodity_column: str,
    session_id: str,
    split_method: str = "percentage",
    train_ratio: int = 80,
    train_end_date: str = None,
    rf_profile: str = "Balanced",
    cv_splits: int = 5,
) -> dict:
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors="coerce")
    df[commodity_column] = clean_commodity_series(df, commodity_column)
    df = df.dropna(subset=[date_column, commodity_column]).sort_values(date_column)

    harga = df.set_index(date_column)[commodity_column].astype(float).sort_index()
    harga = harga[~harga.index.duplicated(keep="last")]

    if split_method == "date" and train_end_date:
        cutoff = pd.Timestamp(train_end_date)
        train_series = harga.loc[harga.index <= cutoff].copy()
        test_series = harga.loc[harga.index > cutoff].copy()
    else:
        split_index = int(len(harga) * train_ratio / 100)
        train_series = harga.iloc[:split_index].copy()
        test_series = harga.iloc[split_index:].copy()

    if len(train_series) == 0 or len(test_series) == 0:
        raise ValueError("Pembagian data menghasilkan train atau test kosong.")

    effective_train_end = train_series.index.max()
    effective_test_start = test_series.index.min()
    end_date = harga.index.max()

    supervised = make_time_series_features(harga, max_lag=MAX_LAG)
    X_all = supervised.drop(columns="target")
    y_all = supervised["target"]

    train_mask = X_all.index <= effective_train_end
    test_mask = (X_all.index >= effective_test_start) & (X_all.index <= end_date)

    X_train = X_all.loc[train_mask].copy()
    y_train = y_all.loc[train_mask].copy()
    X_test = X_all.loc[test_mask].copy()
    y_test = y_all.loc[test_mask].copy()

    if len(X_train) < 100 or len(X_test) < 10:
        raise ValueError(f"Data supervised tidak cukup. Train={len(X_train)}, Test={len(X_test)}")

    rf_iterations = RF_ITERATIONS_BY_PROFILE.get(rf_profile, 24)
    effective_cv = min(cv_splits, max(2, len(X_train) // 60))
    effective_cv = min(effective_cv, len(X_train) - 1)

    rf_search_space = {
        "n_estimators": randint(300, 1001),
        "max_depth": [None, 5, 8, 12, 16, 24, 32],
        "min_samples_split": randint(2, 16),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
        "bootstrap": [True],
    }

    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=rf_search_space,
        n_iter=rf_iterations,
        scoring="neg_root_mean_squared_error",
        cv=valid_tscv(len(X_train), effective_cv),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    rf_test_pred = pd.Series(best_rf.predict(X_test), index=X_test.index)
    metrics = evaluate_prediction(y_test, rf_test_pred, y_train, "Random Forest")

    rf_absolute_error = (y_test - rf_test_pred).abs()
    var90 = float(rf_absolute_error.quantile(0.90))
    var95 = float(rf_absolute_error.quantile(0.95))
    var99 = float(rf_absolute_error.quantile(0.99))

    var_table = []
    for day in range(1, 6):
        var_table.append({
            "day": day,
            "VaR_90": var90 * float(np.sqrt(day)),
            "VaR_95": var95 * float(np.sqrt(day)),
            "VaR_99": var99 * float(np.sqrt(day)),
        })

    prediction_detail = [
        {
            "date": str(idx.date()),
            "actual": float(y_test[idx]),
            "predicted": float(rf_test_pred[idx]),
            "error": float(y_test[idx] - rf_test_pred[idx]),
            "absolute_error": float(abs(y_test[idx] - rf_test_pred[idx])),
        }
        for idx in y_test.index
    ]

    model_path = os.path.join(MODEL_DIR, f"{session_id}.joblib")
    joblib.dump(best_rf, model_path)

    return {
        "model_path": model_path,
        "metrics": metrics,
        "best_params": rf_search.best_params_,
        "best_cv_rmse": float(-rf_search.best_score_),
        "train_size": len(train_series),
        "test_size": len(test_series),
        "train_end": str(effective_train_end.date()),
        "test_start": str(effective_test_start.date()),
        "var": {"var90": var90, "var95": var95, "var99": var99},
        "var_table": var_table,
        "prediction_detail": prediction_detail,
        "commodity_column": commodity_column,
        "rf_profile": rf_profile,
        "rf_iterations": rf_iterations,
        "effective_cv": effective_cv,
    }
