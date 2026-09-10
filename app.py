import io
import threading
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from core import descriptive_analysis, train_model

app = Flask(__name__)

# In-memory store: session_id -> data
_store: dict = {}
_lock = threading.Lock()


def _get_store(session_id: str) -> dict:
    with _lock:
        return _store.get(session_id, {})


def _set_store(session_id: str, data: dict):
    with _lock:
        _store[session_id] = data


# ======================================================================
# UPLOAD DATASET
# POST /upload
# Form-data: file (CSV/Excel), date_column, commodity_column,
#            start_date (optional), end_date (optional)
# ======================================================================

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan."}), 400

    file = request.files["file"]
    date_column = request.form.get("date_column")
    commodity_column = request.form.get("commodity_column")

    if not date_column or not commodity_column:
        return jsonify({"error": "date_column dan commodity_column wajib diisi."}), 400

    try:
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Format file tidak didukung. Gunakan CSV atau Excel."}), 400

        if date_column not in df.columns:
            return jsonify({"error": f"Kolom '{date_column}' tidak ditemukan."}), 400
        if commodity_column not in df.columns:
            return jsonify({"error": f"Kolom '{commodity_column}' tidak ditemukan."}), 400

        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors="coerce")
        df = df.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)

        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        if start_date:
            df = df[df[date_column].dt.date >= pd.Timestamp(start_date).date()]
        if end_date:
            df = df[df[date_column].dt.date <= pd.Timestamp(end_date).date()]

        df = df.reset_index(drop=True)

        session_id = str(uuid.uuid4())
        _set_store(session_id, {
            "df": df.to_json(date_format="iso"),
            "date_column": date_column,
            "commodity_column": commodity_column,
        })

        return jsonify({
            "session_id": session_id,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "date_range": {
                "start": str(df[date_column].min().date()),
                "end": str(df[date_column].max().date()),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================================================================
# DESCRIPTIVE ANALYSIS
# GET /descriptive?session_id=...
# ======================================================================

@app.route("/descriptive", methods=["GET"])
def descriptive():
    session_id = request.args.get("session_id")
    store = _get_store(session_id)

    if not store:
        return jsonify({"error": "Session tidak ditemukan. Silakan upload dataset terlebih dahulu."}), 404

    try:
        df = pd.read_json(io.StringIO(store["df"]))
        date_column = store["date_column"]
        commodity_column = store["commodity_column"]

        result = descriptive_analysis(df, date_column, commodity_column)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================================================================
# TRAIN MODEL (background thread)
# POST /train
# JSON body:
#   session_id, split_method ("percentage"|"date"),
#   train_ratio (int, default 80), train_end_date (str, optional),
#   rf_profile ("Fast"|"Balanced"|"Thorough"), cv_splits (int)
# ======================================================================

def _run_training(session_id: str, df, date_column, commodity_column, kwargs):
    try:
        result = train_model(
            df=df,
            date_column=date_column,
            commodity_column=commodity_column,
            session_id=session_id,
            **kwargs,
        )
        store = _get_store(session_id)
        store["train_result"] = {k: v for k, v in result.items() if k != "model"}
        store["train_status"] = "done"
        _set_store(session_id, store)
    except Exception as e:
        store = _get_store(session_id)
        store["train_status"] = "error"
        store["train_error"] = str(e)
        _set_store(session_id, store)


@app.route("/train", methods=["POST"])
def train():
    body = request.get_json(force=True) or {}
    session_id = body.get("session_id")
    store = _get_store(session_id)

    if not store:
        return jsonify({"error": "Session tidak ditemukan. Silakan upload dataset terlebih dahulu."}), 404

    if store.get("train_status") == "running":
        return jsonify({"message": "Training sedang berjalan.", "status": "running"}), 202

    try:
        df = pd.read_json(io.StringIO(store["df"]))
        date_column = store["date_column"]
        commodity_column = store["commodity_column"]

        store["train_status"] = "running"
        store["train_result"] = None
        store["train_error"] = None
        _set_store(session_id, store)

        kwargs = {
            "split_method": body.get("split_method", "percentage"),
            "train_ratio": int(body.get("train_ratio", 80)),
            "train_end_date": body.get("train_end_date"),
            "rf_profile": body.get("rf_profile", "Balanced"),
            "cv_splits": int(body.get("cv_splits", 5)),
        }

        t = threading.Thread(
            target=_run_training,
            args=(session_id, df, date_column, commodity_column, kwargs),
            daemon=True,
        )
        t.start()

        return jsonify({"message": "Training dimulai.", "status": "running", "session_id": session_id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================================================================
# TRAIN STATUS
# GET /train/status?session_id=...
# ======================================================================

@app.route("/train/status", methods=["GET"])
def train_status():
    session_id = request.args.get("session_id")
    store = _get_store(session_id)

    if not store:
        return jsonify({"error": "Session tidak ditemukan."}), 404

    status = store.get("train_status", "idle")

    if status == "done":
        return jsonify({"status": "done", "result": store.get("train_result")})
    elif status == "error":
        return jsonify({"status": "error", "error": store.get("train_error")}), 500
    else:
        return jsonify({"status": status})


# ======================================================================
# OUTPUT (hasil training yang sudah tersimpan)
# GET /output?session_id=...
# ======================================================================

@app.route("/output", methods=["GET"])
def output():
    session_id = request.args.get("session_id")
    store = _get_store(session_id)

    if not store:
        return jsonify({"error": "Session tidak ditemukan."}), 404

    status = store.get("train_status", "idle")
    if status != "done":
        return jsonify({"error": "Model belum selesai dilatih.", "status": status}), 404

    return jsonify(store.get("train_result"))


# ======================================================================
# PREDICT (pakai model tersimpan di disk)
# POST /predict
# JSON body: session_id, data (list of {date, price})
# ======================================================================

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True) or {}
    session_id = body.get("session_id")
    store = _get_store(session_id)

    if not store:
        return jsonify({"error": "Session tidak ditemukan."}), 404

    train_result = store.get("train_result")
    if not train_result:
        return jsonify({"error": "Model belum dilatih."}), 404

    model_path = train_result.get("model_path")
    if not model_path or not __import__("os").path.exists(model_path):
        return jsonify({"error": "File model tidak ditemukan di disk."}), 404

    try:
        from core import make_time_series_features, MAX_LAG
        import numpy as np

        model = joblib.load(model_path)

        raw_data = body.get("data", [])
        if not raw_data:
            return jsonify({"error": "Field 'data' wajib diisi."}), 400

        df_input = pd.DataFrame(raw_data)
        df_input["date"] = pd.to_datetime(df_input["date"])
        df_input = df_input.sort_values("date").set_index("date")["price"].astype(float)

        features = make_time_series_features(df_input, max_lag=MAX_LAG)
        X = features.drop(columns="target")

        preds = model.predict(X)

        return jsonify({
            "predictions": [
                {"date": str(idx.date()), "predicted": float(p)}
                for idx, p in zip(X.index, preds)
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================================================================
# HEALTH CHECK
# ======================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "PanganCast API"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
