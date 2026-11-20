"""
Training model prediksi kebutuhan part per minggu + experiment tracking MLflow.

Input  : data/processed/fitur.csv
Output : models/model.joblib   (model terbaik, dipakai API nanti)
         models/meta.json      (daftar fitur, peta part, metrik)
         mlflow.db / mlruns/   (catatan eksperimen MLflow)

Jalankan dari root proyek:
    python src/train.py
Lihat hasil eksperimen:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import os
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

IN_PATH = "data/processed/fitur.csv"
MODEL_DIR = "models"
EXPERIMENT = "restock-sparepart"
REGISTERED_NAME = "restock-model"
TARGET = "jumlah"
FITUR = [
    "lag_1", "lag_2", "lag_3", "lag_4",
    "rata4", "rata8", "std4",
    "minggu_ke", "bulan", "musim_hujan", "part_id",
]
N_MINGGU_UJI = 12  # 12 minggu terakhir dipakai untuk menguji


def split_waktu(df):
    """Split berdasarkan waktu (bukan acak) supaya tidak bocor ke masa depan."""
    minggu = sorted(df["minggu"].unique())
    batas = minggu[-N_MINGGU_UJI]
    train = df[df["minggu"] < batas]
    test = df[df["minggu"] >= batas]
    return train, test


def hitung_metrik(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # WAPE: total error dibanding total pemakaian aktual (dalam persen)
    wape = float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1) * 100)
    return {"mae": float(mae), "rmse": rmse, "wape": wape}


def main():
    if not os.path.exists(IN_PATH):
        raise SystemExit(f"{IN_PATH} tidak ada. Jalankan src/preprocess.py dulu.")

    df = pd.read_csv(IN_PATH, parse_dates=["minggu"])
    train, test = split_waktu(df)
    X_tr, y_tr = train[FITUR], train[TARGET]
    X_te, y_te = test[FITUR], test[TARGET]
    print(f"Train: {len(train)} baris | Test: {len(test)} baris")
    print(f"Uji dimulai minggu: {test['minggu'].min().date()}\n")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    hasil = {}

    # 1) Baseline: tebakan = rata-rata 4 minggu terakhir
    with mlflow.start_run(run_name="baseline_rata4"):
        m = hitung_metrik(y_te.values, X_te["rata4"].values)
        mlflow.log_param("model", "baseline_rata4")
        mlflow.log_metrics(m)
        hasil["baseline_rata4"] = m
        print(f"baseline_rata4         MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  WAPE={m['wape']:.1f}%")

    # 2) Kandidat model machine learning
    kandidat = {
        "rf_100_d8": RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "rf_300_d12": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "gb_150_lr005": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
        "gb_300_lr003": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42),
    }

    model_terbaik, nama_terbaik, mae_terbaik = None, None, float("inf")

    for nama, model in kandidat.items():
        with mlflow.start_run(run_name=nama):
            model.fit(X_tr, y_tr)
            pred = np.clip(model.predict(X_te), 0, None)
            m = hitung_metrik(y_te.values, pred)

            mlflow.log_param("model", type(model).__name__)
            mlflow.log_params(model.get_params())
            mlflow.log_param("n_train", len(train))
            mlflow.log_param("n_test", len(test))
            mlflow.log_metrics(m)
            hasil[nama] = m
            print(f"{nama:<22} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  WAPE={m['wape']:.1f}%")

            if m["mae"] < mae_terbaik:
                model_terbaik, nama_terbaik, mae_terbaik = model, nama, m["mae"]

    # 3) Simpan & daftarkan model terbaik
    print(f"\nModel terbaik: {nama_terbaik} (MAE={mae_terbaik:.3f})")
    base = hasil["baseline_rata4"]["mae"]
    perbaikan = (base - mae_terbaik) / base * 100
    print(f"Dibanding baseline: MAE turun {perbaikan:.1f}%")

    with mlflow.start_run(run_name=f"FINAL_{nama_terbaik}"):
        mlflow.log_param("model", nama_terbaik)
        mlflow.log_metrics(hasil[nama_terbaik])
        mlflow.sklearn.log_model(
            model_terbaik,
            name="model",
            registered_model_name=REGISTERED_NAME,
            serialization_format="cloudpickle",
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_terbaik, f"{MODEL_DIR}/model.joblib")

    peta_part = (
        df[["part", "part_id"]].drop_duplicates().sort_values("part_id")
        .set_index("part")["part_id"].to_dict()
    )
    meta = {
        "nama_model": nama_terbaik,
        "dilatih_pada": datetime.now().isoformat(timespec="seconds"),
        "fitur": FITUR,
        "peta_part": {k: int(v) for k, v in peta_part.items()},
        "metrik_uji": hasil[nama_terbaik],
        "metrik_baseline": hasil["baseline_rata4"],
        "minggu_latih_terakhir": str(train["minggu"].max().date()),
    }
    with open(f"{MODEL_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTersimpan: {MODEL_DIR}/model.joblib, {MODEL_DIR}/meta.json")
    print("Lihat eksperimen: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
