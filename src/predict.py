"""
Logika prediksi & rekomendasi restock (dipakai oleh API).

Memprediksi kebutuhan tiap part untuk MINGGU DEPAN berdasarkan
riwayat di data/processed/mingguan.csv dan model di models/model.joblib.
"""
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "meta.json"
MINGGUAN_PATH = ROOT / "data" / "processed" / "mingguan.csv"
LOG_PATH = ROOT / "data" / "prediksi_log.csv"

SAFETY = 0.20          # stok pengaman 20% di atas prediksi
BATAS_BERLEBIH = 2.5   # stok > 2.5x kebutuhan dianggap berlebih


def muat_meta():
    with open(META_PATH) as f:
        return json.load(f)


def muat_model():
    return joblib.load(MODEL_PATH)


def muat_mingguan():
    return pd.read_csv(MINGGUAN_PATH, parse_dates=["minggu"])


def fitur_minggu_depan(mingguan, meta):
    """Bangun baris fitur untuk minggu setelah data terakhir, per part."""
    minggu_target = mingguan["minggu"].max() + pd.Timedelta(days=7)
    baris = []
    for part, g in mingguan.groupby("part"):
        s = g.sort_values("minggu")["jumlah"].astype(float).values
        if len(s) < 8 or part not in meta["peta_part"]:
            continue
        baris.append({
            "part": part,
            "lag_1": s[-1], "lag_2": s[-2], "lag_3": s[-3], "lag_4": s[-4],
            "rata4": s[-4:].mean(),
            "rata8": s[-8:].mean(),
            "std4": pd.Series(s[-4:]).std(),
            "minggu_ke": int(minggu_target.isocalendar().week),
            "bulan": int(minggu_target.month),
            "musim_hujan": int(minggu_target.month in (11, 12, 1, 2, 3)),
            "part_id": meta["peta_part"][part],
        })
    return pd.DataFrame(baris), minggu_target


def simpan_log(df):
    """Simpan prediksi supaya nanti bisa dibandingkan dengan data aktual."""
    log = df[["minggu_target", "part", "prediksi"]].copy()
    log["dibuat"] = datetime.now().isoformat(timespec="seconds")
    if LOG_PATH.exists():
        lama = pd.read_csv(LOG_PATH)
        log = pd.concat([lama, log], ignore_index=True)
    log = log.drop_duplicates(["minggu_target", "part"], keep="last")
    log.to_csv(LOG_PATH, index=False)


def prediksi_minggu_depan(simpan=True):
    meta = muat_meta()
    model = muat_model()
    fitur, minggu_target = fitur_minggu_depan(muat_mingguan(), meta)
    pred = np.clip(model.predict(fitur[meta["fitur"]]), 0, None)
    hasil = pd.DataFrame({
        "minggu_target": minggu_target.strftime("%Y-%m-%d"),
        "part": fitur["part"].values,
        "prediksi": np.round(pred, 2),
    })
    if simpan:
        simpan_log(hasil)
    return hasil


def rekomendasi(stok=None):
    """Bandingkan kebutuhan (prediksi + stok pengaman) dengan stok saat ini."""
    stok = stok or {}
    hasil = prediksi_minggu_depan()
    baris = []
    for _, r in hasil.iterrows():
        butuh = math.ceil(r["prediksi"] * (1 + SAFETY))
        s = stok.get(r["part"])
        if s is None:
            status, restock = "STOK_TIDAK_DIKETAHUI", butuh
        elif s < butuh:
            status, restock = "RESTOCK", butuh - int(s)
        elif butuh > 0 and s > BATAS_BERLEBIH * butuh:
            status, restock = "STOK_BERLEBIH", 0
        else:
            status, restock = "AMAN", 0
        baris.append({
            "minggu_target": r["minggu_target"],
            "part": r["part"],
            "prediksi": float(r["prediksi"]),
            "kebutuhan_dengan_pengaman": int(butuh),
            "stok_saat_ini": None if s is None else int(s),
            "status": status,
            "jumlah_restock": int(restock),
        })
    return baris
