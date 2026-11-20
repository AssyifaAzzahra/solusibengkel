"""
Monitoring model: performa (prediksi vs aktual) dan data drift.

Membaca : data/prediksi_log.csv, data/processed/mingguan.csv,
          data/servis_bengkel.csv, models/meta.json
Menulis : data/monitoring/laporan.json, data/monitoring/per_minggu.csv

Jalankan dari root proyek:
    python src/monitor.py --backfill 12   # isi log prediksi 12 minggu terakhir (simulasi), lalu monitor
    python src/monitor.py                 # monitor dari log yang sudah ada
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import predict  # noqa: E402

RAW_PATH = ROOT / "data" / "servis_bengkel.csv"
OUT_DIR = ROOT / "data" / "monitoring"

WINDOW = 12             # jumlah minggu terakhir yang dipantau
KS_ALPHA = 0.05         # p-value di bawah ini = distribusi berubah
GESER_KENDARAAN = 0.05  # pergeseran porsi jenis kendaraan > 5 poin = drift
BATAS_NAIK_WAPE = 1.25  # WAPE terbaru > 1.25x WAPE saat uji = performa turun
BATAS_SHARE_DRIFT = 0.30  # >30% part mengalami drift = disarankan retraining


def backfill(n_minggu):
    """Simulasi penggunaan nyata: prediksi tiap minggu hanya dari data SEBELUM minggu itu."""
    meta = predict.muat_meta()
    model = predict.muat_model()
    mingguan = predict.muat_mingguan()
    daftar = sorted(mingguan["minggu"].unique())[-n_minggu:]
    hasil = []
    for w in daftar:
        fitur, target = predict.fitur_minggu_depan(mingguan[mingguan["minggu"] < w], meta)
        pred = np.clip(model.predict(fitur[meta["fitur"]]), 0, None)
        hasil.append(pd.DataFrame({
            "minggu_target": target.strftime("%Y-%m-%d"),
            "part": fitur["part"].values,
            "prediksi": np.round(pred, 2),
        }))
    predict.simpan_log(pd.concat(hasil, ignore_index=True))
    print(f"Backfill: {len(daftar)} minggu x {len(hasil[0])} part masuk ke log prediksi")


def evaluasi(log, mingguan):
    """Bandingkan prediksi di log dengan pemakaian aktual."""
    log = log.copy()
    log["minggu_target"] = pd.to_datetime(log["minggu_target"])
    gab = log.merge(
        mingguan, left_on=["minggu_target", "part"], right_on=["minggu", "part"]
    )
    if gab.empty:
        return None, pd.DataFrame()
    gab["abs_err"] = (gab["jumlah"] - gab["prediksi"]).abs()

    per_minggu = gab.groupby("minggu_target").agg(
        mae=("abs_err", "mean"),
        total_aktual=("jumlah", "sum"),
        total_error=("abs_err", "sum"),
    ).reset_index()
    per_minggu["wape"] = per_minggu["total_error"] / per_minggu["total_aktual"].clip(lower=1) * 100

    ringkas = {
        "n_prediksi": int(len(gab)),
        "n_minggu": int(per_minggu.shape[0]),
        "mae": float(gab["abs_err"].mean()),
        "wape": float(gab["abs_err"].sum() / max(gab["jumlah"].sum(), 1) * 100),
    }
    return ringkas, per_minggu


def drift_jumlah(mingguan, window=WINDOW):
    """Uji KS per part: apakah pola pemakaian minggu-minggu terbaru berubah?"""
    batas = sorted(mingguan["minggu"].unique())[-window]
    ref = mingguan[mingguan["minggu"] < batas]
    cur = mingguan[mingguan["minggu"] >= batas]
    hasil = []
    for part in sorted(mingguan["part"].unique()):
        a = ref[ref["part"] == part]["jumlah"]
        b = cur[cur["part"] == part]["jumlah"]
        if len(a) < 5 or len(b) < 5:
            continue
        p = float(ks_2samp(a, b).pvalue)
        hasil.append({
            "part": part,
            "rata_ref": float(a.mean()),
            "rata_terbaru": float(b.mean()),
            "ks_pvalue": p,
            "drift": bool(p < KS_ALPHA),
        })
    return hasil


def drift_kendaraan(window=WINDOW):
    """Porsi jenis kendaraan: periode terbaru vs sebelumnya (jenis baru = drift)."""
    if not RAW_PATH.exists():
        return []
    raw = pd.read_csv(RAW_PATH, parse_dates=["tanggal"])
    raw["minggu"] = raw["tanggal"].dt.to_period("W-SUN").dt.start_time
    batas = sorted(raw["minggu"].unique())[-window]
    ref = raw[raw["minggu"] < batas]["jenis_kendaraan"].value_counts(normalize=True)
    cur = raw[raw["minggu"] >= batas]["jenis_kendaraan"].value_counts(normalize=True)
    hasil = []
    for k in sorted(set(ref.index) | set(cur.index)):
        r, c = float(ref.get(k, 0.0)), float(cur.get(k, 0.0))
        baru = r == 0.0 and c > 0.0
        hasil.append({
            "jenis_kendaraan": k,
            "porsi_ref": r,
            "porsi_terbaru": c,
            "jenis_baru": bool(baru),
            "drift": bool(baru or abs(c - r) > GESER_KENDARAAN),
        })
    return hasil


def putuskan(ringkas, meta, drift_part, drift_kend):
    alasan = []
    if ringkas is not None:
        batas = meta["metrik_uji"]["wape"] * BATAS_NAIK_WAPE
        if ringkas["wape"] > batas:
            alasan.append(
                f"WAPE terbaru {ringkas['wape']:.1f}% melewati batas {batas:.1f}% "
                f"(WAPE saat uji {meta['metrik_uji']['wape']:.1f}%)"
            )
    if drift_part:
        share = sum(d["drift"] for d in drift_part) / len(drift_part)
        if share > BATAS_SHARE_DRIFT:
            alasan.append(f"{share:.0%} part mengalami drift pola pemakaian")
    baru = [d["jenis_kendaraan"] for d in drift_kend if d["jenis_baru"]]
    if baru:
        alasan.append("Jenis kendaraan baru: " + ", ".join(baru))
    status = "RETRAIN_DISARANKAN" if alasan else "OK"
    return status, alasan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0, help="isi log untuk N minggu terakhir")
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()

    if args.backfill:
        backfill(args.backfill)

    meta = predict.muat_meta()
    mingguan = predict.muat_mingguan()
    log = pd.read_csv(predict.LOG_PATH) if predict.LOG_PATH.exists() else pd.DataFrame()

    ringkas, per_minggu = (None, pd.DataFrame())
    if not log.empty:
        ringkas, per_minggu = evaluasi(log, mingguan)
    if ringkas is None:
        print("Belum ada prediksi yang punya data aktual. Jalankan: python src/monitor.py --backfill 12")

    d_part = drift_jumlah(mingguan, args.window)
    d_kend = drift_kendaraan(args.window)
    status, alasan = putuskan(ringkas, meta, d_part, d_kend)

    laporan = {
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "alasan": alasan,
        "window_minggu": args.window,
        "performa": ringkas,
        "performa_saat_uji": meta["metrik_uji"],
        "drift_part": d_part,
        "drift_kendaraan": d_kend,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "laporan.json", "w") as f:
        json.dump(laporan, f, indent=2)
    if not per_minggu.empty:
        per_minggu.to_csv(OUT_DIR / "per_minggu.csv", index=False)

    print(f"\nSTATUS: {status}")
    for a in alasan:
        print(f" - {a}")
    if ringkas:
        print(f"\nPerforma {ringkas['n_minggu']} minggu terakhir: "
              f"MAE={ringkas['mae']:.2f}  WAPE={ringkas['wape']:.1f}%  "
              f"(saat uji: WAPE={meta['metrik_uji']['wape']:.1f}%)")
    print("\nDrift per part (KS test):")
    for d in d_part:
        tanda = "DRIFT" if d["drift"] else "ok"
        print(f"  {d['part']:<14} p={d['ks_pvalue']:.3f}  rata {d['rata_ref']:.1f} -> {d['rata_terbaru']:.1f}  {tanda}")
    print("\nPorsi jenis kendaraan:")
    for d in d_kend:
        tanda = "BARU" if d["jenis_baru"] else ("DRIFT" if d["drift"] else "ok")
        print(f"  {d['jenis_kendaraan']:<14} {d['porsi_ref']:.1%} -> {d['porsi_terbaru']:.1%}  {tanda}")
    print(f"\nLaporan: {OUT_DIR / 'laporan.json'}")


if __name__ == "__main__":
    main()
