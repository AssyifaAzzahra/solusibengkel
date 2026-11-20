"""
Retraining otomatis: preprocessing -> monitoring -> (jika perlu) latih ulang ->
bandingkan model lama vs baru -> promosikan hanya kalau lebih baik.

Jalankan dari root proyek:
    python src/retrain.py          # retrain hanya jika monitoring menyarankan
    python src/retrain.py --paksa  # retrain walau status monitoring OK
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models"
ARSIP_DIR = MODEL_DIR / "arsip"
LAPORAN = ROOT / "data" / "monitoring" / "laporan.json"

N_MINGGU_UJI = 6  # minggu terakhir untuk membandingkan model lama vs baru
MARGIN = 0.02     # model baru harus minimal 2% lebih baik (MAE)


def layak_promosi(mae_lama, mae_baru, margin=MARGIN):
    """Model baru dipakai hanya jika MAE-nya lebih rendah dari model lama (dengan margin)."""
    return mae_baru <= mae_lama * (1 - margin)


def jalankan(perintah):
    print(f"\n$ python {' '.join(perintah)}")
    subprocess.run([sys.executable, *perintah], cwd=ROOT, check=True)


def latih_ulang():
    import joblib
    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    from src import train

    fitur = train.FITUR
    df = pd.read_csv(ROOT / "data" / "processed" / "fitur.csv", parse_dates=["minggu"])
    minggu = sorted(df["minggu"].unique())
    batas = minggu[-N_MINGGU_UJI]
    tr, uji = df[df["minggu"] < batas], df[df["minggu"] >= batas]
    X_tr, y_tr = tr[fitur], tr[train.TARGET]
    X_te, y_te = uji[fitur], uji[train.TARGET]
    print(f"\nLatih: {len(tr)} baris | Uji: {len(uji)} baris (mulai {uji['minggu'].min().date()})")

    model_lama = joblib.load(MODEL_DIR / "model.joblib")
    m_lama = train.hitung_metrik(y_te.values, np.clip(model_lama.predict(X_te), 0, None))
    m_base = train.hitung_metrik(y_te.values, X_te["rata4"].values)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(train.EXPERIMENT)
    with mlflow.start_run(run_name="retrain_model_lama"):
        mlflow.log_param("model", "model_lama")
        mlflow.log_metrics(m_lama)

    print(f"model lama            MAE={m_lama['mae']:.3f}  WAPE={m_lama['wape']:.1f}%")
    print(f"baseline_rata4        MAE={m_base['mae']:.3f}  WAPE={m_base['wape']:.1f}%")

    kandidat = {
        "gb_150_lr005": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
        "gb_300_lr003": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42),
        "rf_300_d12": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    }
    terbaik, nama_terbaik, m_terbaik = None, None, None
    for nama, model in kandidat.items():
        with mlflow.start_run(run_name=f"retrain_{nama}"):
            model.fit(X_tr, y_tr)
            m = train.hitung_metrik(y_te.values, np.clip(model.predict(X_te), 0, None))
            mlflow.log_param("model", type(model).__name__)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(m)
        print(f"{nama:<21} MAE={m['mae']:.3f}  WAPE={m['wape']:.1f}%")
        if m_terbaik is None or m["mae"] < m_terbaik["mae"]:
            terbaik, nama_terbaik, m_terbaik = model, nama, m

    print(f"\nKandidat terbaik: {nama_terbaik} (MAE={m_terbaik['mae']:.3f}) "
          f"vs model lama (MAE={m_lama['mae']:.3f})")

    if not layak_promosi(m_lama["mae"], m_terbaik["mae"]):
        print("HASIL: model baru TIDAK lebih baik. Model lama tetap dipakai.")
        return False

    # arsipkan model lama
    ARSIP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(MODEL_DIR / "model.joblib", ARSIP_DIR / f"model_{ts}.joblib")
    shutil.copy(MODEL_DIR / "meta.json", ARSIP_DIR / f"meta_{ts}.json")

    # latih ulang konfigurasi terbaik dengan SEMUA data, lalu daftarkan & simpan
    final = clone(terbaik).fit(df[fitur], df[train.TARGET])
    with mlflow.start_run(run_name=f"RETRAIN_FINAL_{nama_terbaik}"):
        mlflow.log_param("model", nama_terbaik)
        mlflow.log_param("n_train", len(df))
        mlflow.log_metrics(m_terbaik)
        mlflow.sklearn.log_model(
            final, name="model",
            registered_model_name=train.REGISTERED_NAME,
            serialization_format="cloudpickle",
        )
    joblib.dump(final, MODEL_DIR / "model.joblib")

    peta_part = (
        df[["part", "part_id"]].drop_duplicates().sort_values("part_id")
        .set_index("part")["part_id"].to_dict()
    )
    meta = {
        "nama_model": nama_terbaik,
        "dilatih_pada": datetime.now().isoformat(timespec="seconds"),
        "fitur": fitur,
        "peta_part": {k: int(v) for k, v in peta_part.items()},
        "metrik_uji": m_terbaik,
        "metrik_baseline": m_base,
        "metrik_model_lama_pada_uji_sama": m_lama,
        "minggu_latih_terakhir": str(df["minggu"].max().date()),
        "hasil_retraining": True,
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    turun = (m_lama["mae"] - m_terbaik["mae"]) / m_lama["mae"] * 100
    print(f"HASIL: model baru DIPROMOSIKAN (MAE turun {turun:.1f}%). Model lama diarsipkan di {ARSIP_DIR}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paksa", action="store_true", help="retrain walau status monitoring OK")
    args = ap.parse_args()

    jalankan(["src/preprocess.py"])
    jalankan(["src/monitor.py", "--backfill", "12"])

    with open(LAPORAN) as f:
        laporan = json.load(f)
    print(f"\nStatus monitoring: {laporan['status']}")
    for a in laporan["alasan"]:
        print(f" - {a}")

    if laporan["status"] == "OK" and not args.paksa:
        print("Tidak perlu retraining.")
        return

    latih_ulang()


if __name__ == "__main__":
    main()
