"""
Simulasi data baru dengan pola yang berubah (untuk demo drift & retraining).

Skenario (tahun 2026): bengkel makin ramai (volume x1.8), muncul jenis kendaraan
baru 'mobil_listrik' (tanpa oli/busi/filter), dan kampas rem + ban lebih sering diganti.

Jalankan dari root proyek:
    python src/simulasi_drift.py --minggu 20   # tambah 20 minggu data baru
    python src/simulasi_drift.py --reset       # kembalikan ke data asli
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate_data as g  # noqa: E402

RAW = ROOT / "data" / "servis_bengkel.csv"
ASLI = ROOT / "data" / "servis_bengkel_asli.csv"
LOG = ROOT / "data" / "prediksi_log.csv"
MONITORING = ROOT / "data" / "monitoring"

MULAI = pd.Timestamp("2026-01-01")
FAKTOR_VOLUME = 1.8
PORSI_LISTRIK = 0.20
KENDARAAN_BARU = "mobil_listrik"
PART_MESIN = {"oli_mesin", "busi", "filter_oli", "filter_udara"}
BOOST_PART = {"kampas_rem": 2.0, "ban": 1.5}


def pilih(rng, dict_bobot):
    kunci = list(dict_bobot.keys())
    p = np.array(list(dict_bobot.values()), dtype=float)
    return rng.choice(kunci, p=p / p.sum())


def buat_data_baru(n_minggu, mulai=MULAI, seed=7):
    rng = np.random.default_rng(seed)
    campuran = dict(g.KENDARAAN)
    campuran[g.KENDARAAN_BARU] = 0.12  # mobil_hybrid tetap ada
    bobot = {k: v * (1 - PORSI_LISTRIK) for k, v in campuran.items()}
    bobot[KENDARAAN_BARU] = PORSI_LISTRIK
    bobot_servis = {k: v["bobot"] for k, v in g.SERVIS.items()}

    baris = []
    for hari in range(n_minggu * 7):
        tgl = mulai + pd.Timedelta(days=hari)
        if tgl.day_of_week == 6:  # Minggu tutup
            continue
        f = 1.4 if tgl.day_of_week == 5 else 1.0
        for _ in range(rng.poisson(9 * f * FAKTOR_VOLUME)):
            kend = pilih(rng, bobot)
            servis = pilih(rng, bobot_servis)
            for part, (prob, (lo, hi)) in g.SERVIS[servis]["part"].items():
                if kend == KENDARAAN_BARU and part in PART_MESIN:
                    continue
                if part == "v_belt" and kend != "motor_matic":
                    continue
                if part == "rantai_gear" and kend not in ("motor_bebek", "motor_sport"):
                    continue
                if part == "wiper" and kend.startswith("motor"):
                    continue
                if rng.random() > min(0.98, prob * BOOST_PART.get(part, 1.0)):
                    continue
                jumlah = int(rng.integers(lo, hi + 1))
                if part in ("oli_mesin", "busi") and kend.startswith("mobil"):
                    jumlah = max(jumlah, 3)
                baris.append((tgl.strftime("%Y-%m-%d"), kend, servis, part, jumlah))

    return pd.DataFrame(
        baris, columns=["tanggal", "jenis_kendaraan", "jenis_servis", "part", "jumlah"]
    )


def bersihkan_jejak():
    """Log prediksi & laporan lama dihapus karena skenario data berubah."""
    if LOG.exists():
        LOG.unlink()
    if MONITORING.exists():
        shutil.rmtree(MONITORING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minggu", type=int, default=20)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        if not ASLI.exists():
            raise SystemExit("Data asli belum dicadangkan, tidak ada yang perlu direset.")
        shutil.copy(ASLI, RAW)
        bersihkan_jejak()
        print(f"Data dikembalikan ke data asli: {RAW}")
        return

    if not RAW.exists():
        raise SystemExit(f"{RAW} tidak ada. Jalankan src/generate_data.py dulu.")
    if not ASLI.exists():
        shutil.copy(RAW, ASLI)  # cadangan data asli, hanya sekali

    asli = pd.read_csv(ASLI)
    baru = buat_data_baru(args.minggu)
    gabung = pd.concat([asli, baru], ignore_index=True)
    gabung.to_csv(RAW, index=False)
    bersihkan_jejak()

    print(f"Data asli : {len(asli)} baris (cadangan: {ASLI.name})")
    print(f"Data baru : {len(baru)} baris, {args.minggu} minggu mulai {MULAI.date()}")
    print(f"Total     : {len(gabung)} baris -> {RAW}")
    print("Log prediksi dan laporan monitoring lama dihapus (skenario baru).")


if __name__ == "__main__":
    main()
