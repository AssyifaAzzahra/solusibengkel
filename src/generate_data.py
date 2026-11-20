import os
import numpy as np
import pandas as pd

SEED = 42
START = "2024-01-01"
END = "2025-12-31"
OUT_PATH = "data/servis_bengkel.csv"

rng = np.random.default_rng(SEED)

# jenis kendaraan: bobot kemunculan
KENDARAAN = {
    "motor_matic": 0.45,
    "motor_bebek": 0.25,
    "motor_sport": 0.10,
    "mobil_city": 0.20,
}
# kendaraan baru muncul mulai tanggal ini (untuk simulasi drift)
KENDARAAN_BARU = "mobil_hybrid"
TANGGAL_KENDARAAN_BARU = pd.Timestamp("2025-07-01")

# jenis servis: bobot, dan daftar part dengan probabilitas dipakai & rentang jumlah
SERVIS = {
    "servis_rutin": {
        "bobot": 0.50,
        "part": {
            "oli_mesin": (0.95, (1, 4)),
            "filter_oli": (0.40, (1, 1)),
            "busi": (0.20, (1, 4)),
        },
    },
    "ganti_rem": {
        "bobot": 0.12,
        "part": {
            "kampas_rem": (0.95, (1, 2)),
            "minyak_rem": (0.30, (1, 1)),
        },
    },
    "servis_berkala": {
        "bobot": 0.15,
        "part": {
            "oli_mesin": (0.90, (1, 4)),
            "filter_udara": (0.60, (1, 1)),
            "busi": (0.50, (1, 4)),
            "filter_oli": (0.50, (1, 1)),
        },
    },
    "kelistrikan": {
        "bobot": 0.10,
        "part": {
            "aki": (0.50, (1, 1)),
            "bohlam_lampu": (0.50, (1, 2)),
            "wiper": (0.15, (1, 2)),
        },
    },
    "transmisi_ban": {
        "bobot": 0.13,
        "part": {
            "v_belt": (0.35, (1, 1)),
            "rantai_gear": (0.25, (1, 1)),
            "ban": (0.40, (1, 2)),
        },
    },
}

# part yang naik saat musim hujan
PART_MUSIM_HUJAN = {"aki": 1.8, "wiper": 2.2, "kampas_rem": 1.4, "bohlam_lampu": 1.3}

# Lebaran (perkiraan) -> lonjakan servis dua minggu sebelumnya
LEBARAN = [pd.Timestamp("2024-04-10"), pd.Timestamp("2025-03-31")]


def faktor_hari(tanggal: pd.Timestamp) -> float:
    """Faktor jumlah kendaraan datang pada hari tertentu."""
    f = 1.0
    # akhir pekan lebih ramai, Minggu tutup
    if tanggal.day_of_week == 6:
        return 0.0
    if tanggal.day_of_week == 5:
        f *= 1.4
    # tren pertumbuhan pelanggan pelan-pelan
    hari_ke = (tanggal - pd.Timestamp(START)).days
    f *= 1 + 0.0004 * hari_ke
    # lonjakan sebelum Lebaran
    for lb in LEBARAN:
        selisih = (lb - tanggal).days
        if 0 <= selisih <= 14:
            f *= 1.6
    return f


def musim_hujan(tanggal: pd.Timestamp) -> bool:
    return tanggal.month in (11, 12, 1, 2, 3)


def pilih(dict_bobot: dict) -> str:
    kunci = list(dict_bobot.keys())
    p = np.array(list(dict_bobot.values()), dtype=float)
    return rng.choice(kunci, p=p / p.sum())


def main():
    tanggal_range = pd.date_range(START, END, freq="D")
    baris = []

    for tgl in tanggal_range:
        f = faktor_hari(tgl)
        if f == 0:
            continue
        n_kendaraan = rng.poisson(9 * f)  # rata-rata ~9 kendaraan/hari

        # siapkan bobot kendaraan (tambah kendaraan baru setelah tanggalnya)
        bobot_kend = dict(KENDARAAN)
        if tgl >= TANGGAL_KENDARAAN_BARU:
            bobot_kend[KENDARAAN_BARU] = 0.12

        for _ in range(n_kendaraan):
            kend = pilih(bobot_kend)
            nama_servis = pilih({k: v["bobot"] for k, v in SERVIS.items()})
            daftar_part = SERVIS[nama_servis]["part"]

            for part, (prob, (lo, hi)) in daftar_part.items():
                p = prob
                if musim_hujan(tgl) and part in PART_MUSIM_HUJAN:
                    p = min(0.98, p * PART_MUSIM_HUJAN[part])
                if rng.random() > p:
                    continue

                jumlah = int(rng.integers(lo, hi + 1))

                # penyesuaian per jenis kendaraan
                if part in ("oli_mesin", "busi") and kend.startswith("mobil"):
                    jumlah = max(jumlah, 3)
                if part == "v_belt" and kend not in ("motor_matic",):
                    continue
                if part == "rantai_gear" and kend not in ("motor_bebek", "motor_sport"):
                    continue
                if part == "wiper" and kend.startswith("motor"):
                    continue
                # kendaraan hybrid: lebih sedikit oli, lebih banyak kampas rem
                if kend == KENDARAAN_BARU:
                    if part == "oli_mesin":
                        jumlah = max(1, jumlah - 1)
                    if part == "kampas_rem":
                        jumlah += 1

                baris.append((tgl.strftime("%Y-%m-%d"), kend, nama_servis, part, jumlah))

    df = pd.DataFrame(
        baris, columns=["tanggal", "jenis_kendaraan", "jenis_servis", "part", "jumlah"]
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Selesai: {len(df)} baris -> {OUT_PATH}")
    print("\nTotal pemakaian per part:")
    print(df.groupby("part")["jumlah"].sum().sort_values(ascending=False))
    print("\nContoh data:")
    print(df.head())


if __name__ == "__main__":
    main()