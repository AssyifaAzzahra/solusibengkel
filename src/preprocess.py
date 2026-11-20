"""
Preprocessing & feature engineering.

Input  : data/servis_bengkel.csv
Output : data/processed/mingguan.csv  (pemakaian per part per minggu)
         data/processed/fitur.csv     (siap dipakai untuk training)

Jalankan dari root proyek:
    python src/preprocess.py
"""
import os
import pandas as pd

IN_PATH = "data/servis_bengkel.csv"
OUT_DIR = "data/processed"
LAGS = [1, 2, 3, 4]


def bersihkan(df):
    n_awal = len(df)
    df = df.copy()

    for kolom in ["jenis_kendaraan", "jenis_servis", "part"]:
        df[kolom] = df[kolom].astype(str).str.strip().str.lower()

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce")

    df = df.dropna(subset=["tanggal", "part", "jumlah"])
    df = df[df["jumlah"] > 0]
    df["jumlah"] = df["jumlah"].astype(int)

    # Catatan: baris yang tampak kembar TIDAK dihapus, karena beberapa
    # kendaraan di hari yang sama bisa memakai part yang sama.
    print(f"Pembersihan: {n_awal} -> {len(df)} baris")
    return df


def agregasi_mingguan(df):
    df = df.copy()
    # awal minggu = hari Senin
    df["minggu"] = df["tanggal"].dt.to_period("W-SUN").dt.start_time

    # buang minggu terakhir kalau belum lengkap (data berhenti sebelum Sabtu)
    tgl_akhir = df["tanggal"].max()
    minggu_akhir = df["minggu"].max()
    if tgl_akhir < minggu_akhir + pd.Timedelta(days=5):
        df = df[df["minggu"] != minggu_akhir]
        print(f"Minggu terakhir ({minggu_akhir.date()}) belum lengkap, dibuang")

    agg = df.groupby(["minggu", "part"], as_index=False)["jumlah"].sum()

    # lengkapi semua kombinasi minggu x part, minggu tanpa pemakaian = 0
    semua_minggu = pd.date_range(agg["minggu"].min(), agg["minggu"].max(), freq="7D")
    semua_part = sorted(agg["part"].unique())
    grid = pd.MultiIndex.from_product([semua_minggu, semua_part], names=["minggu", "part"])
    agg = (
        agg.set_index(["minggu", "part"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )
    return agg


def buat_fitur(agg):
    agg = agg.sort_values(["part", "minggu"]).reset_index(drop=True)
    g = agg.groupby("part")["jumlah"]

    # lag: pemakaian 1-4 minggu sebelumnya
    for k in LAGS:
        agg[f"lag_{k}"] = g.shift(k)

    # rata-rata & simpangan bergerak, digeser 1 minggu supaya tidak bocor
    # (tidak memakai nilai minggu yang sedang diprediksi)
    agg["rata4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    agg["rata8"] = g.transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
    agg["std4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=4).std())

    # fitur kalender
    agg["minggu_ke"] = agg["minggu"].dt.isocalendar().week.astype(int)
    agg["bulan"] = agg["minggu"].dt.month
    agg["musim_hujan"] = agg["bulan"].isin([11, 12, 1, 2, 3]).astype(int)

    # kode numerik untuk part
    daftar_part = sorted(agg["part"].unique())
    peta = {p: i for i, p in enumerate(daftar_part)}
    agg["part_id"] = agg["part"].map(peta)

    # buang baris awal yang lag-nya belum tersedia
    n_awal = len(agg)
    agg = agg.dropna(subset=[f"lag_{k}" for k in LAGS] + ["rata4", "rata8", "std4"])
    print(f"Fitur: {n_awal} -> {len(agg)} baris (baris awal tanpa lag dibuang)")
    return agg.reset_index(drop=True), peta


def main():
    if not os.path.exists(IN_PATH):
        raise SystemExit(f"File {IN_PATH} tidak ditemukan. Jalankan generate_data.py dulu.")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH)
    df = bersihkan(df)

    mingguan = agregasi_mingguan(df)
    mingguan.to_csv(f"{OUT_DIR}/mingguan.csv", index=False)

    fitur, peta = buat_fitur(mingguan)
    fitur.to_csv(f"{OUT_DIR}/fitur.csv", index=False)

    print(f"\nRentang minggu : {fitur['minggu'].min().date()} s/d {fitur['minggu'].max().date()}")
    print(f"Jumlah part    : {len(peta)}")
    print(f"Disimpan       : {OUT_DIR}/mingguan.csv, {OUT_DIR}/fitur.csv")
    print("\nContoh fitur (oli_mesin):")
    print(fitur[fitur["part"] == "oli_mesin"].head())


if __name__ == "__main__":
    main()
