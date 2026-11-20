import pandas as pd

from src import preprocess


def data_contoh():
    """Data kecil: 12 minggu x 2 part, 6 hari kerja per minggu."""
    baris = []
    mulai = pd.Timestamp("2024-01-01")  # hari Senin
    for hari in range(12 * 7):
        tgl = mulai + pd.Timedelta(days=hari)
        if tgl.dayofweek == 6:
            continue
        baris.append((tgl.strftime("%Y-%m-%d"), "motor_matic", "servis_rutin", "oli_mesin", 2))
        if hari % 7 == 0:
            baris.append((tgl.strftime("%Y-%m-%d"), "mobil_city", "ganti_rem", "kampas_rem", 1))
    return pd.DataFrame(
        baris, columns=["tanggal", "jenis_kendaraan", "jenis_servis", "part", "jumlah"]
    )


def test_bersihkan_buang_data_rusak():
    df = pd.DataFrame({
        "tanggal": ["2024-01-01", "tanggal-salah", "2024-01-02", "2024-01-03"],
        "jenis_kendaraan": ["Motor_Matic ", "motor_matic", "motor_matic", "motor_matic"],
        "jenis_servis": ["servis_rutin"] * 4,
        "part": ["Oli_Mesin", "oli_mesin", "oli_mesin", "oli_mesin"],
        "jumlah": [2, 1, 0, None],
    })
    hasil = preprocess.bersihkan(df)
    assert len(hasil) == 1  # hanya baris pertama yang valid
    assert hasil["part"].iloc[0] == "oli_mesin"  # sudah di-lowercase
    assert hasil["jenis_kendaraan"].iloc[0] == "motor_matic"  # spasi dibuang


def test_agregasi_mingguan_lengkap_dan_nol():
    df = preprocess.bersihkan(data_contoh())
    agg = preprocess.agregasi_mingguan(df)
    # setiap part punya jumlah minggu yang sama (grid lengkap)
    per_part = agg.groupby("part").size()
    assert per_part.nunique() == 1
    # kampas_rem hanya ada 1 kali per minggu -> jumlah mingguan 1
    rem = agg[agg["part"] == "kampas_rem"]["jumlah"]
    assert (rem == 1).all()
    # oli_mesin: 6 hari kerja x 2 = 12 per minggu
    oli = agg[agg["part"] == "oli_mesin"]["jumlah"]
    assert (oli == 12).all()


def test_fitur_tidak_bocor_dan_tanpa_nan():
    df = preprocess.bersihkan(data_contoh())
    agg = preprocess.agregasi_mingguan(df)
    fitur, peta = preprocess.buat_fitur(agg)
    assert not fitur.isna().any().any()
    assert set(peta) == {"oli_mesin", "kampas_rem"}
    # lag_1 harus sama dengan jumlah minggu sebelumnya
    oli = fitur[fitur["part"] == "oli_mesin"].sort_values("minggu")
    assert (oli["lag_1"] == oli["jumlah"].shift(1).fillna(oli["lag_1"])).all()
    # rata4 hanya memakai 4 minggu SEBELUMNYA, bukan minggu ini
    asli = agg[agg["part"] == "oli_mesin"].sort_values("minggu")["jumlah"].reset_index(drop=True)
    minggu_pertama = oli.iloc[0]["minggu"]
    idx = asli.index[
        agg[agg["part"] == "oli_mesin"].sort_values("minggu")["minggu"].reset_index(drop=True)
        == minggu_pertama
    ][0]
    assert oli.iloc[0]["rata4"] == asli.iloc[idx - 4:idx].mean()
