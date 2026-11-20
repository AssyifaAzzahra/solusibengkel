import pandas as pd

from src import predict


def palsu_prediksi(*args, **kwargs):
    return pd.DataFrame({
        "minggu_target": ["2026-01-05"] * 4,
        "part": ["oli_mesin", "busi", "aki", "ban"],
        "prediksi": [50.0, 10.0, 4.0, 6.0],
    })


def test_status_rekomendasi(monkeypatch):
    monkeypatch.setattr(predict, "prediksi_minggu_depan", palsu_prediksi)
    hasil = {
        r["part"]: r
        for r in predict.rekomendasi({"oli_mesin": 10, "busi": 12, "aki": 100})
    }
    # kebutuhan = ceil(prediksi * 1.2)
    assert hasil["oli_mesin"]["kebutuhan_dengan_pengaman"] == 60
    assert hasil["oli_mesin"]["status"] == "RESTOCK"
    assert hasil["oli_mesin"]["jumlah_restock"] == 50
    assert hasil["busi"]["status"] == "AMAN"
    assert hasil["busi"]["jumlah_restock"] == 0
    assert hasil["aki"]["status"] == "STOK_BERLEBIH"
    assert hasil["ban"]["status"] == "STOK_TIDAK_DIKETAHUI"


def test_fitur_minggu_depan():
    meta = {"peta_part": {"oli_mesin": 0}}
    minggu = pd.date_range("2024-01-01", periods=10, freq="7D")
    mingguan = pd.DataFrame({
        "minggu": minggu, "part": "oli_mesin", "jumlah": range(1, 11),
    })
    fitur, target = predict.fitur_minggu_depan(mingguan, meta)
    assert target == minggu[-1] + pd.Timedelta(days=7)
    assert fitur.iloc[0]["lag_1"] == 10
    assert fitur.iloc[0]["lag_4"] == 7
    assert fitur.iloc[0]["rata4"] == (7 + 8 + 9 + 10) / 4
