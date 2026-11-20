import numpy as np
import pandas as pd

from src import monitor


def mingguan_contoh(n=40, geser=0.0):
    minggu = pd.date_range("2024-01-01", periods=n, freq="7D")
    rng = np.random.default_rng(0)
    jumlah = rng.poisson(20, size=n)
    jumlah[-12:] = rng.poisson(20 + geser, size=12)
    return pd.DataFrame({"minggu": minggu, "part": "oli_mesin", "jumlah": jumlah})


def test_evaluasi_hitung_error():
    mingguan = pd.DataFrame({
        "minggu": pd.to_datetime(["2024-01-01", "2024-01-08"]),
        "part": ["aki", "aki"],
        "jumlah": [10, 20],
    })
    log = pd.DataFrame({
        "minggu_target": ["2024-01-01", "2024-01-08"],
        "part": ["aki", "aki"],
        "prediksi": [8.0, 25.0],
    })
    ringkas, per_minggu = monitor.evaluasi(log, mingguan)
    assert ringkas["n_prediksi"] == 2
    assert ringkas["mae"] == 3.5          # (2 + 5) / 2
    assert round(ringkas["wape"], 2) == round(7 / 30 * 100, 2)
    assert len(per_minggu) == 2


def test_evaluasi_tanpa_data_aktual():
    mingguan = mingguan_contoh()
    log = pd.DataFrame({"minggu_target": ["2030-01-01"], "part": ["oli_mesin"], "prediksi": [1.0]})
    ringkas, per_minggu = monitor.evaluasi(log, mingguan)
    assert ringkas is None and per_minggu.empty


def test_drift_terdeteksi_kalau_pola_bergeser():
    stabil = monitor.drift_jumlah(mingguan_contoh(geser=0.0), window=12)
    bergeser = monitor.drift_jumlah(mingguan_contoh(geser=25.0), window=12)
    assert stabil[0]["drift"] is False
    assert bergeser[0]["drift"] is True


def test_putuskan_status():
    meta = {"metrik_uji": {"wape": 20.0}}
    status, alasan = monitor.putuskan({"wape": 21.0}, meta, [], [])
    assert status == "OK" and alasan == []

    status, alasan = monitor.putuskan({"wape": 40.0}, meta, [], [])
    assert status == "RETRAIN_DISARANKAN"

    drift = [{"jenis_kendaraan": "ev", "jenis_baru": True}]
    status, alasan = monitor.putuskan({"wape": 21.0}, meta, [], drift)
    assert status == "RETRAIN_DISARANKAN" and "ev" in alasan[0]
