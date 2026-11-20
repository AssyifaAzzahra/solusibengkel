from src import retrain, simulasi_drift


def test_layak_promosi():
    assert retrain.layak_promosi(5.0, 4.0) is True      # jelas lebih baik
    assert retrain.layak_promosi(5.0, 5.0) is False     # sama saja -> tidak diganti
    assert retrain.layak_promosi(5.0, 4.95) is False    # kurang dari margin 2%
    assert retrain.layak_promosi(5.0, 6.0) is False     # lebih buruk


def test_data_simulasi_punya_kendaraan_baru():
    df = simulasi_drift.buat_data_baru(n_minggu=2)
    assert list(df.columns) == ["tanggal", "jenis_kendaraan", "jenis_servis", "part", "jumlah"]
    assert "mobil_listrik" in set(df["jenis_kendaraan"])
    # kendaraan listrik tidak memakai oli
    listrik = df[df["jenis_kendaraan"] == "mobil_listrik"]
    assert "oli_mesin" not in set(listrik["part"])
    assert (df["jumlah"] > 0).all()
