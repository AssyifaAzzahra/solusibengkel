import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src import predict

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_recommend_dengan_stok(monkeypatch):
    monkeypatch.setattr(
        predict, "prediksi_minggu_depan",
        lambda *a, **k: pd.DataFrame({
            "minggu_target": ["2026-01-05"], "part": ["oli_mesin"], "prediksi": [20.0],
        }),
    )
    r = client.post("/recommend", json={"stok": {"oli_mesin": 5}})
    assert r.status_code == 200
    data = r.json()
    assert data[0]["status"] == "RESTOCK"
    assert data[0]["jumlah_restock"] == 19  # ceil(20*1.2)=24, stok 5


def test_riwayat_part_tidak_ada(monkeypatch):
    monkeypatch.setattr(
        predict, "muat_mingguan",
        lambda: pd.DataFrame({
            "minggu": pd.to_datetime(["2024-01-01"]), "part": ["aki"], "jumlah": [3],
        }),
    )
    assert client.get("/riwayat/tidak_ada").status_code == 404





def test_monitoring_laporan_belum_ada(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "ROOT", tmp_path)
    assert client.get("/monitoring").status_code == 404
