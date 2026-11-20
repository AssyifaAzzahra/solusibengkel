"""
API sistem prediksi kebutuhan sparepart.

Jalankan dari root proyek:
    uvicorn api.main:app --reload
Dokumentasi interaktif: http://localhost:8000/docs
"""
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import predict

app = FastAPI(title="Prediksi Restock Sparepart Bengkel", version="1.0")


class PermintaanRekomendasi(BaseModel):
    # contoh: {"oli_mesin": 40, "kampas_rem": 5}. Part yang tidak diisi dianggap tidak diketahui.
    stok: Optional[Dict[str, int]] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        return predict.muat_meta()
    except FileNotFoundError:
        raise HTTPException(503, "Model belum ada. Jalankan python src/train.py")


@app.get("/predict")
def prediksi():
    try:
        return predict.prediksi_minggu_depan().to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(503, f"File belum ada: {e.filename}")


@app.post("/recommend")
def rekomendasi(req: PermintaanRekomendasi):
    try:
        return predict.rekomendasi(req.stok)
    except FileNotFoundError as e:
        raise HTTPException(503, f"File belum ada: {e.filename}")


@app.get("/riwayat/{part}")
def riwayat(part: str, n: int = 26):
    df = predict.muat_mingguan()
    df = df[df["part"] == part].sort_values("minggu").tail(n)
    if df.empty:
        raise HTTPException(404, f"Part '{part}' tidak ditemukan")
    return [
        {"minggu": r["minggu"].strftime("%Y-%m-%d"), "jumlah": int(r["jumlah"])}
        for _, r in df.iterrows()
    ]


@app.get("/monitoring")
def monitoring():
    import json

    import pandas as pd

    folder = predict.ROOT / "data" / "monitoring"
    f_laporan = folder / "laporan.json"
    if not f_laporan.exists():
        raise HTTPException(404, "Laporan belum ada. Jalankan python src/monitor.py --backfill 12")
    laporan = json.loads(f_laporan.read_text())
    f_minggu = folder / "per_minggu.csv"
    laporan["per_minggu"] = (
        pd.read_csv(f_minggu).to_dict(orient="records") if f_minggu.exists() else []
    )
    return laporan
