"""Halaman monitoring: performa model dan data drift."""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Monitoring Model", layout="wide")
st.title("Monitoring Model")

r = requests.get(f"{API_URL}/monitoring", timeout=10)
if r.status_code == 404:
    st.warning("Laporan monitoring belum ada. Jalankan: `python src/monitor.py --backfill 12`")
    st.stop()
r.raise_for_status()
lap = r.json()

st.caption(f"Laporan dibuat: {lap['dibuat']} | jendela pantau: {lap['window_minggu']} minggu terakhir")

if lap["status"] == "OK":
    st.success("Status: OK. Model masih layak dipakai.")
else:
    st.error("Status: RETRAINING DISARANKAN")
    for a in lap["alasan"]:
        st.write(f"- {a}")

st.subheader("Performa: prediksi vs aktual")
p, u = lap["performa"], lap["performa_saat_uji"]
if p:
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE terbaru", f"{p['mae']:.2f}", f"{p['mae'] - u['mae']:+.2f} vs saat uji", delta_color="inverse")
    c2.metric("WAPE terbaru", f"{p['wape']:.1f}%", f"{p['wape'] - u['wape']:+.1f} poin vs saat uji", delta_color="inverse")
    c3.metric("Jumlah prediksi dievaluasi", p["n_prediksi"])
    if lap["per_minggu"]:
        pm = pd.DataFrame(lap["per_minggu"])
        pm["minggu_target"] = pd.to_datetime(pm["minggu_target"])
        st.line_chart(pm.set_index("minggu_target")[["wape"]])
else:
    st.info("Belum ada prediksi yang punya data aktual.")

st.subheader("Drift pola pemakaian per part (uji KS)")
dp = pd.DataFrame(lap["drift_part"])
if not dp.empty:
    st.dataframe(dp, hide_index=True)

st.subheader("Porsi jenis kendaraan (sebelum vs terbaru)")
dk = pd.DataFrame(lap["drift_kendaraan"])
if not dk.empty:
    st.dataframe(dk, hide_index=True)
    st.bar_chart(dk.set_index("jenis_kendaraan")[["porsi_ref", "porsi_terbaru"]])
