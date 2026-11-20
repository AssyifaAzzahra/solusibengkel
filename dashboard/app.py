"""
Dashboard sederhana untuk pemilik bengkel (Streamlit).

Jalankan dari root proyek (API harus sudah jalan di port 8000):
    streamlit run dashboard/app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
WARNA = {
    "RESTOCK": "background-color: #ffd6d6",
    "STOK_BERLEBIH": "background-color: #fff3c4",
    "AMAN": "background-color: #d9f2d9",
    "STOK_TIDAK_DIKETAHUI": "background-color: #e8e8e8",
}

st.set_page_config(page_title="Restock Sparepart", layout="wide")
st.title("Rekomendasi Restock Sparepart Bengkel")


def api_get(path):
    r = requests.get(f"{API_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


try:
    info = api_get("/model-info")
except Exception as e:
    st.error(f"Tidak bisa terhubung ke API di {API_URL}. Pastikan API sudah jalan.\n\n{e}")
    st.stop()

with st.sidebar:
    st.subheader("Info model")
    st.write(f"Model: **{info['nama_model']}**")
    st.write(f"Dilatih: {info['dilatih_pada']}")
    st.write(f"MAE uji: {info['metrik_uji']['mae']:.2f}")
    st.write(f"MAE baseline: {info['metrik_baseline']['mae']:.2f}")

st.subheader("1. Isi stok saat ini")
st.caption("Ubah angka stok sesuai kondisi gudang, lalu klik tombol di bawah.")

daftar_part = sorted(info["peta_part"].keys())
stok_awal = pd.DataFrame({"part": daftar_part, "stok": [20] * len(daftar_part)})
stok_edit = st.data_editor(stok_awal, hide_index=True, disabled=["part"])

if st.button("Hitung rekomendasi", type="primary"):
    stok = {r["part"]: int(r["stok"]) for _, r in stok_edit.iterrows()}
    r = requests.post(f"{API_URL}/recommend", json={"stok": stok}, timeout=20)
    r.raise_for_status()
    st.session_state["hasil"] = pd.DataFrame(r.json())

if "hasil" in st.session_state:
    df = st.session_state["hasil"]
    st.subheader(f"2. Rekomendasi untuk minggu {df['minggu_target'].iloc[0]}")

    perlu = df[df["status"] == "RESTOCK"]
    if perlu.empty:
        st.success("Tidak ada part yang perlu di-restock minggu ini.")
    else:
        st.warning(
            "Perlu restock: "
            + ", ".join(f"{r.part} (+{r.jumlah_restock})" for r in perlu.itertuples())
        )
    berlebih = df[df["status"] == "STOK_BERLEBIH"]
    if not berlebih.empty:
        st.info("Stok berlebih (modal tertahan): " + ", ".join(berlebih["part"]))

    tampil = df[["part", "prediksi", "kebutuhan_dengan_pengaman",
                 "stok_saat_ini", "status", "jumlah_restock"]]
    st.dataframe(
        tampil.style.apply(lambda row: [WARNA.get(row["status"], "")] * len(row), axis=1),
        hide_index=True,
    )

    st.subheader("3. Riwayat pemakaian")
    pilih = st.selectbox("Pilih part", daftar_part)
    riwayat = pd.DataFrame(api_get(f"/riwayat/{pilih}?n=26"))
    riwayat["minggu"] = pd.to_datetime(riwayat["minggu"])
    st.line_chart(riwayat.set_index("minggu")["jumlah"])
