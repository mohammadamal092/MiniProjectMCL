"""Aplikasi Streamlit: prediksi risiko gagal bayar.

Jalankan:  streamlit run app.py
Membutuhkan artifacts/model_terbaik.pkl dan artifacts/metadata.json hasil train.py.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import common  # diperlukan agar joblib bisa memuat fungsi clean_features di dalam pipeline
from common import FEATURE_SPEC, FEATURES

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Skor Risiko Gagal Bayar", page_icon="🏦", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ART / "model_terbaik.pkl")
    meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
    return model, meta


def validate(values):
    """Validasi tipe dan rentang nilai. Mengembalikan daftar pesan kesalahan."""
    errors = []
    for key, spec in FEATURE_SPEC.items():
        v = values[key]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue  # pendapatan boleh kosong, akan diimputasi oleh pipeline
        if not (spec["min"] <= v <= spec["max"]):
            errors.append(f"{spec['label']} harus antara {spec['min']} dan {spec['max']}.")
    if values["jumlah_kredit_aktif"] is not None and values["jumlah_kpr"] > values["jumlah_kredit_aktif"]:
        errors.append("Jumlah KPR tidak boleh lebih besar dari jumlah kredit aktif.")
    return errors


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model belum ada. Jalankan dulu: `python train.py --data data/cs-training.csv`")
    st.stop()

threshold = meta["ambang_keputusan"]

st.title("🏦 Skor Risiko Gagal Bayar Peminjam")
st.caption(
    "Alat bantu untuk menandai aplikasi yang perlu ditinjau ulang. "
    "Keputusan akhir tetap di tangan petugas kredit."
)

with st.sidebar:
    st.header("Tentang model")
    st.write(f"**Model:** {meta['model']}")
    st.write(f"**Dilatih:** {meta['dilatih_pada'][:10]} pada {meta['jumlah_data']:,} data")
    st.write(f"**Ambang peninjauan:** {threshold:.1%}")
    cv, te = meta["cv"], meta["test"]["ambang_terpilih"]
    st.markdown("**Kinerja (test set)**")
    st.write(f"Recall gagal bayar: {te['recall']:.1%}")
    st.write(f"Precision: {te['precision']:.1%}")
    st.write(f"PR-AUC: {meta['test']['pr_auc']:.3f}")
    fig = ART / "figures" / "kepentingan_fitur.png"
    if fig.exists():
        st.image(str(fig), caption="Fitur paling berpengaruh")

st.subheader("Data peminjam")
values = {}
cols = st.columns(2)
for i, key in enumerate(FEATURES):
    spec = FEATURE_SPEC[key]
    cast = int if spec["kind"] == "int" else float
    with cols[i % 2]:
        if key == "pendapatan_bulanan":
            unknown = st.checkbox("Pendapatan tidak diketahui", value=False)
            val = st.number_input(
                spec["label"], min_value=cast(spec["min"]), max_value=cast(spec["max"]),
                value=cast(spec["default"]), step=cast(spec["step"]), help=spec["help"], disabled=unknown,
            )
            values[key] = np.nan if unknown else float(val)
        else:
            val = st.number_input(
                spec["label"], min_value=cast(spec["min"]), max_value=cast(spec["max"]),
                value=cast(spec["default"]), step=cast(spec["step"]), help=spec["help"],
            )
            values[key] = val

if st.button("Prediksi", type="primary"):
    errors = validate(values)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # Kolom dan urutan sama persis dengan data training
    x = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    prob = float(model.predict_proba(x)[0, 1])

    if prob >= threshold:
        level, color, advice = "TINGGI", "🔴", "Tinjau ulang secara manual sebelum persetujuan."
    elif prob >= threshold / 2:
        level, color, advice = "SEDANG", "🟡", "Periksa riwayat keterlambatan dan rasio utang sebelum memutuskan."
    else:
        level, color, advice = "RENDAH", "🟢", "Dapat diproses melalui alur standar."

    c1, c2 = st.columns(2)
    c1.metric("Skor risiko gagal bayar", f"{prob:.1%}")
    c2.metric("Kategori", f"{color} {level}")
    st.progress(min(prob, 1.0))
    st.info(f"**Rekomendasi:** {advice}")
    st.caption(f"Kategori TINGGI jika skor ≥ {threshold:.1%} (ambang dipilih agar sebagian besar gagal bayar terdeteksi).")

    with st.expander("Data yang dikirim ke model"):
        st.dataframe(x, hide_index=True)

with st.expander("Batas penggunaan model (baca sebelum dipakai)"):
    st.markdown(
        """
- Skor adalah **peringkat risiko**, bukan probabilitas gagal bayar yang sebenarnya. Model dilatih dengan
  oversampling (SMOTE), sehingga angka cenderung lebih tinggi dari kejadian nyata.
- Dilatih pada data historis satu dataset publik. Distribusi nasabah yang berbeda dapat menurunkan akurasi (data drift).
- Model hanya memakai 10 fitur dan **tidak boleh menjadi satu-satunya dasar** penolakan pinjaman.
- Belum diaudit per kelompok demografis. Lakukan audit bias sebelum dipakai untuk keputusan nyata.
- Evaluasi ulang secara berkala dengan label aktual terbaru.
"""
    )
