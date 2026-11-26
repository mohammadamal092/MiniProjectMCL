"""Kode bersama untuk train.py dan app.py.

Semua transformasi data (pembersihan, imputasi, skala, SMOTE) ada di dalam satu pipeline,
sehingga data dari form Streamlit diproses persis sama dengan data training.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TARGET = "gagal_bayar"

# Nama kolom asli dataset Give Me Some Credit -> nama kolom yang dipakai di proyek ini.
RAW_TO_NEW = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "utilisasi_kredit",
    "age": "usia",
    "NumberOfTime30-59DaysPastDueNotWorse": "telat_30_59",
    "DebtRatio": "rasio_utang",
    "MonthlyIncome": "pendapatan_bulanan",
    "NumberOfOpenCreditLinesAndLoans": "jumlah_kredit_aktif",
    "NumberOfTimes90DaysLate": "telat_90_plus",
    "NumberRealEstateLoansOrLines": "jumlah_kpr",
    "NumberOfTime60-89DaysPastDueNotWorse": "telat_60_89",
    "NumberOfDependents": "jumlah_tanggungan",
}

# Urutan ini adalah urutan kolom yang dipakai saat training DAN saat prediksi di aplikasi.
FEATURES = [
    "utilisasi_kredit",
    "usia",
    "telat_30_59",
    "rasio_utang",
    "pendapatan_bulanan",
    "jumlah_kredit_aktif",
    "telat_90_plus",
    "jumlah_kpr",
    "telat_60_89",
    "jumlah_tanggungan",
]

# Spesifikasi form aplikasi (label, tipe, rentang valid, nilai awal, penjelasan).
FEATURE_SPEC = {
    "utilisasi_kredit": dict(
        label="Utilisasi kredit tanpa jaminan", kind="float", min=0.0, max=5.0, default=0.30, step=0.05,
        help="Saldo kartu kredit dan kredit tanpa agunan dibagi total limit. 0,30 berarti 30%.",
    ),
    "usia": dict(
        label="Usia (tahun)", kind="int", min=18, max=100, default=40, step=1,
        help="Usia peminjam saat mengajukan.",
    ),
    "telat_30_59": dict(
        label="Jumlah keterlambatan 30-59 hari", kind="int", min=0, max=20, default=0, step=1,
        help="Berapa kali terlambat bayar 30-59 hari dalam 2 tahun terakhir.",
    ),
    "rasio_utang": dict(
        label="Rasio utang (cicilan / pendapatan)", kind="float", min=0.0, max=10.0, default=0.30, step=0.05,
        help="Total cicilan dan biaya hidup bulanan dibagi pendapatan bulanan.",
    ),
    "pendapatan_bulanan": dict(
        label="Pendapatan bulanan", kind="float", min=0.0, max=1_000_000.0, default=5000.0, step=100.0,
        help="Satuan mata uang mengikuti dataset. Kosongkan lewat kotak centang jika tidak diketahui.",
    ),
    "jumlah_kredit_aktif": dict(
        label="Jumlah kredit/pinjaman aktif", kind="int", min=0, max=60, default=6, step=1,
        help="Termasuk kartu kredit, cicilan kendaraan, dan pinjaman lain yang masih berjalan.",
    ),
    "telat_90_plus": dict(
        label="Jumlah keterlambatan 90 hari atau lebih", kind="int", min=0, max=20, default=0, step=1,
        help="Berapa kali terlambat bayar 90 hari atau lebih.",
    ),
    "jumlah_kpr": dict(
        label="Jumlah KPR / kredit properti", kind="int", min=0, max=20, default=1, step=1,
        help="Kredit rumah dan pinjaman berjaminan properti.",
    ),
    "telat_60_89": dict(
        label="Jumlah keterlambatan 60-89 hari", kind="int", min=0, max=20, default=0, step=1,
        help="Berapa kali terlambat bayar 60-89 hari dalam 2 tahun terakhir.",
    ),
    "jumlah_tanggungan": dict(
        label="Jumlah tanggungan", kind="int", min=0, max=15, default=0, step=1,
        help="Jumlah anggota keluarga yang ditanggung.",
    ),
}

LATE_COLS = ["telat_30_59", "telat_60_89", "telat_90_plus"]


def load_data(path):
    """Baca cs-training.csv, ubah nama kolom, dan pastikan skema sesuai."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    df = df.rename(columns=RAW_TO_NEW)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di {path}: {missing}. "
            "Pastikan file adalah cs-training.csv dari dataset Give Me Some Credit."
        )
    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def clean_features(X):
    """Aturan pembersihan yang berlaku sama di training dan di aplikasi.

    - usia di luar 18-110 dianggap tidak valid -> NaN (nanti diimputasi)
    - nilai >= 90 pada kolom keterlambatan adalah kode khusus (96/98) di dataset -> NaN
    - nilai ekstrem pada rasio dipotong agar tidak mendominasi model
    """
    X = pd.DataFrame(X).astype(float).copy()
    X.loc[(X["usia"] < 18) | (X["usia"] > 110), "usia"] = np.nan
    for col in LATE_COLS:
        X.loc[X[col] >= 90, col] = np.nan
    X["utilisasi_kredit"] = X["utilisasi_kredit"].clip(0, 5)
    X["rasio_utang"] = X["rasio_utang"].clip(0, 10)
    X["pendapatan_bulanan"] = X["pendapatan_bulanan"].clip(0, 1_000_000)
    return X


def build_pipeline(estimator, scale=False, seed=42):
    """Pipeline lengkap. SMOTE hanya aktif saat fit (hanya pada data latih tiap fold)."""
    steps = [
        ("clean", FunctionTransformer(clean_features)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps += [("smote", SMOTE(random_state=seed)), ("model", estimator)]
    return ImbPipeline(steps)
