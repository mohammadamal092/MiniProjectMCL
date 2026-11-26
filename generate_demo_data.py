"""Membuat DATA SINTETIS berskema sama dengan cs-training.csv (Give Me Some Credit).

HANYA untuk menguji bahwa train.py dan app.py berjalan di komputer Anda.
Jangan dipakai untuk laporan: angka hasilnya tidak mewakili data nyata.

    python generate_demo_data.py        -> data/demo_cs-training.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 20000

age = rng.integers(21, 85, n)
income = rng.lognormal(8.6, 0.6, n).round()
util = np.clip(rng.beta(0.6, 1.5, n) * 1.2, 0, None)
debt = rng.lognormal(-1.0, 0.9, n).round(3)
late3059 = rng.poisson(0.25, n)
late6089 = rng.poisson(0.12, n)
late90 = rng.poisson(0.10, n)
open_lines = rng.poisson(8, n)
real_estate = rng.poisson(1.0, n)
dependents = rng.poisson(0.8, n).astype(float)

z = (1.6 * util + 0.55 * late3059 + 0.7 * late6089 + 0.9 * late90
     - 0.03 * (age - 45) + 0.2 * np.log1p(debt) - 0.15 * np.log(income / 5000))


def mean_prob(b):
    return (1 / (1 + np.exp(-(z + b)))).mean()


lo, hi = -15.0, 5.0
for _ in range(60):  # cari intersep agar rata-rata gagal bayar sekitar 7%
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if mean_prob(mid) < 0.07 else (lo, mid)
p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
y = (rng.random(n) < p).astype(int)

income = income.astype(float)
income[rng.random(n) < 0.20] = np.nan
dependents[rng.random(n) < 0.026] = np.nan
age = age.astype(float)
age[rng.random(n) < 0.0005] = 0
for arr in (late3059, late6089, late90):
    arr[rng.random(n) < 0.002] = rng.choice([96, 98])

df = pd.DataFrame({
    "SeriousDlqin2yrs": y,
    "RevolvingUtilizationOfUnsecuredLines": util.round(4),
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": late3059,
    "DebtRatio": debt,
    "MonthlyIncome": income,
    "NumberOfOpenCreditLinesAndLoans": open_lines,
    "NumberOfTimes90DaysLate": late90,
    "NumberRealEstateLoansOrLines": real_estate,
    "NumberOfTime60-89DaysPastDueNotWorse": late6089,
    "NumberOfDependents": dependents,
})
df.index = df.index + 1
Path("data").mkdir(exist_ok=True)
df.to_csv("data/demo_cs-training.csv", index_label="")
print(f"Dibuat data/demo_cs-training.csv ({len(df):,} baris, gagal bayar {y.mean():.1%})")
