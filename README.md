# Mini Project ML End-to-End: Skor Risiko Gagal Bayar

Komparasi 4 model, tuning dengan Stratified 5-fold CV, pipeline tersimpan, dan aplikasi Streamlit.
Dataset publik: **Give Me Some Credit** (Kaggle), 150.000 nasabah, sekitar 6,7% gagal bayar.

## Isi folder

| File | Fungsi |
|---|---|
| `common.py` | Skema data, aturan pembersihan fitur, dan pipeline (dipakai training dan aplikasi) |
| `train.py` | EDA, split, komparasi + tuning model, evaluasi test, simpan pipeline, buat ringkasan laporan |
| `app.py` | Aplikasi Streamlit yang memuat pipeline tersimpan |
| `generate_demo_data.py` | Data **sintetis** hanya untuk uji coba alur. Jangan dipakai untuk laporan |
| `requirements.txt` | Daftar library |

## 1. Persiapan (Windows)

Butuh Python 3.10 sampai 3.12. Laragon dan PostgreSQL **tidak diperlukan** untuk project ini
(Streamlit punya server sendiri dan datanya berupa file CSV).

```bash
cd ml-risiko-pinjaman
python -m venv .venv
.venv\Scripts\activate  
pip install -r requirements.txt
```

## 2. Siapkan dataset

1. Buka https://www.kaggle.com/competitions/GiveMeSomeCredit/data (perlu akun Kaggle dan klik "Join Competition").
2. Unduh `cs-training.csv`, simpan sebagai `data/cs-training.csv`.

Ingin uji dulu sebelum mengunduh? `python generate_demo_data.py` membuat `data/demo_cs-training.csv`.

## 3. Training

```bash
# Uji cepat (beberapa menit)
python train.py --data data/cs-training.csv --sample 30000

# Data penuh untuk hasil akhir laporan (bisa 20 sampai 60 menit)
python train.py --data data/cs-training.csv
```

Opsi: `--n-iter 8` (percobaan RandomizedSearchCV), `--select-metric pr_auc|f1`, `--w-metric 0.7`, `--seed 42`.

Hasil ada di `artifacts/`: `model_terbaik.pkl`, `metadata.json`, `komparasi_model.csv`,
`sensitivitas_bobot.csv`, `hasil_ringkas.md` (siap ditempel ke laporan), dan folder `figures/`.

## 4. Aplikasi

```bash
streamlit run app.py
```

Buka http://localhost:8501. Isi data peminjam lalu klik **Prediksi**.

## 5. Keputusan desain (siap dijelaskan saat presentasi)

- **Satu pipeline**: pembersihan -> imputasi median -> (skala, khusus Logistic Regression) -> SMOTE -> model.
  Dalam CV, SMOTE dan imputer hanya dilatih pada data latih tiap fold, jadi tidak ada data leakage.
- **Komparasi adil**: split train/test sama, StratifiedKFold 5 sama, metrik sama. Model dengan ruang parameter kecil
  (Logistic Regression, Decision Tree) memakai `GridSearchCV`; Random Forest dan Gradient Boosting memakai
  `RandomizedSearchCV` dengan jumlah percobaan sama (`--n-iter`).
- **Metrik**: akurasi menyesatkan karena kelas tidak seimbang. Dipakai recall, precision, F1, dan PR-AUC
  sesuai tujuan bisnis (false negative lebih mahal).
- **Pemilihan model**: skor = bobot x PR-AUC(CV) + (1 - bobot) x Interpretability/5. PR-AUC dipilih karena tidak
  bergantung ambang; F1 pada ambang 0,5 merugikan model berbasis pohon yang dilatih dengan SMOTE. Untuk mengikuti
  rumus F1 di slide dosen, jalankan dengan `--select-metric f1`.
- **Skor interpretability** (Logistic Regression 4, Decision Tree 5, Random Forest 3, Gradient Boosting 2) adalah
  asumsi. Ubah di `model_configs()` pada `train.py` dan sepakati bobotnya sebelum melihat hasil (slide 14).
- **Test set dipakai sekali**, hanya untuk model terpilih. Tabel `sensitivitas_bobot.csv` menunjukkan model mana
  yang terpilih jika bobot berubah.
- **Ambang keputusan** dipilih dari prediksi out-of-fold data latih (memaksimalkan F2, recall diutamakan).
- **Skor bukan probabilitas asli** karena SMOTE mengubah proporsi kelas. Aplikasi menyebut ini di bagian batas penggunaan.

## 6. Pemetaan ke rubrik

| Komponen | Bobot | Bukti di project ini |
|---|---|---|
| Business dan data | 20% | Bagian Business Understanding di laporan, `eda_*.png`, `eda_statistik.csv`, `eda_missing.csv` |
| Model dan tuning | 30% | `komparasi_model.csv` (4 model, CV, ruang pencarian di `model_configs()`) |
| Evaluasi dan interpretasi | 25% | `hasil_ringkas.md`, `confusion_matrix.png`, `pr_curve.png`, `kepentingan_fitur.png`, `analisis_error_median_fitur.csv` |
| Deployment | 15% | `app.py` berjalan, input divalidasi, kolom sama dengan training |
| Laporan dan presentasi | 10% | Struktur di bawah |

## 7. Struktur laporan

1. **Pendahuluan**: masalah gagal bayar, tujuan, biaya false negative, metrik utama.
2. **Tinjauan data**: sumber dataset, 10 fitur, proporsi kelas, missing value, nilai tidak wajar (usia 0, kode 96/98).
3. **Metodologi**: pipeline, SMOTE hanya pada train, CV, ruang pencarian, rumus pemilihan model.
4. **Hasil**: tabel komparasi, sensitivitas bobot, hasil test, confusion matrix dan hitung manual, error analysis.
5. **Deployment**: screenshot aplikasi, cara kerja input -> validasi -> pipeline -> hasil, batas penggunaan.
6. **Kesimpulan**: temuan, keterbatasan (drift, bias, skor tidak terkalibrasi), saran pengembangan.

Angka untuk bagian 4 ada di `artifacts/hasil_ringkas.md` setelah training selesai.

## 8. Checklist presentasi 10-15 menit

1. Masalah dan metrik (1 menit)
2. Data dan EDA (2 menit)
3. Pipeline dan anti-leakage (2 menit)
4. Komparasi dan tuning (3 menit)
5. Alasan pemilihan model dan sensitivitas bobot (2 menit)
6. Demo aplikasi dengan satu profil berisiko tinggi dan satu rendah (3 menit)
7. Batas penggunaan dan risiko deployment (1 menit)

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Kolom ... tidak ditemukan` | File bukan `cs-training.csv` dari Give Me Some Credit |
| Training terlalu lama | Pakai `--sample 30000` atau kurangi `--n-iter` |
| `Model belum ada` di aplikasi | Jalankan `train.py` dulu |
| `No module named common` | Jalankan `streamlit run app.py` dari dalam folder project |
| Error saat memuat model | Versi scikit-learn saat training dan menjalankan aplikasi harus sama |
