# Ringkasan Hasil Mini Project

## 1. Data
- Jumlah baris dipakai: 150,000 (latih 120,000, uji 30,000)
- Proporsi gagal bayar: 6.68% (data tidak seimbang -> akurasi menyesatkan, dipakai F1/recall/PR-AUC)
- Fitur dengan missing terbanyak: pendapatan_bulanan (19.8%)

## 2. Komparasi model (Stratified 5-fold, split dan metrik sama)
| model | pencarian | kombinasi_diuji | cv_f1 | cv_pr_auc | cv_recall | cv_precision | interpretability_1_5 | skor_pemilihan |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | GridSearchCV | 4 | 0.338 | 0.359 | 0.734 | 0.220 | 4 | 0.491 |
| Decision Tree | GridSearchCV | 12 | 0.366 | 0.328 | 0.306 | 0.454 | 5 | 0.529 |
| Random Forest | RandomizedSearchCV | 8 | 0.421 | 0.383 | 0.404 | 0.440 | 3 | 0.448 |
| Gradient Boosting | RandomizedSearchCV | 8 | 0.334 | 0.378 | 0.243 | 0.534 | 2 | 0.384 |

Parameter terbaik per model ada di `komparasi_model.csv`.

## 3. Pemilihan model
Skor = 0.7 x PR_AUC(CV) + 0.3 x (Interpretability / 5)

| bobot_pr_auc | bobot_interpretasi | Logistic Regression | Decision Tree | Random Forest | Gradient Boosting | terpilih |
|---|---|---|---|---|---|---|
| 0.700 | 0.300 | 0.491 | 0.529 | 0.448 | 0.384 | Decision Tree |
| 0.900 | 0.100 | 0.403 | 0.395 | 0.405 | 0.380 | Random Forest |
| 1.000 | 0.000 | 0.359 | 0.328 | 0.383 | 0.378 | Random Forest |

**Model terpilih: Decision Tree**

## 4. Evaluasi test set (dipakai sekali)
- PR-AUC: 0.329 | ROC-AUC: 0.835
- Ambang 0,5: recall 0.321, precision 0.459, F1 0.378
- Ambang terpilih 0.207: recall 0.669, precision 0.222, F1 0.334

Confusion matrix (ambang 0.207): TP = 1342, FP = 4698, FN = 663, TN = 23297

Perhitungan manual:
- Precision = 1342 / (1342 + 4698) = 0.222
- Recall = 1342 / (1342 + 663) = 0.669
- F1 = 2 x 0.222 x 0.669 / (0.222 + 0.669) = 0.334

## 5. Fitur paling berpengaruh
| fitur | penurunan_pr_auc |
|---|---|
| telat_90_plus | 0.1417 |
| utilisasi_kredit | 0.0612 |
| telat_60_89 | 0.0486 |
| telat_30_59 | 0.0464 |
| usia | 0.0016 |

## 6. Analisis error
Jumlah per kelompok: TN=23297, FP=4698, TP=1342, FN=663.
Median fitur tiap kelompok ada di `analisis_error_median_fitur.csv`. Bandingkan kelompok FN
(gagal bayar yang lolos) dengan TP untuk mencari pola yang terlewat model.

## 7. Grafik
`figures/`: eda_kelas, eda_missing, eda_korelasi, komparasi_model, confusion_matrix, pr_curve, kepentingan_fitur.
