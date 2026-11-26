"""Mini Project ML End-to-End: prediksi risiko gagal bayar (Give Me Some Credit).

Alur (CRISP-DM): EDA -> split -> komparasi 4 model dengan CV yang sama -> tuning ->
pilih model dengan alasan terukur -> evaluasi test set SEKALI -> simpan pipeline.

Contoh:
    python train.py --data data/cs-training.csv --sample 30000     # uji cepat
    python train.py --data data/cs-training.csv                    # data penuh
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

from common import FEATURES, TARGET, build_pipeline, load_data

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- util
def to_md(df, digits=3):
    """DataFrame -> tabel Markdown (tanpa library tambahan)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def py(obj):
    """Ubah tipe numpy agar bisa disimpan ke JSON."""
    if isinstance(obj, dict):
        return {k: py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# --------------------------------------------------------------------------- EDA
def run_eda(df, out):
    fig_dir = out / "figures"
    df.describe().T.to_csv(out / "eda_statistik.csv")

    miss = (df[FEATURES].isna().mean() * 100).sort_values(ascending=False)
    miss.rename("persen_missing").to_csv(out / "eda_missing.csv")

    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["Lancar (0)", "Gagal bayar (1)"], counts.values, color=["#2e7d32", "#c62828"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Distribusi kelas target")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_kelas.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    miss.plot.barh(ax=ax, color="#1565c0")
    ax.set_xlabel("% data kosong")
    ax.set_title("Missing value per fitur")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_missing.png", dpi=150)
    plt.close(fig)

    corr = df[FEATURES + [TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Korelasi Spearman")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_korelasi.png", dpi=150)
    plt.close(fig)
    return counts, miss


# --------------------------------------------------------------------------- model
def model_configs(seed, n_iter):
    """Empat model. Model dengan ruang parameter kecil memakai GridSearchCV, ruang lebih besar RandomizedSearchCV."""
    return {
        "Logistic Regression": dict(
            est=LogisticRegression(max_iter=1000), scale=True, interp=4, search="grid",
            params={"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": dict(
            est=DecisionTreeClassifier(random_state=seed), scale=False, interp=5, search="grid",
            params={"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [20, 50, 100]},
        ),
        "Random Forest": dict(
            est=RandomForestClassifier(random_state=seed, n_jobs=1), scale=False, interp=3, search="random",
            params={
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_leaf": [1, 5, 20],
            },
        ),
        "Gradient Boosting": dict(
            est=GradientBoostingClassifier(random_state=seed), scale=False, interp=2, search="random",
            params={
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cs-training.csv")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--sample", type=int, default=0, help="ambil N baris (stratified) agar cepat; 0 = semua")
    ap.add_argument("--n-iter", type=int, default=8, help="jumlah percobaan RandomizedSearchCV")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--w-metric", type=float, default=0.7, help="bobot metrik kinerja pada skor pemilihan model")
    ap.add_argument("--select-metric", choices=["pr_auc", "f1"], default="pr_auc",
                    help="metrik kinerja untuk skor pemilihan. pr_auc tidak bergantung ambang sehingga adil "
                         "untuk model berprobabilitas berbeda; f1 mengikuti rumus di slide dosen")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # 1. Data + EDA ---------------------------------------------------------------
    df = load_data(args.data)
    print(f"Data dibaca: {len(df):,} baris")
    if args.sample and args.sample < len(df):
        df, _ = train_test_split(df, train_size=args.sample, stratify=df[TARGET], random_state=args.seed)
        print(f"Memakai sampel stratified: {len(df):,} baris")
    counts, miss = run_eda(df, out)
    pos_rate = counts.get(1, 0) / counts.sum()
    print(f"Proporsi gagal bayar: {pos_rate:.2%}")

    # 2. Split SATU KALI, dipakai semua model ------------------------------------
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # 3. Komparasi + tuning dengan CV yang sama ---------------------------------
    scoring = {"f1": "f1", "pr_auc": "average_precision", "recall": "recall", "precision": "precision"}
    cfgs = model_configs(args.seed, args.n_iter)
    searches, rows = {}, []
    for name, cfg in cfgs.items():
        pipe = build_pipeline(cfg["est"], scale=cfg["scale"], seed=args.seed)
        common_kw = dict(scoring=scoring, refit="pr_auc", cv=skf, n_jobs=-1)
        space = len(ParameterGrid(cfg["params"]))
        if cfg["search"] == "grid":
            search = GridSearchCV(pipe, cfg["params"], **common_kw)
            n_try = space
        else:
            n_try = min(args.n_iter, space)
            search = RandomizedSearchCV(pipe, cfg["params"], n_iter=n_try, random_state=args.seed, **common_kw)
        t0 = time.time()
        search.fit(X_train, y_train)
        secs = time.time() - t0
        i = search.best_index_
        r = search.cv_results_
        rows.append({
            "model": name,
            "pencarian": "GridSearchCV" if cfg["search"] == "grid" else "RandomizedSearchCV",
            "kombinasi_diuji": n_try,
            "cv_f1": r["mean_test_f1"][i], "cv_f1_std": r["std_test_f1"][i],
            "cv_pr_auc": r["mean_test_pr_auc"][i],
            "cv_recall": r["mean_test_recall"][i],
            "cv_precision": r["mean_test_precision"][i],
            "interpretability_1_5": cfg["interp"],
            "waktu_detik": round(secs, 1),
            "parameter_terbaik": json.dumps(py(search.best_params_)),
        })
        searches[name] = search
        print(f"[{name}] CV F1={r['mean_test_f1'][i]:.3f} PR-AUC={r['mean_test_pr_auc'][i]:.3f} "
              f"recall={r['mean_test_recall'][i]:.3f} ({secs:.0f} dtk, {n_try} kombinasi)")

    comp = pd.DataFrame(rows)
    mcol = f"cv_{args.select_metric}"
    comp["skor_pemilihan"] = args.w_metric * comp[mcol] + (1 - args.w_metric) * comp["interpretability_1_5"] / 5
    comp.to_csv(out / "komparasi_model.csv", index=False)

    # Sensitivitas bobot (keputusan harus disepakati SEBELUM melihat hasil, lihat README)
    sens_rows = []
    for w in sorted({args.w_metric, 0.9, 1.0}):
        sc = w * comp[mcol] + (1 - w) * comp["interpretability_1_5"] / 5
        row = {f"bobot_{args.select_metric}": w, "bobot_interpretasi": round(1 - w, 2)}
        row.update({m: s for m, s in zip(comp["model"], sc)})
        row["terpilih"] = comp.loc[sc.idxmax(), "model"]
        sens_rows.append(row)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out / "sensitivitas_bobot.csv", index=False)

    # Grafik komparasi
    fig, ax = plt.subplots(figsize=(7, 3.8))
    xpos = np.arange(len(comp))
    ax.bar(xpos - 0.2, comp["cv_f1"], 0.4, label="F1 (CV)", color="#1565c0")
    ax.bar(xpos + 0.2, comp["cv_pr_auc"], 0.4, label="PR-AUC (CV)", color="#ef6c00")
    ax.set_xticks(xpos, comp["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Komparasi model (Stratified 5-fold, data latih)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "komparasi_model.png", dpi=150)
    plt.close(fig)

    # 4. Pilih model (berdasarkan CV, BUKAN test set) ----------------------------
    best_name = comp.loc[comp["skor_pemilihan"].idxmax(), "model"]
    best = searches[best_name].best_estimator_  # sudah di-refit pada seluruh data latih
    print(f"\nModel terpilih: {best_name} (skor pemilihan {comp['skor_pemilihan'].max():.3f})")

    # Ambang keputusan dipilih dari prediksi out-of-fold data latih (recall diutamakan: F2)
    oof = cross_val_predict(clone(best), X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    p, r_, thr = precision_recall_curve(y_train, oof)
    f2 = 5 * p[:-1] * r_[:-1] / (4 * p[:-1] + r_[:-1] + 1e-12)
    threshold = float(thr[int(np.argmax(f2))])
    print(f"Ambang keputusan (maks. F2 pada OOF): {threshold:.3f}")

    # 5. Evaluasi test set: SEKALI, hanya model terpilih -------------------------
    proba = best.predict_proba(X_test)[:, 1]
    m_default = metrics_at(y_test, proba, 0.5)
    m_tuned = metrics_at(y_test, proba, threshold)
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    print(f"TEST  PR-AUC={pr_auc:.3f} ROC-AUC={roc_auc:.3f}")
    print(f"TEST  ambang 0,5     : recall={m_default['recall']:.3f} precision={m_default['precision']:.3f} F1={m_default['f1']:.3f}")
    print(f"TEST  ambang terpilih: recall={m_tuned['recall']:.3f} precision={m_tuned['precision']:.3f} F1={m_tuned['f1']:.3f}")

    # Grafik evaluasi
    pred = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Lancar", "Gagal bayar"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix (ambang {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out / "figures" / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
    ax.axhline(y_test.mean(), ls="--", color="gray", label="Baseline acak")
    ax.legend()
    ax.set_title("Precision-Recall curve (test)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "pr_curve.png", dpi=150)
    plt.close(fig)

    # Kepentingan fitur (permutation importance, penjelasan setelah keputusan model final)
    idx = np.random.RandomState(args.seed).permutation(len(X_test))[:5000]
    pi = permutation_importance(
        best, X_test.iloc[idx], y_test.iloc[idx], scoring="average_precision",
        n_repeats=5, random_state=args.seed, n_jobs=-1,
    )
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    imp.sort_values(ascending=False).rename("penurunan_pr_auc").to_csv(out / "kepentingan_fitur.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax, color="#1565c0")
    ax.set_xlabel("Penurunan PR-AUC saat fitur diacak")
    ax.set_title("Kepentingan fitur (permutation importance)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "kepentingan_fitur.png", dpi=150)
    plt.close(fig)

    # Analisis error: rata-rata fitur per kelompok TP/FN/FP/TN
    grp = np.select(
        [(y_test == 1) & (pred == 1), (y_test == 1) & (pred == 0), (y_test == 0) & (pred == 1)],
        ["TP", "FN", "FP"], default="TN",
    )
    err = X_test.assign(kelompok=grp).groupby("kelompok")[FEATURES].median().T
    err.to_csv(out / "analisis_error_median_fitur.csv")
    grp_counts = pd.Series(grp).value_counts()

    # 6. Simpan pipeline lengkap + metadata -------------------------------------
    joblib.dump(best, out / "model_terbaik.pkl")
    meta = {
        "model": best_name,
        "dilatih_pada": datetime.now().isoformat(timespec="seconds"),
        "versi_sklearn": sklearn.__version__,
        "fitur_urutan": FEATURES,
        "ambang_keputusan": threshold,
        "parameter_terbaik": py(searches[best_name].best_params_),
        "jumlah_data": int(len(df)),
        "proporsi_gagal_bayar": float(pos_rate),
        "metrik_pemilihan": args.select_metric,
        "bobot_metrik": args.w_metric,
        "cv": py(comp.loc[comp["model"] == best_name].iloc[0][
            ["cv_f1", "cv_pr_auc", "cv_recall", "cv_precision"]].to_dict()),
        "test": {"pr_auc": pr_auc, "roc_auc": roc_auc, "ambang_0.5": m_default, "ambang_terpilih": m_tuned},
    }
    (out / "metadata.json").write_text(json.dumps(py(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    # 7. Ringkasan siap tempel ke laporan ---------------------------------------
    t = m_tuned
    show = comp[["model", "pencarian", "kombinasi_diuji", "cv_f1", "cv_pr_auc", "cv_recall",
                 "cv_precision", "interpretability_1_5", "skor_pemilihan"]]
    md = f"""# Ringkasan Hasil Mini Project

## 1. Data
- Jumlah baris dipakai: {len(df):,} (latih {len(X_train):,}, uji {len(X_test):,})
- Proporsi gagal bayar: {pos_rate:.2%} (data tidak seimbang -> akurasi menyesatkan, dipakai F1/recall/PR-AUC)
- Fitur dengan missing terbanyak: {miss.index[0]} ({miss.iloc[0]:.1f}%)

## 2. Komparasi model (Stratified 5-fold, split dan metrik sama)
{to_md(show)}

Parameter terbaik per model ada di `komparasi_model.csv`.

## 3. Pemilihan model
Skor = {args.w_metric} x {args.select_metric.upper()}(CV) + {round(1 - args.w_metric, 2)} x (Interpretability / 5)

{to_md(sens)}

**Model terpilih: {best_name}**

## 4. Evaluasi test set (dipakai sekali)
- PR-AUC: {pr_auc:.3f} | ROC-AUC: {roc_auc:.3f}
- Ambang 0,5: recall {m_default['recall']:.3f}, precision {m_default['precision']:.3f}, F1 {m_default['f1']:.3f}
- Ambang terpilih {threshold:.3f}: recall {t['recall']:.3f}, precision {t['precision']:.3f}, F1 {t['f1']:.3f}

Confusion matrix (ambang {threshold:.3f}): TP = {t['TP']}, FP = {t['FP']}, FN = {t['FN']}, TN = {t['TN']}

Perhitungan manual:
- Precision = {t['TP']} / ({t['TP']} + {t['FP']}) = {t['precision']:.3f}
- Recall = {t['TP']} / ({t['TP']} + {t['FN']}) = {t['recall']:.3f}
- F1 = 2 x {t['precision']:.3f} x {t['recall']:.3f} / ({t['precision']:.3f} + {t['recall']:.3f}) = {t['f1']:.3f}

## 5. Fitur paling berpengaruh
{to_md(imp.sort_values(ascending=False).head(5).rename('penurunan_pr_auc').reset_index().rename(columns={'index': 'fitur'}), 4)}

## 6. Analisis error
Jumlah per kelompok: {', '.join(f'{k}={v}' for k, v in grp_counts.items())}.
Median fitur tiap kelompok ada di `analisis_error_median_fitur.csv`. Bandingkan kelompok FN
(gagal bayar yang lolos) dengan TP untuk mencari pola yang terlewat model.

## 7. Grafik
`figures/`: eda_kelas, eda_missing, eda_korelasi, komparasi_model, confusion_matrix, pr_curve, kepentingan_fitur.
"""
    (out / "hasil_ringkas.md").write_text(md, encoding="utf-8")
    print(f"\nSelesai. Semua hasil ada di folder: {out}/")


if __name__ == "__main__":
    main()
