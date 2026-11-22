from pathlib import Path
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fitur import RekayasaFiturKanker, audit_data, encode_target


def main():
    path_dataset = Path("dataset/kasus02_breast_cancer.csv")
    folder_model = Path("models")
    folder_model.mkdir(parents=True, exist_ok=True)
    path_output_model = folder_model / "model_kasus02.joblib"

    if not path_dataset.exists():
        raise FileNotFoundError(
            f"File {path_dataset} tidak ditemukan. Letakkan kasus02_breast_cancer.csv di folder dataset/."
        )

    # 1. Muat & Audit Data
    df = pd.read_csv(path_dataset)
    audit_data(df)
    df = df.drop_duplicates().reset_index(drop=True)

    kolom_buang_awal = [
        c for c in df.columns
        if c.strip().lower() in ["id", "unnamed: 0", "unnamed: 32"] or c.strip().lower().startswith("unnamed")
    ]
    if kolom_buang_awal:
        df = df.drop(columns=kolom_buang_awal)

    X = df.drop(columns=["diagnosis"])
    y = encode_target(df["diagnosis"])

    # 2. Pembagian Data Train & Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    rasio_kelas_train = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Data Train: {len(X_train)} sampel | Data Test: {len(X_test)} sampel")
    print(f"Rasio scale_pos_weight XGBoost (dari data train): {rasio_kelas_train:.4f}\n")

    # 3. Perbandingan 2 Algoritma dalam Pipeline scikit-learn (Saya memilih Logistic Regression dan Gradient Boosting)
    daftar_pipeline = {
        "Logistic Regression": Pipeline([
            ("fe", RekayasaFiturKanker(corr_threshold=0.92)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "XGBoost (Gradient Boosting)": Pipeline([
            ("fe", RekayasaFiturKanker(corr_threshold=0.92)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=rasio_kelas_train,
                eval_metric="logloss",
                random_state=42,
            )),
        ]),
    }

    # 4. Pemilihan Model dengan 5-Fold Cross-Validation pada Data Train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    daftar_metrik = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    hasil_cv = {}

    print("=== 2. PERBANDINGAN ALGORITMA (5-FOLD CV PADA DATA TRAIN) ===")
    for nama_algo, pipe in daftar_pipeline.items():
        skor = cross_validate(pipe, X_train, y_train, cv=cv, scoring=daftar_metrik)
        hasil_cv[nama_algo] = {m: float(skor[f"test_{m}"].mean()) for m in daftar_metrik}
        print(
            f"{nama_algo:28s} -> Recall: {hasil_cv[nama_algo]['recall']:.4f} | "
            f"ROC-AUC: {hasil_cv[nama_algo]['roc_auc']:.4f} | "
            f"F1: {hasil_cv[nama_algo]['f1']:.4f} | "
            f"Accuracy: {hasil_cv[nama_algo]['accuracy']:.4f}"
        )

    model_terbaik_nama = max(
        hasil_cv.keys(), key=lambda k: (hasil_cv[k]["recall"], hasil_cv[k]["roc_auc"])
    )
    pipeline_terbaik = daftar_pipeline[model_terbaik_nama]
    print(f"\nModel Terbaik Terpilih (Prioritas Recall Medis): {model_terbaik_nama}\n")

    # 5. Evaluasi SATU KALI pada Data Test
    pipeline_terbaik.fit(X_train, y_train)
    y_pred = pipeline_terbaik.predict(X_test)
    y_prob = pipeline_terbaik.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrik_test = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
    }

    print("=== 3. EVALUASI FINAL DATA TEST (1 KALI EKSEKUSI) ===")
    for k, v in metrik_test.items():
        if k != "confusion_matrix":
            print(f"{k.upper():10s}: {v:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Jinak", "Ganas"]))

    # 6. Simpan Pipeline Lengkap + Metadata dengan joblib
    tahap_fe = pipeline_terbaik.named_steps["fe"]
    paket_deploy = {
        "pipeline": pipeline_terbaik,
        "nama_model": model_terbaik_nama,
        "fitur_mentah": X.columns.tolist(),
        "nilai_median_default": X_train.median().to_dict(),
        "kolom_dibuang_korelasi": tahap_fe.kolom_dibuang_,
        "fitur_terpilih": tahap_fe.fitur_terpilih_,
        "hasil_cv": hasil_cv,
        "metrik_test": metrik_test,
    }
    joblib.dump(paket_deploy, path_output_model)
    print(f"=== 4. DEPLOY JOBLIB SUKSES: {path_output_model} ===")


if __name__ == "__main__":
    main()
