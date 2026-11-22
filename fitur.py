import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def audit_data(df: pd.DataFrame) -> None:
    """Melakukan audit awal dataset: ukuran, duplikat, missing value, dan nilai tidak wajar."""
    print("=== 1. HASIL AUDIT DATA ===")
    print(f"Ukuran dataset       : {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"Jumlah baris duplikat: {df.duplicated().sum()}")

    missing = df.isnull().sum()
    missing_ada = missing[missing > 0]
    print(f"Missing value        : {'Tidak ada (0)' if len(missing_ada) == 0 else missing_ada.to_dict()}")

    kolom_numerik = df.select_dtypes(include=[np.number]).columns
    nilai_negatif = (df[kolom_numerik] < 0).sum()
    negatif_ada = nilai_negatif[nilai_negatif > 0]
    print(f"Nilai tidak wajar    : {'Tidak ada nilai negatif' if len(negatif_ada) == 0 else negatif_ada.to_dict()}")

    if "diagnosis" in df.columns:
        print("\nDistribusi Target (diagnosis):")
        print(df["diagnosis"].value_counts())
    print("===========================\n")


def encode_target(y: pd.Series) -> np.ndarray:
    """Mengubah label target menjadi angka biner: Ganas = 1 (kelas positif), Jinak = 0."""
    peta_label = {
        "ganas": 1, "m": 1, "malignant": 1, "1": 1,
        "jinak": 0, "b": 0, "benign": 0, "0": 0,
    }
    y_bersih = y.astype(str).str.strip().str.lower().map(peta_label)
    if y_bersih.isnull().any():
        raise ValueError(f"Label target tidak dikenali: {y[y_bersih.isnull()].unique()}")
    return y_bersih.astype(int).values


class RekayasaFiturKanker(BaseEstimator, TransformerMixin):
    """
    Transformer scikit-learn untuk:
    1. Membuat 3 fitur rasio baru yang menggambarkan heterogenitas ukuran & tekstur sel.
    2. Menyeleksi fitur dengan membuang multikolinearitas tinggi (|r| > corr_threshold)
       yang dihitung HANYA pada data train saat fit().
    """

    def __init__(self, corr_threshold: float = 0.92):
        self.corr_threshold = corr_threshold

    def _tambah_rasio(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        kolom_buang = [
            c for c in df.columns
            if c in ["id", "unnamed:_0", "unnamed:_32"] or c.startswith("unnamed")
        ]
        if kolom_buang:
            df = df.drop(columns=kolom_buang)

        eps = 1e-6
        if "worst_radius" in df.columns and "mean_radius" in df.columns:
            df["rasio_worst_mean_radius"] = df["worst_radius"] / (df["mean_radius"] + eps)
        if "worst_texture" in df.columns and "mean_texture" in df.columns:
            df["rasio_worst_mean_texture"] = df["worst_texture"] / (df["mean_texture"] + eps)
        if "mean_compactness" in df.columns and "mean_smoothness" in df.columns:
            df["rasio_compactness_smoothness"] = df["mean_compactness"] / (df["mean_smoothness"] + eps)

        return df

    def fit(self, X: pd.DataFrame, y=None):
        df = self._tambah_rasio(X)
        matriks_korelasi = df.corr().abs()
        segitiga_atas = matriks_korelasi.where(
            np.triu(np.ones(matriks_korelasi.shape), k=1).astype(bool)
        )
        self.kolom_dibuang_ = [
            col for col in segitiga_atas.columns if any(segitiga_atas[col] > self.corr_threshold)
        ]
        self.fitur_terpilih_ = [c for c in df.columns if c not in self.kolom_dibuang_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._tambah_rasio(X)
        return df[self.fitur_terpilih_]
