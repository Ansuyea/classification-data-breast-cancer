%%writefile app.py
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diagnosis Kanker Payudara - Kasus 2", layout="wide")
PATH_MODEL = Path("models/model_kasus02.joblib")


@st.cache_resource
def muat_model():
    if not PATH_MODEL.exists():
        return None
    return joblib.load(PATH_MODEL)


def main():
    st.title(" Prediksi Diagnosis Kanker Payudara")
    st.caption("UTS Intelegensi Buatan — Kasus 2 (NIM: 24552011020)")

    paket = muat_model()
    if paket is None:
        st.error("Model belum ditemukan. Jalankan `python latih.py` terlebih dahulu.")
        st.stop()

    pipeline = paket["pipeline"]
    fitur_mentah = paket["fitur_mentah"]
    nilai_default = paket["nilai_median_default"]
    metrik_test = paket["metrik_test"]
    hasil_cv = paket["hasil_cv"]

    with st.sidebar:
        st.header("Metrik Evaluasi Model")
        st.write(f"**Algoritma Terpilih:** {paket['nama_model']}")
        st.write(f"**Fitur Mentah → Akhir:** {len(fitur_mentah)} → {len(paket['fitur_terpilih'])} fitur")

        c1, c2 = st.columns(2)
        c1.metric("Recall (Utama)", f"{metrik_test['recall']*100:.2f}%")
        c2.metric("ROC-AUC", f"{metrik_test['roc_auc']:.4f}")
        c1.metric("Accuracy", f"{metrik_test['accuracy']*100:.2f}%")
        c2.metric("F1-Score", f"{metrik_test['f1']*100:.2f}%")
        st.metric("Precision", f"{metrik_test['precision']*100:.2f}%")

        with st.expander("Detail Cross-Validation & Confusion Matrix"):
            st.write("**Rata-rata 5-Fold CV (Data Train):**")
            st.dataframe(pd.DataFrame(hasil_cv).T)
            st.write("**Confusion Matrix (Data Test):**")
            st.dataframe(
                pd.DataFrame(
                    metrik_test["confusion_matrix"],
                    index=["Aktual: Jinak", "Aktual: Ganas"],
                    columns=["Prediksi: Jinak", "Prediksi: Ganas"],
                )
            )

    tab_manual, tab_csv = st.tabs(["Input Manual (30 Fitur Mentah)", "Unggah File CSV"])

    with tab_manual:
        st.info("Nilai awal diisi otomatis dengan nilai median data train. Ubah angka sesuai hasil lab pasien.")
        with st.form("form_input_mentah"):
            t_mean, t_err, t_worst = st.tabs(["Rata-rata (mean)", "Galat (error)", "Terburuk (worst)"])
            input_user = {}

            kelompok_fitur = [
                (t_mean, [f for f in fitur_mentah if f.startswith("mean")]),
                (t_err, [f for f in fitur_mentah if f.endswith("error")]),
                (t_worst, [f for f in fitur_mentah if f.startswith("worst")]),
            ]
            for tab_aktif, daftar_f in kelompok_fitur:
                with tab_aktif:
                    cols = st.columns(3)
                    for i, f_nama in enumerate(daftar_f):
                        with cols[i % 3]:
                            input_user[f_nama] = st.number_input(
                                f_nama, value=float(nilai_default[f_nama]), format="%.5f"
                            )

            tombol_prediksi = st.form_submit_button(" Prediksi Diagnosis", type="primary")

        if tombol_prediksi:
            df_input = pd.DataFrame([input_user])[fitur_mentah]
            hasil_kelas = int(pipeline.predict(df_input)[0])
            prob = pipeline.predict_proba(df_input)[0]
            prob_jinak, prob_ganas = float(prob[0]), float(prob[1])

            st.divider()
            col_hasil, col_p1, col_p2 = st.columns([2, 1, 1])
            with col_hasil:
                if hasil_kelas == 1:
                    st.error("### Hasil Prediksi: GANAS (Malignant)")
                else:
                    st.success("### Hasil Prediksi: JINAK (Benign)")
            col_p1.metric("Probabilitas Ganas", f"{prob_ganas*100:.2f}%")
            col_p2.metric("Probabilitas Jinak", f"{prob_jinak*100:.2f}%")
            st.progress(prob_ganas, text=f"Skor Probabilitas Keganasan: {prob_ganas*100:.2f}%")

    with tab_csv:
        file_csv = st.file_uploader("Unggah file CSV berformat sama dengan kasus02_breast_cancer.csv", type=["csv"])
        if file_csv is not None:
            df_baru = pd.read_csv(file_csv)
            preds = pipeline.predict(df_baru[fitur_mentah])
            probs = pipeline.predict_proba(df_baru[fitur_mentah])

            df_baru["Prediksi_Diagnosis"] = ["Ganas" if p == 1 else "Jinak" for p in preds]
            df_baru["Probabilitas_Ganas (%)"] = (probs[:, 1] * 100).round(2)
            df_baru["Probabilitas_Jinak (%)"] = (probs[:, 0] * 100).round(2)
            st.dataframe(df_baru[["Prediksi_Diagnosis", "Probabilitas_Ganas (%)", "Probabilitas_Jinak (%)"] + fitur_mentah[:4]])


if __name__ == "__main__":
    main()
