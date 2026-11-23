# Laporan Ujian Tengah Semester — Intelegensi Buatan
**Universitas Teknologi Bandung — Departemen Teknik Informatika (Kelas TIF RP 23D)**

* **Nama:** Anisa Ansuya Henna
* **NIM:** 24552011020

---

## 1. Penentuan Kasus Berdasarkan NIM
* NIM Anda: `24552011020`
* Sisa bagi dengan 10: $22 \bmod 10 = 2$
* **Kasus Terpilih:** **Kasus 2 — Diagnosis Kanker Payudara** (`kasus02_breast_cancer.csv`, target: `diagnosis`)

---

## 2. Alur Pembagian & Eksplorasi Data
* **Dataset:** Memiliki 569 baris data dan 31 kolom.
* **Audit Data:** Tidak ditemukan baris duplikat, tidak ada data kosong (*missing values*), serta tidak ada nilai negatif yang tidak wajar.
* **Rasio Kelas Target:** Terdiri dari 357 kasus Jinak (*Benign*) dan 212 kasus Ganas (*Malignant*).
* **Strategi Penanganan Korelasi Tinggi & Multikolinearitas:**
  * Seleksi fitur berbasis nilai korelasi threshold $|r| > 0.92$ diimplementasikan secara otomatis menggunakan kelas kustom `RekayasaFiturKanker`.
  * Metode seleksi ini dipilih untuk mempertahankan fitur fisik yang mudah diinterpretasikan dibanding PCA.
  * Penanganan multikolinearitas ini dihitung **hanya pada data train saja** untuk menghindari kebocoran data (*data leakage*).

---

## 3. Strategi Pembagian Data & Pemodelan
* **Pemisahan Data:** Dataset dibagi menjadi **80% Data Latih (455 sampel)** dan **20% Data Uji (114 sampel)** menggunakan pemisahan terstratifikasi (`stratify=y`) untuk menjaga rasio distribusi target.
* **Uji Banding Model:** Menggunakan 5-Fold Stratified Cross-Validation pada data latih dengan membandingkan:
  1. **Logistic Regression** (menggunakan parameter `class_weight="balanced"`)
  2. **XGBoost Classifier** (menggunakan parameter `scale_pos_weight`)

---

## 4. Evaluasi Model & Justifikasi Medis
* **Metrik Utama:** Dalam diagnosis kanker payudara, kesalahan **False Negative (FN)**—di mana pasien penderita kanker ganas diprediksi jinak—sangatlah fatal. Oleh karena itu, metrik **Recall** kelas positif (Ganas) beserta **ROC-AUC** diprioritaskan utama.
* **Hasil 5-Fold Cross-Validation:**
  * **Logistic Regression:** Recall: **0.9588** | ROC-AUC: **0.9957**
  * **XGBoost:** Recall: **0.9588** | ROC-AUC: **0.9921**
  * *Logistic Regression dipilih karena nilai kestabilan ROC-AUC yang lebih tinggi.*
* **Hasil Akhir Uji Data Test:**
  * Akurasi (*Accuracy*): **96.49%**
  * Presisi (*Precision*): **97.50%**
  * Sensitivitas (*Recall*): **92.86%**
  * F1-Score: **95.12%**
  * ROC-AUC: **0.9970**

---

## 5. Implementasi Aplikasi Web Streamlit
<img width="968" height="596" alt="image" src="https://github.com/user-attachments/assets/cf54eaf2-3ccf-4d62-96bd-b8cc1291a004" />

<img width="459" height="326" alt="image" src="https://github.com/user-attachments/assets/d8764d13-e2ba-4c3b-b063-6231db8cc8bf" />

<img width="469" height="333" alt="image" src="https://github.com/user-attachments/assets/4e84ff8f-299d-47ed-b5e5-c8f87663621a" />


