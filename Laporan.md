# LAPORAN TUGAS UTS: ANALISIS KLASIFIKASI DIAGNOSIS KANKER PAYUDARA
**Mata Kuliah:** Intelegensi Buatan  
**Dosen Pengampu:** Muhammad Ikhwan Fathulloh  
**Nama Mahasiswa:** Anisa Ansuya Henna  
**NIM:** 24552011020  
**Kelas:** TIF RP 23D)  


---

## 1. PENDAHULUAN DAN AUDIT DATA
Pada tugas ini, dilakukan pemodelan klasifikasi untuk mendiagnosis tumor payudara apakah bersifat **Jinak (Benign)** atau **Ganas (Malignant)** menggunakan dataset klinis `kasus02_breast_cancer.csv`. 

### Temuan Awal Audit Data:
* **Dimensi Dataset:** Terdiri atas **569 baris** data pasien dengan **31 kolom** (30 fitur klinis numerik hasil biopsi FNA dan 1 kolom target `diagnosis`).
* **Kondisi Data:** 
  * **Duplikat:** 0 baris (bersih dari data kembar).
  * **Missing Values:** Tidak ditemukan nilai kosong, jadi tidak memerlukan imputasi berat.
  * **Nilai Tidak Wajar:** Tidak ada nilai negatif terdeteksi pada seluruh fitur numerik geometri tumor.
  * **Distribusi Kelas Target:** Terjadi sedikit ketimpangan (*imbalance*) kelas:
    * **Jinak (Benign):** 357 sampel (~62.7%)
    * **Ganas (Malignant):** 212 sampel (~37.3%)

---

## 2. TEMUAN FEATURE ENGINEERING (`fitur.py`)
Dirancang skema rekayasa fitur agar meningkatkan kan performa model

1. **Penambahan Fitur Rasio Baru:**
   * **`rasio_worst_mean_radius`**: Mengukur rasio antara radius sel terburuk dengan radius rata-rata untuk melihat tingkat deviasi ukuran sel.
   * **`rasio_worst_mean_texture`**: Mengukur rasio tekstur sel terburuk dibanding rata-rata.
   * **`rasio_compactness_smoothness`**: Rasio antara kepadatan sel (`mean_compactness`) dan kelembutan sel (`mean_smoothness`) untuk menangkap anomali bentuk sel kanker yang biasanya lebih padat dan tidak beraturan.

2. **Seleksi Fitur Multikolinearitas:**
   * Banyak fitur bawaan (seperti `radius`, `perimeter`, dan `area`) memiliki korelasi linear yang sangat ekstrem ($|r| > 0.98$).
   * Diterapkan ambang batas korelasi $|r| > 0.92$. Seluruh fitur dengan korelasi di atas nilai tersebut dibuang secara otomatis dari data latih demi mencegah masalah multikolinearitas dan overfitting.

---

## 3. PERBANDINGAN MODEL & HASIL METRIK
Evaluasi perbandingan algoritma dilakukan menggunakan **5-Fold Stratified Cross-Validation** pada data latih (80% data), disusul pengujian final **satu kali** pada data uji (20% data).

### A. Performa Cross-Validation (Rata-rata 5-Fold pada Data Latih):

| Algoritma | Recall | ROC-AUC | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | **0.9588** | **0.9957** | **0.9643** | **0.9736** |
| **XGBoost Classifier** | 0.9588 | 0.9921 | 0.9644 | 0.9736 |

* **Model Terbaik Terpilih:** **Logistic Regression** dipilih karena memiliki skor **ROC-AUC (0.9957)** karena sedikit lebih unggul dan stabil dibandingkan XGBoost, dengan nilai **Recall (0.9588)** yang sama tinggi.

### B. Hasil Evaluasi Final (1 Kali Eksekusi Data Uji):

* **Accuracy:** 96.49%
* **Precision:** 97.50%
* **Recall:** 92.86%
* **F1-Score:** 95.12%
* **ROC-AUC:** 0.9970

#### Confusion Matrix Data Test:
* **True Negative (Benign terdeteksi Benign):** ada 71 pasien
* **False Positive (Benign terdeteksi Malignant):** ada 1 pasien
* **False Negative (Malignant terdeteksi Benign):** ada 3 pasien
* **True Positive (Malignant terdeteksi Malignant):** ada 39 pasien

---

## 4. JAWABAN TUGAS KHUSUS KASUS 2

### Pertanyaan 1: Banyak fitur saling berkorelasi tinggi. Jelaskan strategi yang Anda pilih: seleksi fitur, reduksi dimensi, atau keduanya!

**Jawaban:**
Saya memilih pendekatan Seleksi Fitur Berbasis Korelasi (Filter Method) dengan ambang batas korelasi $|r| > 0.92$. 
* Alasan Menolak Reduksi Dimensi (seperti PCA) adalah karena dalam ranah medis, interpretabilitas model sangatlah krusial.
  Teknik reduksi dimensi seperti PCA akan mentransformasikan fitur-fitur fisik (radius, area, tekstur) menjadi komponen utama (PC1, PC2, dst.)
  yang abstrak dan tidak lagi memiliki arti klinis langsung. Dokter tidak bisa menjelaskan secara logis mengapa model mengambil keputusan tersebut.
* Keuntungan Seleksi Fitur: Dengan mendeteksi pasangan fitur berkorelasi tinggi (misalnya `perimeter_mean` dan `radius_mean`) lalu membuang salah satunya,
  kita berhasil menghilangkan multikolinearitas ekstrem (menyisakan fitur independen) tetapi **tetap mempertahankan interpretabilitas fisik** dari fitur asli yang tersisa.

### Pertanyaan 2: Pada konteks medis, kesalahan jenis apa yang lebih mahal? Pilih metrik utama dan jelaskan alasannya!

**Jawaban:**
* **Kesalahan yang Lebih Mahal:** Kesalahan **False Negative (FN)** jauh lebih mahal dan fatal secara medis.
  * *False Negative* terjadi ketika seorang pasien yang sebenarnya memiliki kanker **Ganas (Malignant)** salah diprediksi oleh model sebagai **Jinak (Benign)**.
  * **Dampak Fatal:** Pasien tidak akan mendapatkan tindakan medis, kemoterapi, atau operasi segera, sehingga sel kanker akan terus menyebar secara senyap dan dapat berujung pada kematian.
  * Sebaliknya, *False Positive* (pasien jinak diprediksi ganas) hanya akan menyebabkan kecemasan sementara dan dapat dikoreksi melalui biopsi atau tes sekunder lanjutan.
* **Pemilihan Metrik Utama:** Berdasarkan kondisi di atas, metrik utama yang digunakan adalah **Recall (Sensitivity)**.
  $$\text{Recall} = \frac{\text{True Positive (TP)}}{\text{True Positive (TP)} + \text{False Negative (FN)}}$$
  Fokus utama model adalah memperkecil nilai penyebut bagian *False Negative (FN)* hingga mendekati nol, guna menjamin tidak ada pasien kanker ganas yang lolos dari deteksi.

---

## 5. INTERPRETASI KLINIS DAN KESIMPULAN
1. Model **Logistic Regression** yang dikombinasikan dengan kustomisasi kelas penyeimbang (`class_weight="balanced"`) berhasil mencapai keseimbangan performa yang sangat luar biasa dengan
   **ROC-AUC sebesar 0.9970** pada data uji.
3. Dari 42 kasus kanker ganas pada data uji, model berhasil mendeteksi **39 pasien dengan benar (Recall 92.86%)** dan hanya melewatkan 3 pasien.
4. Model ini berarti layak diintegrasikan ke dalam sistem penunjang keputusan klinis (*Clinical Decision Support System*) karena proses prapemrosesan datanya yang bersih,
   aman dari kebocoran data (*data leakage*), serta menghasilkan keputusan klasifikasi yang transparan dan dapat dipahami secara medis.
