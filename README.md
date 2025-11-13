# 🏨 Hotel Booking Cancellation Prediction: Mengurangi Kerugian Overbooking dengan Machine Learning

Model Klasifikasi Biner ini dibangun untuk memprediksi probabilitas pembatalan pemesanan hotel, memungkinkan tim *Revenue Management* mengambil tindakan proaktif untuk memaksimalkan pendapatan dan memitigasi risiko *overbooking* yang tidak terkelola.

---

## 1. Konteks Bisnis dan Penjelasan Proyek

### Permasalahan Bisnis
Industri perhotelan sering menghadapi tantangan **overbooking yang tidak terkelola** karena tingginya angka pesanan yang diasumsikan tidak dibatalkan. Hal ini berujung pada dua skenario kerugian utama:
1.  **Kerugian Pendapatan:** Terjadinya kamar kosong (*no-show* yang tidak terduga).
2.  **Kerugian Reputasi:** Harus memindahkan tamu yang sudah tiba (*walk-in*) karena kamar penuh.

### Tujuan Model
Membangun model *Machine Learning* Klasifikasi Biner (`is_canceled` target: 0/1) untuk **mengidentifikasi pemesanan berisiko tinggi batal** secara akurat. Fokus utama proyek ini adalah pada **dampak finansial** dari setiap prediksi, yaitu meminimalkan kerugian terbesar.

### Pemangku Kepentingan (Stakeholder)
* **Manajemen Hotel:** Pengambilan keputusan strategis.
* **Tim Revenue Management:** Strategi harga dan manajemen inventaris.
* **Tim Operasional:** Penanganan dan layanan tamu.

---

## 2. Metrik Evaluasi dan Pemilihan Scoring

### Metrik yang Digunakan
Model dievaluasi menggunakan metrik klasifikasi standar:
* `Recall Score`
* `Confusion Matrix`
* `PR AUC (Average Precision Score)`

### Justifikasi Pemilihan Scoring (Recall)
Metrik optimasi utama yang dipilih adalah **Recall**. Alasannya didasarkan pada analisis biaya kesalahan (*Cost-Benefit Analysis*):

| Jenis Kesalahan | Defenisi Model | Dampak Bisnis (Kerugian) | Biaya Per Kasus (Contoh) |
| :--- | :--- | :--- | :--- |
| **False Negative (FN)** | Memprediksi Tidak Batal, padahal Sebenarnya Batal. | **Kamar Kosong** (Kerugian Pendapatan) | **Rp450.000** |
| **False Positive (FP)** | Memprediksi Batal, padahal Sebenarnya Tidak Batal. | Salah Duga *No-Show* (Biaya Penanganan / *Overbooking* Aman) | **Rp50.000** |

Karena **biaya FN (Kamar Kosong) jauh lebih besar** daripada biaya FP, model harus dioptimalkan untuk **meminimalkan FN**. Metrik **Recall** (kemampuan model untuk menangkap semua kasus positif Batal yang sebenarnya) adalah yang paling efektif untuk tujuan ini.

### Model dan Teknik Utama
* **Algoritma Terbaik:** **Random Forest Classifier**.
* **Penanganan Data Tidak Seimbang:** Digunakan teknik **SMOTE (Synthetic Minority Over-sampling Technique)** untuk menyeimbangkan kelas *Canceled* (Minoritas, 36.9%) dan *Not Canceled* (Mayoritas, 63.1%).

---

## 3. Library yang Digunakan

Proyek ini menggunakan beberapa *library* utama Python (versi spesifik dapat dilihat di `requirements.txt`):

| Kategori | Library | Tujuan Utama |
| :--- | :--- | :--- |
| **Data & Core** | `pandas`, `numpy` | Manipulasi data, aljabar linear. |
| **Modeling** | `scikit-learn`, `xgboost` | Implementasi model, validasi silang, metrik. |
| **Imbalance** | `imblearn` (SMOTE) | Penanganan data tidak seimbang dalam *pipeline*. |
| **Preprocessing** | `category-encoders` | *Encoding* fitur kategorikal (misalnya `BinaryEncoder`). |
| **Interpretability**| `shap` | Analisis kontribusi fitur pada setiap prediksi (SHAP Value/Waterfall Plot). |
| **Visualisasi** | `matplotlib`, `seaborn` | Eksplorasi Data dan Visualisasi Hasil. |
| **Deployment** | `streamlit`, `pickle`, `dill` | *Web application* interaktif dan persistensi model/explainer. |

---

## 4. Dampak dan Rekomendasi Bisnis

### Analisis Finansial (Potensi Penghematan)
Berdasarkan simulasi pada data uji, penerapan model yang telah disetel (*tuned model*) menunjukkan potensi penghematan signifikan:

* **Total Kerugian Tanpa Model (Baseline):** Diperkirakan sekitar **Rp2.76 Miliar** (hanya dari FN/kamar kosong).
* **Total Kerugian Dengan Model (Final):** Berkurang drastis menjadi sekitar **Rp656 Juta**.
* **Total Penghematan Bersih:** Model ini memberikan potensi penghematan sebesar **Rp2.1 Miliar**.

### Fitur Paling Berpengaruh (SHAP Analysis)
Analisis SHAP menunjukkan fitur-fitur kunci yang mendorong keputusan pembatalan:
* **`deposit_type_Non Refund`**: Merupakan indikator risiko pembatalan yang sangat tinggi.
* **`total_of_special_requests`**: Merupakan pendorong negatif (komitmen tamu tinggi, risiko pembatalan rendah).
* **`market_segment_Online TA`**: Nilai tinggi menunjukkan peningkatan risiko batal yang signifikan.

### Rekomendasi Aksi
1.  **Strategi *Overbooking* Adaptif:** Untuk pemesanan yang diprediksi berisiko tinggi batal (terutama dari segmen **"Online TA"**), tim *Revenue Management* dapat menerapkan strategi *overbooking* yang lebih agresif untuk memaksimalkan okupansi.
2.  **Insentif Konfirmasi:** Tawarkan insentif konfirmasi atau *upgrade* terbatas kepada tamu dengan risiko batal tinggi untuk mengamankan pemesanan mereka.
3.  **Penggunaan Deposit:** Mengingat Deposit Non-Refundable memiliki tingkat pembatalan tertinggi (90%), kebijakan deposit harus diulas untuk memitigasi risiko.
   

### Streamlit
berikut file demo streamlit beserta model yang digunakan :https://drive.google.com/drive/folders/1gB_CpjTo-OUu_eGpjGYbrVHhCBhuwIkq?usp=sharing
