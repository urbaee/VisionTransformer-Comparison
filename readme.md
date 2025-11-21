# Vision Transformer Comparison

Perbandingan tiga arsitektur Vision Transformer (DeiT Small, Swin Transformer Tiny, dan EVA-02 Small) pada dataset Animal-5 Mammal. Proyek ini mencakup proses training, evaluasi metrik performa, waktu inferensi, visualisasi hasil, serta analisis mendalam untuk menentukan model terbaik berdasarkan berbagai use case.

---

## 👤 Author
**Nama:** Rahmat Aldi Nasda
**NIM:** 122140077  
---

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk:
- Membandingkan performa DeiT Small, Swin Tiny, dan EVA-02 Small pada dataset Animal-5 Mammal.
- Mengukur akurasi, precision, recall, F1-score, dan waktu inferensi.
- Menganalisis stabilitas training melalui learning curves.
- Menyajikan visualisasi seperti confusion matrix dan contoh prediksi.
- Memberikan rekomendasi model terbaik berdasarkan kebutuhan (akuras tertinggi, efisiensi komputasi, atau aplikasi real-time).

Setiap model diuji pada dua konfigurasi training:
1. Batch size 16, learning rate 1e-4  
2. Batch size 32, learning rate 1e-5  

---

## 📂 Struktur Folder
```
VisionTransformer-Comparison/
│
├── best_model/                # Model terbaik dari masing-masing arsitektur
├── confmatrics_result/        # Confusion matrix (raw dan normalized)
├── inference/                 # Hasil inference dan waktu inferensi
├── plot_result/               # Learning curves (loss & accuracy)
├── result_image/              # Contoh prediksi 8 sampel
│
├── VisionTransformer-Comparison.ipynb   # Notebook utama training dan evaluasi
├── inference.py               # Script inference mandiri
├── requirements.txt           # Dependencies proyek
└── README.md                  # Dokumentasi
```

---

## 🛠️ Instalasi Requirements
Jalankan perintah berikut:
```
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Training
Gunakan file:
```
VisionTransformer-Comparison.ipynb
```
Notebook ini mencakup:
- Loading dataset
- Transformasi dan augmentasi data
- Training
- Evaluasi metrik
- Pembuatan confusion matrix
- Visualisasi learning curve
- Penyimpanan model terbaik

Jalankan semua sel secara berurutan.

---

## ⚡ Menjalankan Inference (Benchmarking)

Script `inference.py` digunakan untuk melakukan *benchmarking* waktu inferensi pada seluruh test set, bukan untuk memprediksi satu gambar.

### 1. Atur model dan checkpoint di bagian paling atas file
Edit variabel berikut di `inference.py`:

```python
TIMM_MODEL_NAME = MODEL_EVA_SMALL      # pilih: MODEL_DEIT_SMALL / MODEL_SWIN_TINY / MODEL_EVA_SMALL
CHECKPOINT_PATH = CKPT_EVA_2           # pilih checkpoint dari training 1 atau 2
```

Contoh untuk DeiT training ke-2:

```python
TIMM_MODEL_NAME = MODEL_DEIT_SMALL
CHECKPOINT_PATH = CKPT_DEIT_2
```

### 2. Jalankan script
```bash
python inference.py
```

### 3. Hasil inference otomatis akan tersimpan ke:
```
inference/inference_<nama_training>.txt
```

Isi file mencakup:
- total gambar
- average inference time (ms)
- std dev
- throughput (img/s)
- nama model & checkpoint yang digunakan
- informasi hardware GPU/CPU

---

## 📈 Hasil Utama Proyek
- Semua model mencapai akurasi > 98% pada konfigurasi training kedua.
- Swin Transformer Tiny: akurasi tertinggi.
- DeiT Small: inferensi tercepat.
- EVA-02 Small: kombinasi paling seimbang antara akurasi dan efisiensi.

---

