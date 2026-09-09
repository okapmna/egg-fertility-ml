# Egg Fertility Detection using YOLOv8

Proyek Machine Learning untuk mendeteksi kesuburan telur (Egg Candling Classification & Detection) menggunakan model **YOLOv8** (Ultralytics) dengan membedakan 2 kelas utama:
- `0`: **fertile** (telur fertil / berembrio)
- `1`: **infertile** (telur infertil / tidak berkembang / kosong)

Pipeline ini dirancang untuk mengunduh dataset secara otomatis dari Roboflow/Kaggle, menggabungkan dan mengharmonisasi anotasi split data (`train`, `valid`, `test`), melatih model YOLOv8, mengevaluasi metrik performa, serta mengekspor bobot model (PyTorch & ONNX).

---

## 📁 Struktur Proyek

```text
egg-fertility-ml/
├── .gitignore              # Mengabaikan kredensial, dataset, dan model weights
├── .env.example            # Template environment variable
├── requirements.txt        # Daftar dependensi Python
├── README.md               # Dokumentasi proyek
└── egg_fertil_ml.ipynb     # Notebook pipeline training & evaluasi
```

---

## Panduan Memulai

### 1. Prasyarat & Instalasi

Pastikan Python (>= 3.8) terinstal, lalu instal seluruh pustaka yang diperlukan:

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi API Key Roboflow

Proyek ini membutuhkan API Key dari [Roboflow Universe](https://universe.roboflow.com/) untuk mengunduh dataset.

#### Opsi A: Menjalankan di Lingkungan Lokal
1. Salin file `.env.example` menjadi `.env`:
   ```bash
   cp .env.example .env
   ```
2. Buka `.env` dan masukkan API Key Anda:
   ```ini
   ROBOFLOW_API_KEY=your_actual_roboflow_api_key
   ```

#### Opsi B: Menjalankan di Google Colab
1. Buka Google Colab.
2. Klik ikon kunci (Secrets) di panel sebelah kiri.
3. Tambahkan secret baru dengan nama **`ROBOFLOW_API_KEY`** dan masukkan nilai API Key Anda.
4. Berikan izin akses notebook ke secret tersebut.

*Catatan: Jika API key belum terpasang di `.env` maupun Colab Secrets, notebook akan secara otomatis menampilkan prompt interaktif yang aman untuk menginputkan API key.*

### 3. Menjalankan Pipeline

Buka dan jalankan notebook [`egg_fertil_ml.ipynb`](./egg_fertil_ml.ipynb) secara berurutan:
1. Pengecekan akselerator GPU (CUDA).
2. Download & penggabungan dataset ke folder lokal runtime.
3. Harmonisasi label format YOLOv8.
4. Training model YOLOv8n.
5. Evaluasi performa (mAP, Precision, Recall, Confusion Matrix).
6. Export model ke format ONNX.

---

## 🔒 Keamanan & Kebersihan Repositori

- File `.env` sudah masuk ke `.gitignore` sehingga tidak akan pernah ter-push ke GitHub.
- Output cell notebook telah dibersihkan sebelum commit agar ukuran repository tetap ringan dan riwayat Git tetap bersih.
