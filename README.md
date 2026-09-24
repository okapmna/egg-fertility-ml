# Egg Fertility Detection using YOLOv8

Proyek Machine Learning untuk mendeteksi kesuburan telur (Egg Candling Classification & Detection) menggunakan model **YOLOv8** (Ultralytics) dengan membedakan 2 kelas utama:
- `0`: **fertile** (telur fertil / berembrio)
- `1`: **infertile** (telur infertil / tidak berkembang / kosong)

Pipeline ini dirancang untuk mengunduh dataset secara otomatis dari Roboflow/Kaggle, menggabungkan dan mengharmonisasi anotasi split data (`train`, `valid`, `test`), melatih model YOLOv8, mengevaluasi metrik performa, serta mengekspor bobot model (PyTorch & ONNX).

---

## Struktur Proyek

```text
egg-fertility-ml/
├── .gitignore                      # Mengabaikan kredensial, dataset, dan model weights
├── .env.example                    # Template environment variable
├── requirements.txt                # Dependensi training utama
├── requirements-web.txt            # Dependensi web demo (Flask, PyTorch, Torchvision, Ultralytics)
├── README.md                       # Dokumentasi proyek
├── app.py                          # Aplikasi web demo (Flask) untuk inferensi deteksi + klasifikasi
├── templates/
│   └── index.html                  # Antarmuka web modern untuk upload gambar & visualisasi hasil
├── model/                          # Direktori penyimpanan bobot model (diabaikan oleh git)
│   └── .gitkeep
├── egg_fertil_ml.ipynb             # Notebook pipeline YOLOv8 end-to-end (deteksi & klasifikasi langsung)
└── train_mobilenet_classifier.ipynb # Notebook training classifier MobileNetV2 (arsitektur dua-tahap)
```

---

## Alur Pipeline & Arsitektur

Proyek ini mendukung dua arsitektur deteksi kesuburan telur:
1. **Deteksi Langsung (YOLOv8)**: Dilatih pada [`egg_fertil_ml.ipynb`](./egg_fertil_ml.ipynb).
2. **Arsitektur Dua-Tahap (Two-Stage Pipeline)**:
   - **Tahap 1 (Deteksi Telur)**: YOLO11n mendeteksi lokasi telur dalam gambar candling.
   - **Tahap 2 (Klasifikasi Fertilitas)**: Crop telur dimasukkan ke MobileNetV2 (input 320x320) untuk membedakan `fertile` (berembrio/pembuluh darah) vs `infertile` (kosong).
   - Training classifier dilakukan pada [`train_mobilenet_classifier.ipynb`](./train_mobilenet_classifier.ipynb) dengan augmentasi khusus anti-shortcut learning (ColorJitter & Grayscale) untuk mencegah bias warna candling.

---

## Panduan Memulai

### 1. Prasyarat & Instalasi

Pastikan Python (>= 3.8) terinstal:

- Untuk kebutuhan training:
  ```bash
  pip install -r requirements.txt
  ```

- Untuk menjalankan web demo:
  ```bash
  pip install -r requirements-web.txt
  ```

### 2. Menjalankan Web Demo (Inference)

1. Pastikan file model sudah ditempatkan di direktori `model/`:
   - `model/yolo11n.pt` (detector)
   - `model/mobilenet_egg_best.pt` (classifier)
2. Jalankan aplikasi web:
   ```bash
   python app.py
   ```
3. Buka browser di `http://127.0.0.1:5000`. Anda dapat mengunggah foto candling telur dan menyesuaikan threshold deteksi serta fertilitas secara interaktif.

### 3. Training & Pengembangan

- **Klasifikasi MobileNetV2**: Buka [`train_mobilenet_classifier.ipynb`](./train_mobilenet_classifier.ipynb) (di Colab / lokal) untuk melatih classifier dengan fitur transfer learning, penyeimbangan sampling, dan evaluasi F-score.
- **Deteksi YOLOv8**: Buka [`egg_fertil_ml.ipynb`](./egg_fertil_ml.ipynb) untuk melatih detektor bounding box telur.

---

## Keamanan & Kebersihan Repositori

- File `.env` dan file bobot besar (`*.pt`, `*.onnx`) diabaikan oleh `.gitignore` sehingga tidak ter-push ke repository.
- Output cell notebook telah dibersihkan agar ukuran riwayat Git tetap ringan.
