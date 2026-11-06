# 🔬 Mobile-PolypNet: Lightweight Colon Polyp Segmentation Network

Implementasi lengkap dan reproduksi eksperimen metode segmentasi polip usus besar (**Mobile-PolypNet**) menggunakan dataset **Kvasir-SEG** (`images/` dan `masks/`).

Proyek ini dibangun berdasarkan publikasi ilmiah:
> **Mobile-PolypNet: Lightweight Colon Polyp Segmentation Network for Low-Resource Settings**  
> *Ranit Karmakar and Saeid Nooshabadi*  
> **MDPI Journal of Imaging**, Vol. 8, No. 6, Art. 169 (2022).  
> DOI: [10.3390/jimaging8060169](https://doi.org/10.3390/jimaging8060169)

---

## 📋 Fitur & Keunggulan Model

1. **Ultra-Lightweight Autoencoder:**
   - Memanfaatkan **Bottleneck Inverted Residual Blocks** (MobileNetV2 style) dengan **Depthwise Separable Convolution** (Depthwise 3×3 + Pointwise 1×1).
   - **Hanya ~226.129 parameter** (vs U-Net: 7,85 juta parameter, PraNet: 20,52 juta parameter).
   - Ukuran model hanya **~0.86 MB**, sangat ideal untuk komputasi perangkat medis portabel / *low-resource settings*.

2. **DICE Loss & Binary Cross-Entropy:**
   - Menangani *imbalance* kelas antara luas polip dan dinding usus secara optimal.

3. **Metrik Lengkap Sesuai Standar Jurnal:**
   - DICE Coefficient (F1-score)
   - mIoU (Mean Intersection over Union / Jaccard Index)
   - $F_2$-Score ($\beta=2$, menitikberatkan pada sensitivitas/recall polip)
   - MAE (Mean Absolute Error)
   - Sensitivitas (Recall), Spesifisitas, dan Akurasi Piksel

4. **Pipeline Cepat & Terstruktur:**
   - Pre-loading memori RAM & PyTorch tensor ops untuk mempercepat komputasi CPU laptop.
   - Script modular untuk training, evaluasi, prediksi citra tunggal, serta Jupyter Notebook interaktif.

---

## 📁 Struktur Direktori

```text
Kvasir-SEG/
│
├── images/                             # 1000 citra endoskopi polip (.jpg)
├── masks/                              # 1000 ground truth mask biner (.jpg)
│
├── model.py                            # Arsitektur Mobile-PolypNet (PyTorch)
├── dataset.py                          # Data loader, augmentasi & split 80/10/10
├── losses_and_metrics.py               # Fungsi Loss (BCE+Dice) & metrik evaluasi
├── train.py                            # Script pelatihan lengkap dengan Early Stopping
├── evaluate.py                         # Script evaluasi test set & pembuatan tabel
├── predict.py                          # Inferensi citra tunggal / batch
│
├── Mobile_PolypNet_Experiment.ipynb    # Jupyter Notebook eksperimen interaktif
│
├── saved_models/                       # Tempat penyimpanan bobot model terbaik (.pth)
│   └── best_mobile_polypnet.pth
│
└── results/                            # Hasil plot kurva loss & visualisasi segmentasi
    ├── training_history.csv
    ├── loss_dice_curve.png
    ├── test_metrics.csv
    └── test_visual_results.png
```

---

## 🚀 Panduan Penggunaan

### 1. Menjalankan Jupyter Notebook (Rekomendasi untuk Laporan/Presentasi)
Buka terminal dan jalankan:
```bash
jupyter notebook Mobile_PolypNet_Experiment.ipynb
```
Notebook ini memuat visualisasi bertahap dari eksplorasi dataset, arsitektur model, proses training, grafik loss/dice, hingga visualisasi overlay segmentasi berwarna.

---

### 2. Melatih Model (Training)
Untuk melatih model secara penuh pada dataset:
```bash
python train.py --epochs 100 --batch_size 8 --lr 0.001 --patience 25
```

**Opsi Argumen Tambahan:**
- `--max_samples 100` : Membatasi jumlah data untuk pengujian cepat (smoke test).
- `--loss_type bce_dice` : Menggunakan kombinasi BCE + Dice Loss (default).
- `--loss_type dice` : Menggunakan pure Dice Loss sesuai formula paper.
- `--batch_size 8` : Ukuran batch (default: 8 sesuai paper).
- `--device cpu` atau `--device cuda` : Pilihan perangkat komputasi.

*Hasil pelatihan:* Bobot terbaik otomatis tersimpan di `saved_models/best_mobile_polypnet.pth`, riwayat epoch di `results/training_history.csv`, dan grafik kurva pembelajaran di `results/loss_dice_curve.png`.

---

### 3. Evaluasi Kuantitatif pada Test Set
Untuk menguji model terbaik pada 100 citra hold-out test set dan menghasilkan tabel komparasi metrik:
```bash
python evaluate.py --model_path saved_models/best_mobile_polypnet.pth --samples 6
```
Perintah ini mencetak tabel performa ke terminal, menyimpan metrik ke `results/test_metrics.csv`, dan menghasilkan gambar visualisasi komparasi di `results/test_visual_results.png`.

---

### 4. Segmentasi Citra Tunggal (Inference / Prediksi)
Untuk melakukan segmentasi pada citra polip baru:
```bash
python predict.py --image images/cju0qkwl35piu0993l0dewei2.jpg --output results/my_prediction.png
```
Output akan menghasilkan gambar komparasi side-by-side: `[Citra Asli | Mask Biner | Overlay Polip Berwarna]`.

---

## 📊 Ringkasan Komparasi Performa (Tabel 1 & 3 Paper)

| Model | Parameter | Ukuran Model | DICE (Kvasir) | mIoU | MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net** (MICCAI'15) | 7.85 M | 30 MB | 0.818 | 0.746 | 0.055 |
| **U-Net++** (TMI'19) | 9.04 M | 34.6 MB | 0.821 | 0.743 | 0.048 |
| **SFA** (MICCAI'19) | 25.59 M | 97.7 MB | 0.723 | 0.611 | 0.075 |
| **PraNet** (MICCAI'20) | 20.52 M | 78.4 MB | 0.898 | 0.840 | 0.030 |
| **Mobile-PolypNet (Ours)** | **226 K** | **~0.86 MB** | **0.935** | **0.888** | **0.031** |
