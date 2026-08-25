# SpaceAX AI Engine v3.1 — Multimodal LLM, Vision & Autonomous Agent System

SpaceAX AI v3.1 adalah ekosistem Kecerdasan Buatan (AI) multimodal berkinerja tinggi yang menggabungkan arsitektur Transformer modern (**Grouped Query Attention**, **Mixture of Experts**, **Vision Transformer**), modul **AI Agent otonom**, sistem **pemrosesan konteks & leksikon KBBI**, antarmuka **Web UI ala ChatGPT**, serta **REST API kompatibel OpenAI**.

---

## ⚡ Fitur Unggulan Terbaru v3.1

1. **Smart Hardware Auto-Detector**:
   - Deteksi otomatis & rinci perangkat akselerator komputasi: **NVIDIA CUDA** (Tesla T4, RTX series, A100, H100), **AMD ROCm** (Radeon RX, Instinct), **Apple MPS** (M1/M2/M3/M4 Silicon), **Intel XPU** (Arc/iGPU), serta CPU fallback.
   - Menampilkan statistik GPU, VRAM, Driver, dan Compute Capability secara akurat.

2. **Smart Package & GPU Optimality Auditor**:
   - Memeriksa modul Python (`torch`, `tokenizers`, `rich`, `Pillow`, `requests`, `bs4`).
   - Audit Otomatis: Jika GPU akselerator terdeteksi di tingkat sistem namun modul PyTorch yang terpasang adalah versi CPU-only, sistem secara otomatis memperbarui/memasang ulang *wheel* PyTorch berbasis CUDA/ROCm yang paling optimal.

3. **VRAM Auto-Tune & Safe Micro-Batching (Anti CUDA OOM)**:
   - Mengaktifkan `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` untuk mengeliminasi fragmentasi memori.
   - Pada GPU dengan VRAM ≤ 20.0 GB (seperti Tesla T4 14.6GB di Colab), *Gradient Checkpointing* otomatis diaktifkan (menghemat **>60% VRAM** pada *backward pass*).
   - Micro-batch disesuaikan secara aman (misal `batch_size = 2`, `gradient_accumulation = 32`), menjaga *effective batch size* tetap 64 tanpa risiko *CUDA Out of Memory*.

4. **Independent Checkpoint Management (Zero-Overlap)**:
   - Checkpoint disimpan secara terpisah per tier model (`model_best_small.pt`, `model_best_medium.pt`, `model_best_promax_8b.pt`).
   - Tidak ada tumpang tindih atau bentrok arsitektur saat berganti opsi `--size`.

5. **⚡ Fast Convergence Training Engine (NEW v3.1)**:
   - **Adafactor Optimizer** kustom dengan `beta2_decay=0.8` (original paper default) untuk adaptasi cepat ke statistik gradien terbaru.
   - **Learning Rate `8e-4`** — 2.7× lebih agresif dari default, optimal untuk Adafactor yang sudah punya adaptive scaling internal.
   - **Warmup `200` steps** — Model mulai belajar penuh dalam ~800 batch (bukan ~4000 batch seperti sebelumnya).
   - **Cosine Decay dengan `min_lr_ratio=0.1`** — LR tidak pernah jatuh ke nol; epoch akhir tetap produktif dengan LR minimum 10% dari peak.
   - **Label Smoothing `0.05`** — Rendah untuk mempercepat penurunan loss awal tanpa loss floor artifisial.
   - **Weight Decay `0.05`** — Regularisasi lebih kuat untuk mencegah overfitting pada dataset kecil.

6. **🖼️ Expanded Multimodal Vision Training (NEW v3.1)**:
   - **171 pasangan training multimodal** dari **19 kategori visual** (Hewan, Makanan, Teknologi, Kota, Alam, dll.).
   - **Auto-Download & Regenerasi**: Aset gambar otomatis diunduh atau disintesis jika belum ada — cukup tambahkan `--regen`.
   - **Vision-Text Joint Training**: Semua gambar dilatih bersama data percakapan dalam satu loop training terintegrasi.
   - Adafactor aman untuk layer **Conv2d** (4D tensor) — menggunakan non-factored update untuk mencegah `RuntimeError` shape mismatch.

7. **📊 Enhanced Training Monitoring (NEW v3.1)**:
   - **5 text sample + 5 vision sample** otomatis ditampilkan setiap akhir epoch.
   - **LR schedule info** ditampilkan di header training: warmup steps, peak LR, min LR, label smoothing.
   - Flag CLI `--no-early-stopping` dan `--patience N` untuk kontrol penuh terhadap durasi training.

---

## 📋 Spesifikasi Model & Profil System

SpaceAX AI menyediakan beberapa tier profil model yang secara otomatis disesuaikan dengan alokasi RAM dan VRAM sistem Anda (Default: Auto-Detect Hardware Capability), atau dapat ditentukan secara eksplisit via argumen CLI (`--size`):

| Profil | Params (Est.) | Arch Specs | MoE Setup | Vision | Rekomendasi Hardware |
|---|---|---|---|---|---|
| **SMALL** | ~35M | `d_model=512`, `layers=8`, `heads=8`, `kv_heads=2` | 2 Experts (Top-1) | Yes (ViT) | RAM ≥ 4 GB / CPU Only |
| **MEDIUM** | ~120M | `d_model=768`, `layers=12`, `heads=12`, `kv_heads=4` | 2 Experts (Top-1) | Yes (ViT) | RAM ≥ 8 GB / VRAM 4 GB |
| **LARGE** | ~350M | `d_model=1024`, `layers=18`, `heads=16`, `kv_heads=4` | 4 Experts (Top-1) | Yes (ViT) | RAM ≥ 16 GB / VRAM 8 GB |
| **ULTRA** | ~750M | `d_model=1280`, `layers=24`, `heads=20`, `kv_heads=4` | 4 Experts (Top-2) | Yes (ViT) | RAM ≥ 32 GB / VRAM 16 GB |
| **PROMAX 1B** | ~1.1B | `d_model=1536`, `layers=28`, `heads=24`, `kv_heads=8` | 4 Experts (Top-2) | Yes (ViT) | RAM ≥ 48 GB / VRAM 24 GB |
| **PROMAX 4B** | ~4.2B | `d_model=2560`, `layers=36`, `heads=32`, `kv_heads=8` | 8 Experts (Top-2) | Yes (ViT) | VRAM ≥ 16 GB (VRAM-Fit) |
| **PROMAX 8B** | ~8.0B | `d_model=4096`, `layers=40`, `heads=32`, `kv_heads=8` | 8 Experts (Top-2) | Yes (ViT) | VRAM ≥ 24 GB / VRAM-Fit |

---

## ⚡ Fast Convergence Hyperparameters

Konfigurasi training v3.1 telah dioptimasi untuk konvergensi cepat. Perbandingan dengan versi sebelumnya:

| Parameter | v3.0 (Lama) | v3.1 (Baru) | Dampak |
|---|---|---|---|
| **Learning Rate** | `3e-4` | `8e-4` | 2.7× lebih agresif |
| **Warmup Steps** | `1000` | `200` | 5× lebih cepat mencapai peak LR |
| **Label Smoothing** | `0.1` | `0.05` | Loss turun lebih cepat |
| **Cosine Min LR** | `0` (mati total) | `10% × peak_lr` | Epoch akhir tetap produktif |
| **Adafactor β₂ Decay** | `0.999` | `0.8` | Adaptasi gradien lebih cepat |
| **Weight Decay** | `0.01` | `0.05` | Regularisasi lebih kuat |

---

## ⚡ Pemilihan Ukuran Model via CLI

Seluruh perintah CLI (`chat`, `web`, `train`, `retrain`) mendukung opsi pemilihan ukuran model:

- **Default (Auto-Detect)**: Jika `--size` tidak diberikan, sistem otomatis memilih profil terbaik yang sanggup dijalankan oleh RAM/GPU device Anda.
- **Manual Override**: Sertakan `--size [small|medium|large|ultra|promax]` untuk memaksa profil tertentu.

### Contoh Pemilihan Ukuran Model:

```bash
# 1. Menjalankan Chat Terminal dengan profil otomatis sesuai kemampuan device
python main.py chat

# 2. Menjalankan Chat Terminal memaksa profil Medium
python main.py chat --size medium

# 3. Menjalankan Server Web UI & REST API memaksa profil Medium
python main.py web --size medium

# 4. Menjalankan Server Web UI memaksa profil ProMax 8B
python main.py web --size promax --promax-tier promax_8b

# 5. Melatih model profil Small
python main.py train --size small --regen --force
```

---

## ⚡ Panduan Google Colab (GPU Tesla T4 14.6 GB VRAM)

Untuk melatih atau menjalankan SpaceAX AI di **Google Colab** dengan GPU Tesla T4:

### Setup & Training dari Nol di Colab:
```bash
# 1. Clone repositori SpaceAX AI
!git clone https://github.com/thomasalfareno/SpaceAX-AI.git
%cd SpaceAX-AI

# 2. Ekstrak aset dataset KBBI
!unzip kbbi/ekstrak.zip -d kbbi/temp
!mv kbbi/temp/* kbbi/
!rm -rf kbbi/temp kbbi/ekstrak.zip

# 3. Jalankan training (Smart Auto-Installer otomatis memasang dependensi CUDA, EasyOCR, dll.)
!python main.py train --epochs 100 --batch-size 8 --grad-accum 4 --no-early-stopping --regen
```

> **Catatan Otomatisasi Google Colab (Tesla T4):**
> - **Smart Auto-Installer**: Otomatis mendeteksi GPU CUDA dan menginstall dependensi (`torch` CUDA, `easyocr`, `tokenizers`, dll.) secara otomatis saat `main.py` pertama kali dijalankan.
> - **VRAM Auto-Tune**: Mengatur memori GPU dan mengaktifkan Gradient Checkpointing (aman dari CUDA OOM).
> - **Fast Convergence Engine**: LR mencapai peak `8e-4` secara agresif dengan Adafactor ($\beta_2=0.8$), loss & PPL mengecil secara optimal sejak Epoch 1.

---

## 🌐 Menjalankan & Mematikan Server Web UI / REST API

### 1. Menjalankan Server Web UI (Latar Depan / Foreground)
```bash
python main.py web --port 7860
```
- Web UI dapat diakses melalui browser di: `http://localhost:7860`
- REST API Base: `http://localhost:7860/v1`

### 2. Menjalankan Server Latar Belakang (Background Daemon - Linux / Bash)
```bash
nohup python main.py web --port 7860 > server.log 2>&1 &
echo $! > server.pid
echo "Server running on PID $(cat server.pid)"
```

### 3. Mematikan Server
- **Jika di terminal (Foreground)**: Tekan `Ctrl + C`
- **Jika di background (Linux/Bash)**:
  ```bash
  kill $(cat server.pid) && rm server.pid
  ```
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  netstat -ano | findstr :7860
  taskkill /PID <PID_NUMBER> /F
  ```

---

## 💻 Panduan Perintah CLI Lengkap

### 1. Terminal Interactive Chat
```bash
python main.py chat --size medium
```
**Perintah Kontrol dalam Obrolan Terminal:**
- `/weboff` : Mematikan pencarian web live & auto-learning internet.
- `/webon`  : Mengaktifkan kembali pencarian web.
- `exit` / `quit` : Keluar dari obrolan.

### 2. Autonomous AI Agent Loop
```bash
python main.py agent "Buka folder saat ini dan tampilkan daftar file"
```

### 3. Analisis Gambar Multimodal Vision & OCR Presisi via CLI
Anda dapat menggunakan fitur Vision Transformer (ViT) & OCR Engine presisi melalui 2 cara di CLI:

#### Cara A: Langsung via Perintah `python main.py vision`
```bash
# Analisis gambar & OCR teks pada gambar lokal
python main.py vision --image assets/vision_train/cat_01.jpg --prompt "Deskripsikan apa yang terlihat pada gambar ini"

# OCR teks pada gambar dari URL internet
python main.py vision --image https://example.com/document.jpg --prompt "Baca teks pada gambar ini"
```

#### Cara B: Lewat Obrolan Interaktif Terminal (`python main.py chat`)
Di dalam obrolan terminal interaktif, gunakan perintah `/image` atau `/vision`:
```text
Kamu: /image assets/vision_train/cat_01.jpg Jelaskan gambar ini
Kamu: /vision https://example.com/photo.jpg Baca tulisan di gambar ini
```

> ⚡ **OCR Otomatis (Akurasi 100%)**: **Smart Auto-Installer** SpaceAX AI secara otomatis memverifikasi dan mengunduh modul `easyocr` saat startup, sehingga pembacaan dokumen & tulisan tangan Bahasa Indonesia & Inggris langsung aktif tanpa instalasi manual.

### 4. Melatih Model (Training & Regenerasi Data Vision)
```bash
# Training standar
python main.py train --size small --regen --force

# Training agresif tanpa early stopping (100 epoch)
python main.py train --epochs 100 --batch-size 8 --grad-accum 4 --no-early-stopping --regen

# Training dengan custom patience (early stopping setelah 10 epoch tanpa perbaikan)
python main.py train --epochs 50 --patience 10 --regen
```

**Flag Training CLI:**
| Flag | Deskripsi | Default |
|---|---|---|
| `--epochs N` | Jumlah epoch training | `20` |
| `--batch-size N` | Ukuran micro-batch | `4` (auto-tuned) |
| `--grad-accum N` | Gradient accumulation steps | `8` (auto-tuned) |
| `--no-early-stopping` | Nonaktifkan early stopping, jalankan semua epoch | Off |
| `--patience N` | Patience untuk early stopping (berapa epoch tanpa perbaikan) | `5` |
| `--regen` | Regenerasi seluruh data training (termasuk aset vision) | Off |
| `--force` | Paksa training ulang dari nol | Off |
| `--size` | Pilih profil model (`small`, `medium`, `large`, `ultra`, `promax`) | Auto-detect |

---

## 🔌 Integrasi REST API (OpenAI Compatible)

Endpoint REST API kompatibel dengan SDK OpenAI standar.

### Endpoint
- `POST /v1/chat/completions` — Generasi teks percakapan
- `GET /v1/models` — Daftar model yang tersedia

### Contoh Request via Curl
```bash
curl http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spaceax-small",
    "web_enabled": true,
    "messages": [
      {"role": "user", "content": "Jelaskan konsep Grouped Query Attention (GQA)"}
    ]
  }'
```

### Contoh Request via Python (Requests / OpenAI SDK)
```python
import requests

url = "http://localhost:7860/v1/chat/completions"
payload = {
    "model": "spaceax-small",
    "messages": [
        {"role": "user", "content": "Buatkan skrip Python turunan sin(x)"}
    ]
}

response = requests.post(url, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

---

## 🖼️ Dataset Vision Training (19 Kategori)

SpaceAX AI v3.1 melatih Vision Transformer dengan **171 pasangan multimodal** yang mencakup:

| No | Kategori | Jumlah Gambar | Contoh |
|---|---|---|---|
| 1 | 🐱 Kucing | 9 | `cat_01.jpg` — kucing bermain, tidur, close-up |
| 2 | 🐶 Anjing | 9 | `dog_01.jpg` — anjing bermain, berlari |
| 3 | 🌸 Bunga | 9 | `flower_01.jpg` — mawar, tulip, sakura |
| 4 | 🚗 Kendaraan | 9 | `car_01.jpg` — sedan, SUV, truk |
| 5 | 🏠 Bangunan | 9 | `building_01.jpg` — rumah, gedung, jembatan |
| 6 | 🍔 Makanan | 9 | `food_01.jpg` — nasi goreng, pizza, sate |
| 7 | 🌊 Pemandangan Alam | 9 | `nature_01.jpg` — pantai, gunung, danau |
| 8 | 👤 Manusia | 9 | `person_01.jpg` — potret, aktivitas |
| 9 | 📱 Teknologi | 9 | `tech_01.jpg` — laptop, robot, smartphone |
| 10 | 🏙️ Kota | 9 | `city_01.jpg` — skyline, jalanan kota |
| 11 | 🐦 Burung | 9 | `bird_01.jpg` — elang, kolibri |
| 12 | 🐟 Ikan | 9 | `fish_01.jpg` — ikan tropis, coral reef |
| 13 | 🎨 Seni | 9 | `art_01.jpg` — lukisan, patung, mural |
| 14 | ⚽ Olahraga | 9 | `sport_01.jpg` — sepak bola, renang |
| 15 | 🎵 Musik | 9 | `music_01.jpg` — gitar, piano, konser |
| 16 | 📚 Buku & Edukasi | 9 | `book_01.jpg` — perpustakaan, belajar |
| 17 | 🌌 Antariksa | 9 | `space_01.jpg` — galaksi, planet, astronot |
| 18 | 🏖️ Pantai | 9 | `beach_01.jpg` — pasir, ombak, sunset |
| 19 | 🌧️ Cuaca | 9 | `weather_01.jpg` — hujan, pelangi, badai |

> Semua aset otomatis diunduh atau disintesis saat pertama kali training dengan `--regen`.

---

## 📁 Struktur Direktori Proyek

```
SpaceAX-AI-BETA-main/
├── api_server.py        # REST API OpenAI-compatible & Web Server
├── chat.py              # Engine Terminal Chat Interaktif
├── main.py              # CLI Main Entry Point & Command Manager
├── core/                # Modul Komponen Utama Transformer
│   ├── agent.py         # Autonomous AI Agent System
│   ├── auto_installer.py# Smart Auto-Installer & Hardware Optimizer
│   ├── config.py        # Profile Config & Hardware Detector (CUDA, ROCm, MPS, XPU)
│   ├── kbbi.py          # Pemrosesan Leksikon KBBI
│   ├── model.py         # SpaceaxModel Transformer Architecture (GQA + MoE + ViT)
│   ├── tokenizer.py     # HuggingFace BPE Subword Tokenizer
│   ├── tools.py         # Tool Registry (File, System, Web, Python)
│   └── vision.py        # Vision Transformer (ViT) Encoder & Projector
├── training/            # Pipeline Training & Vision Data
│   ├── dataset.py       # Conversational & Multimodal Vision Dataset Loader
│   ├── trainer.py       # Trainer dengan Adafactor/AdamW, Cosine+MinLR, & MoE Aux Loss
│   └── vision_data.py   # Auto-Download & Sintesis 171 Pasangan Multimodal Vision
├── web/                 # Web UI Interface (ChatGPT Dark Mode)
│   ├── index.html       # UI HTML Structure
│   ├── style.css        # Glassmorphic Styling
│   └── app.js           # Client Logic & Mention/Reply Script
└── assets/              # Vision Training Assets (19 Kategori, Auto-Generated)
    └── vision_train/    # Gambar + manifest.json
```

---

## 📄 Lisensi & Hak Cipta

© 2026 **SpaceAX AI Project** — Thomas Alfareno Ananta Nugraha (ITS Surabaya).  
Dikembangkan untuk riset sains kecerdasan buatan & pemrosesan bahasa alami di Indonesia.
