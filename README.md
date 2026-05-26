# SpaceAx AI - Conversational AI Engine

SpaceAx AI adalah mesin kecerdasan buatan (Conversational AI Engine) berbasis Transformer yang dirancang untuk dapat belajar secara mandiri, mengingat konteks percakapan, memiliki kepribadian/emosi, dan terintegrasi dengan Kamus Besar Bahasa Indonesia (KBBI) untuk meningkatkan kualitas bahasa.

Proyek ini dibuat dan dioptimasi sedemikian rupa sehingga memiliki arsitektur modular yang rapi, mulai dari *core model* hingga *personality engine* dan *web learning*.

---

## 🏗️ Struktur Proyek dan Modul

Program ini dibagi menjadi beberapa modul utama, masing-masing dengan fungsi spesifiknya:

### 1. `core/` (Inti Sistem)
Modul ini berisi dasar dari arsitektur model dan pemrosesan teks.
- **`config.py`**: Mengatur konfigurasi hyperparameter untuk berbagai ukuran model (small, medium, large, ultra) dan direktori penyimpanan data.
- **`tokenizer.py`**: Implementasi BPE (Byte-Pair Encoding) Tokenizer yang akan dilatih menggunakan dataset percakapan dan corpus KBBI.
- **`model.py`**: Implementasi arsitektur jaringan saraf (Neural Network) berbasis Transformer khusus untuk SpaceAx AI.
- **`kbbi.py`**: Integrasi dan pemrosesan dataset Kamus Besar Bahasa Indonesia (KBBI) untuk memperkaya kosa kata (vocabulary) dari AI.

### 2. `training/` (Pelatihan)
Modul yang bertanggung jawab atas proses pelatihan (training) model.
- **`dataset.py`**: Pemrosesan data teks mentah menjadi tensor PyTorch (Dataloader) yang siap dimasukkan ke dalam model.
- **`generate_seed_data.py`**: Menghasilkan data percakapan awal (seed data) secara otomatis untuk pelatihan tahap pertama.
- **`trainer.py`**: Mengatur *training loop*, evaluasi, penyimpanan *checkpoint* terbaik, dan pemuatan model untuk melanjutkan pelatihan (resume).

### 3. `memory/` (Memori & Konteks)
Modul untuk memberikan AI kemampuan mengingat percakapan sebelumnya dan memahami konteks.
- **`memory.py`**: Mengelola *short-term memory* (konteks percakapan aktif) dan *long-term memory*.
- **`vector_store.py`**: Implementasi penyimpanan memori berbasis vektor untuk pencarian kemiripan semantik (semantic search).

### 4. `learning/` (Pembelajaran Mandiri & Internet)
Modul yang memungkinkan AI memperluas pengetahuannya dengan mengakses informasi dari luar secara dinamis.
- **`internet.py`**: Menggunakan DuckDuckGo Search untuk mencari jawaban dari internet secara langsung (real-time) saat pengguna bertanya.
- **`web_learner.py`**: Mengambil (scraping) konten dari website atau artikel untuk dipelajari lebih dalam secara asinkronus.
- **`knowledge_base.py`**: Menyimpan dan menstrukturkan pengetahuan hasil belajar agar bisa diakses kembali.
- **`auto_trainer.py`**: Modul yang mengatur proses pelatihan otomatis (auto-retrain) dari data baru tanpa intervensi manual.

### 5. `personality/` (Kepribadian & Emosi)
Memberikan respons yang lebih natural, empatik, dan berkarakter.
- **`emotion_engine.py`**: Mesin emosi yang menentukan sentimen (misal: senang, sedih, netral) berdasarkan masukan pengguna dan mengubah gaya bicara model.
- **`personality.py`**: Mengatur perilaku dasar (behavior) dan prompt dasar dari AI.
- **`preferences.py`**: Menyimpan dan mempelajari preferensi pengguna (misal: gaya bahasa pengguna) selama percakapan berlangsung.

### 6. `main.py` & `chat.py`
- **`main.py`**: Entry point atau titik masuk utama program (CLI). Digunakan untuk menjalankan perintah seperti melatih, mengobrol, atau belajar.
- **`chat.py`**: Antarmuka interaktif CLI (*Command Line Interface*) untuk pengguna berbincang dengan AI.

---

## ⚙️ Persyaratan (Requirements)

Pastikan sistem Anda sudah terinstal Python 3.8 atau lebih baru. Pustaka yang dibutuhkan (berdasarkan `requirements.txt`):
- `torch >= 2.0.0`
- `beautifulsoup4 >= 4.11.0`
- `duckduckgo-search >= 5.0.0`
- `ddgs >= 9.14.0`
- `rich >= 13.0.0`
- `tokenizers` (via GitHub: `git+https://github.com/huggingface/tokenizers.git#subdirectory=bindings/python`)
- `requests >= 2.31.0`
- `numpy`

---

## 💻 Panduan Instalasi

### 🔹 Instalasi di Windows
1. **Instal Python**: Unduh Python (versi 3.8 - 3.11 disarankan) dari [python.org](https://www.python.org/downloads/windows/). Pastikan mencentang "Add Python to PATH" saat instalasi.
2. **Clone/Download Repository**: Buka Command Prompt / PowerShell, lalu masuk ke folder project.
   ```cmd
   git clone https://github.com/thomasalfareno/SpaceAX-AI/
   cd SpaceaxAiDebug
   ```
3. **Ekstrak Data KBBI**:
   Sebelum menjalankan program, Anda **wajib** mengekstrak file `kbbi_v_part.zip` yang ada di dalam folder `kbbi/`. Jika file tersebut tidak ada atau bermasalah saat di-*clone*, Anda dapat mengunduhnya secara manual [di sini](https://github.com/thomasalfareno/SpaceAX-AI/tree/main/kbbi/kbbi_v_part.zip).
4. **Buat Virtual Environment (Opsional namun disarankan)**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
5. **Instal Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   *(Catatan: Jika Anda memiliki GPU NVIDIA, instal PyTorch versi CUDA agar pelatihan lebih cepat mengikuti instruksi di [situs resmi PyTorch](https://pytorch.org/).)*

### 🔹 Instalasi di Linux (Contoh: Arch Linux)
1. **Update Sistem & Instal Python**:
   Arch Linux umumnya sudah memiliki versi Python terbaru.
   ```bash
   sudo pacman -Syu
   sudo pacman -S python python-pip git
   ```
2. **Clone/Download Repository**:
   ```bash
   git clone <url-repo-anda>
   cd SpaceaxAiDebug
   ```
3. **Ekstrak Data KBBI**:
   Sebelum menjalankan program, Anda **wajib** mengekstrak file `kbbi_v_part.zip` yang ada di dalam folder `kbbi/`. Jika file tersebut tidak ada atau bermasalah saat di-*clone*, Anda dapat mengunduhnya secara manual [di sini](https://github.com/thomasalfareno/SpaceAX-AI/tree/main/kbbi/kbbi_v_part.zip).
4. **Buat Virtual Environment (Sangat disarankan di Arch untuk menghindari konflik package sistem `PEP 668`)**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
5. **Instal Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Panduan Penggunaan

Gunakan `main.py` untuk mengakses seluruh fitur utama dari SpaceAx AI.

### 1. Melatih Model (Training)
Untuk melatih model dari awal atau melanjutkan pelatihan sebelumnya, jalankan:
```bash
python main.py train
```
**Opsi tambahan**:
- `--size`: Menentukan ukuran model (`small`, `medium`, `large`, `ultra`). Jika tidak diisi, akan menyesuaikan otomatis.
- `--epochs`: Mengubah jumlah *epochs* (putaran) training.
- `--regen`: Memaksa sistem untuk membuat ulang *seed data* dan tokenizer dari awal (hapus cache).

*Contoh:* `python main.py train --size medium --epochs 50`

### 2. Memulai Obrolan (Chat)
Setelah model dilatih minimal sekali, Anda bisa mulai mengobrol dengan AI:
```bash
python main.py chat
```
**Opsi tambahan**:
- `--mode chatdev`: Menjalankan dengan mode eksperimental "ChatDev".

### 3. Meminta AI Belajar Topik Baru (Learn)
Anda dapat menyuruh AI untuk mempelajari topik spesifik dari internet, yang kemudian akan disimpan di *knowledge base*:
```bash
python main.py learn "Kecerdasan Buatan"
```

### 4. Pelatihan Ulang (Retrain)
Setelah Anda melakukan beberapa sesi obrolan, percakapan Anda dengan AI akan tersimpan di dalam *log*. Anda dapat menggunakan data tersebut untuk memperbarui dan melatih ulang AI:
```bash
python main.py retrain
```

### 5. Menjalankan Tes Otomatis (Test)
Untuk memverifikasi apakah integrasi modul (pencarian internet, emosi, model dasar) berjalan dengan lancar tanpa error:
```bash
python main.py test
```

---

## 👨‍💻 Tentang Pengembang

**Thomas Alfareno Ananta Nugraha**
Program Studi Teknik Informatika, Departemen Teknik Informatika
Fakultas Teknologi Elektro dan Informatika Cerdas (FTEIC)
Institut Teknologi Sepuluh Nopember (ITS)

---
*Dokumentasi ini dihasilkan untuk mempermudah kontribusi dan implementasi ke environment lokal.*
