"""
Tambahan seed percakapan SpaceAx — topik & variasi ekstra.
Digabung oleh generate_seed_data.generate_all().
"""

import random
from training.generate_seed_data import _entry, _id


def gen_bahasa_formal_santai():
    convs = []
    formal = [
        ("Selamat datang, silakan sampaikan kebutuhan Anda.", "neutral"),
        ("Terima kasih atas pertanyaannya.", "trust"),
        ("Mohon maaf atas ketidaknyamanan ini.", "sadness"),
        ("Saya akan membantu sebaik mungkin.", "trust"),
    ]
    santai = [
        ("Oke siap, ceritain aja!", "joy"),
        ("Wah keren tuh, lanjut dong!", "joy"),
        ("Gapapa kok, santai aja.", "trust"),
        ("Yaudah kita bahas pelan-pelan ya.", "neutral"),
    ]
    prompts_f = [
        "tolong jawab dengan formal",
        "pakai bahasa formal",
        "jawab sopan dong",
        "bisa bahasa baku?",
    ]
    prompts_s = [
        "jawab santai aja",
        "ngomongnya gaul dong",
        "bahasa sehari-hari aja",
        "jangan terlalu kaku",
    ]
    for q in prompts_f:
        r = random.choice(formal)
        convs.append(_entry(q, f"<pikir>Menyesuaikan gaya formal...</pikir>{r[0]}", r[1], "bahasa_gaya"))
    for q in prompts_s:
        r = random.choice(santai)
        convs.append(_entry(q, f"<pikir>Menyesuaikan gaya santai...</pikir>{r[0]}", r[1], "bahasa_gaya"))

    pasangan = [
        ("gimana cara bilang terima kasih dalam bahasa inggris", "Thank you / Thanks — kalau lebih sopan: Thank you very much."),
        ("apa bedanya 'akan' dan 'sedang'", "'Akan' menunjuk rencana masa depan; 'sedang' menunjuk aktivitas yang berlangsung sekarang."),
        ("apa itu kata baku", "Kata baku dipakai dalam tulisan resmi; kata tidak baku lebih santai di percakapan."),
        ("contoh kalimat efektif", "Kalimat efektif: subjek jelas, predikat tepat, hindari mubazir. Contoh: 'Tim menyelesaikan proyek tepat waktu.'"),
    ]
    for q, a in pasangan:
        convs.append(_entry(q, f"<pikir>Menjelaskan bahasa Indonesia...</pikir>{a}", "neutral", "bahasa_gaya"))
    return convs


def gen_pendidikan():
    convs = []
    topik = [
        ("tips belajar efektif", "Pomodoro 25 menit, ulang materi, aktifkan dengan latihan soal, tidur cukup.", "trust"),
        ("cara nyiapin ujian", "Buat jadwal, kerjakan bank soal, review kesalahan, jangan begadang semalam.", "anticipation"),
        ("apa itu skripsi", "Skripsi = tugas akhir penelitian mahasiswa S1; ada proposal, metode, hasil, dan sidang.", "neutral"),
        ("bedanya S1 S2 S3", "S1 sarjana, S2 magister, S3 doktor — jenjang pendidikan tinggi.", "neutral"),
        ("apa itu IPK", "IPK = Indeks Prestasi Kumulatif, rata-rata nilai semester tertimbang SKS.", "neutral"),
        ("jurusan informatika belajar apa", "Algoritma, struktur data, basis data, jaringan, AI, software engineering, dan proyek.", "joy"),
        ("kenapa harus ngoding", "Ngoding melatih logika, problem solving, dan membuka peluang karier di teknologi.", "trust"),
    ]
    for q, a, emo in topik:
        convs.append(_entry(q, f"<pikir>Menjawab topik pendidikan...</pikir>{a}", emo, "pendidikan"))

    kampus = ["ITS", "UI", "UGM", "ITB", "UNAIR", "UNDIP"]
    for uni in kampus:
        q = f"tentang kampus {uni}"
        r = (
            f"{uni} adalah perguruan tinggi di Indonesia. "
            f"Kalau kamu mahasiswa {uni}, semangat kuliahnya! "
            f"Aku sendiri terkait erat dengan { _id()['university'] }."
        )
        convs.append(_entry(q, f"<pikir>Membahas kampus...</pikir>{r}", "neutral", "pendidikan"))
    return convs


def gen_teknologi():
    convs = []
    items = [
        ("apa itu machine learning", "ML = mesin belajar pola dari data tanpa diatur aturan manual satu per satu.", "neutral"),
        ("apa itu deep learning", "Deep learning memakai neural network banyak lapis untuk gambar, teks, suara.", "neutral"),
        ("apa itu transformer", "Transformer memakai mekanisme attention untuk memproses urutan token — dasar banyak LLM modern.", "neutral"),
        ("apa itu GPU", "GPU memproses banyak operasi paralel — cocok untuk training neural network.", "neutral"),
        ("apa itu RAM", "RAM = memori kerja sementara; semakin besar semakin banyak data/model yang muat.", "neutral"),
        ("apa itu API REST", "REST API menukar data via HTTP (GET baca, POST kirim, dll.) biasanya format JSON.", "neutral"),
        ("apa itu docker", "Docker membungkus aplikasi + dependensi dalam container agar jalan konsisten di mana saja.", "neutral"),
        ("apa itu linux", "Linux = kernel OS open source; banyak server dan developer memakainya.", "neutral"),
        ("apa itu cybersecurity", "Keamanan siber melindungi sistem dari peretasan, malware, dan kebocoran data.", "fear"),
        ("apa itu blockchain", "Blockchain = rantai blok data terenkripsi; dikenal dari cryptocurrency tapi ada use case lain.", "neutral"),
    ]
    for q, a, emo in items:
        convs.append(_entry(q, f"<pikir>Menjelaskan teknologi...</pikir>{a}", emo, "teknologi"))

    for brand in ["android", "iphone", "windows", "macos", "vscode", "pycharm"]:
        q = random.choice([f"apa itu {brand}", f"kenapa pakai {brand}", f"tips {brand}"])
        r = f"{brand.capitalize()} adalah produk/platform populer di ekosistem teknologi. Mau penjelasan spesifik fitur atau perbandingan?"
        convs.append(_entry(q, f"<pikir>Membahas produk tech...</pikir>{r}", "neutral", "teknologi"))
    return convs


def gen_spaceax_bantuan():
    info = _id()
    convs = []
    cmds = [
        (
            "cara pakai spaceax",
            f"<pikir>Menjelaskan cara pakai SpaceAx...</pikir>"
            f"1) Latih: `python main.py train`\n"
            f"2) Chat: `python main.py chat`\n"
            f"3) Cari: ketik `!search topik` di chat\n"
            f"4) Retrain: `python main.py retrain` setelah banyak ngobrol\n"
            f"SpaceAx AI oleh {info['developer']} — {info['university']}.",
        ),
        (
            "apa itu mode chatdev",
            "<pikir>Menjelaskan chatdev...</pikir>Mode chatdev untuk debugging & introspeksi training — "
            "jalankan `python main.py chat --mode chatdev`.",
        ),
        (
            "apa itu promax",
            "<pikir>Menjelaskan ProMax...</pikir>ProMax = skala model besar SpaceAx (tier 1B/4B/8B). "
            "Pilih dengan `--promax-tier` saat train/chat.",
        ),
        (
            "apa itu --force training",
            "<pikir>Menjelaskan flag force...</pikir>`--force` memaksa tier walau RAM kurang "
            "dan menonaktifkan early stopping agar semua epoch selesai.",
        ),
        (
            "dimana file checkpoint",
            "<pikir>Menunjukkan path checkpoint...</pikir>Checkpoint terbaik: `data/checkpoints/model_best.pt`. "
            "Per epoch: `data/checkpoints/model_epoch_N.pt`.",
        ),
        (
            "kenapa chat masih template",
            "<pikir>Menjelaskan val_loss...</pikir>Checkpoint mungkin masih awal (val_loss tinggi). "
            "Lanjutkan training ≥15–30 epoch ProMax; chat akan lebih sering generate dari model.",
        ),
    ]
    for q, r in cmds:
        convs.append(_entry(q, r, "trust", "spaceax_bantuan"))

    for tier in ["promax_1b", "promax_4b", "promax_8b"]:
        q = f"apa bedanya {tier}"
        r = (
            f"<pikir>Menjelaskan tier...</pikir>{tier} = sub-profil ProMax dengan ukuran parameter & vocab berbeda. "
            f"Tier lebih besar butuh RAM/VRAM lebih banyak; checkpoint antar tier tidak interchangeable."
        )
        convs.append(_entry(q, r, "neutral", "spaceax_bantuan"))
    return convs


def gen_kesehatan_motivasi():
    convs = []
    pairs = [
        ("aku capek banget", "Istirahat dulu ya — minum air, tarik napas, boleh pause. Aku tunggu kalau kamu siap lanjut.", "sadness"),
        ("aku stress", "Stress itu wajar. Coba pecah masalah jadi langkah kecil. Mau cerita pemicunya?", "fear"),
        ("aku tidak percaya diri", "Kamu sudah berani mulai — itu langkah besar. Fokus pada progres kecil hari ini.", "trust"),
        ("tips tidur nyenyak", "Jadwal tidur tetap, kurangi layar sebelum tidur, kamar gelap & adem.", "neutral"),
        ("olahraga ringan apa", "Jalan kaki 20 menit, stretching, atau yoga ringan sudah membantu mood.", "joy"),
        ("aku burnout", "Burnout butuh recovery. Kurangi beban, delegasi kalau bisa, dan bicara ke orang tepercaya.", "sadness"),
    ]
    for q, r, emo in pairs:
        convs.append(_entry(q, f"<pikir>Merespons dengan empati...</pikir>{r}", emo, "kesehatan"))

    for _ in range(80):
        n = random.randint(5, 12)
        q = f"tips produktif {n} jam"
        r = (
            f"Blok waktu {n} jam: 1) 10 menit rencana, 2) fokus tanpa HP, "
            f"3) istirahat 5 menit tiap jam, 4) review hasil di akhir."
        )
        convs.append(_entry(q, f"<pikir>Memberi tips produktivitas...</pikir>{r}", "anticipation", "kesehatan"))
    return convs


def gen_followup_konteks():
    convs = []
    after_ask_wellbeing = [
        "kabar ku baik",
        "kabar saya baik",
        "alhamdulillah baik",
        "puji tuhan baik",
        "baik kok",
        "lumayan",
        "biasa aja",
    ]
    replies = [
        "Senang dengarnya! Ada yang mau dibahas hari ini?",
        "Syukurlah! Mau lanjut topik apa?",
        "Mantap! Ceritain aktivitasmu dong.",
    ]
    for u in after_ask_wellbeing:
        convs.append(
            _entry(
                u,
                f"<pikir>Mengaitkan dengan tanya kabar sebelumnya...</pikir>{random.choice(replies)}",
                "joy",
                "followup_konteks",
                pref={"last_ai_asked_wellbeing": True},
            )
        )

    clarifications = [
        ("maksudnya gimana", "Maksudku: bisa dijelaskan lebih spesifik bagian mana yang membingungkan?"),
        ("bisa diulang", "Tentu! Aku rangkum lagi dengan bahasa lebih sederhana."),
        ("kurang paham", "Oke, aku pecah jadi langkah-langkah kecil ya."),
        ("lanjut", "Siap, kita lanjut. Topik sebelumnya mau dideep atau ada hal baru?"),
        ("terus gimana", "Nah, langkah berikutnya biasanya begini..."),
    ]
    for q, r in clarifications:
        convs.append(_entry(q, f"<pikir>Menangani follow-up...</pikir>{r}", "neutral", "followup_konteks"))
    return convs


def gen_sains_umum():
    convs = []
    facts = [
        ("berapa planet di tata surya", "Delapan planet: Merkurius, Venus, Bumi, Mars, Jupiter, Saturnus, Uranus, Neptunus."),
        ("apa itu fotosintesis", "Proses tumbuhan mengubah cahaya + CO₂ + air menjadi glukosa dan O₂."),
        ("apa itu gravitasi", "Gaya tarik massa; di Bumi membuat benda jatuh ke bawah."),
        ("apa itu atom", "Unit dasar materi: inti (proton, neutron) + elektron mengelilingi."),
        ("kenapa langit biru", "Hamburan Rayleigh — cahaya biru lebih tersebar di atmosfer."),
        ("apa itu DNA", "Molekul penyimpan informasi genetik makhluk hidup."),
        ("apa itu energi terbarukan", "Sumber energi yang diisi ulang: surya, angin, air, panas bumi."),
    ]
    for q, a in facts:
        convs.append(_entry(q, f"<pikir>Menjelaskan sains...</pikir>{a}", "neutral", "sains"))

    for _ in range(350):
        a, b = random.randint(1, 50), random.randint(1, 50)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        else:
            res = a * b
        q = random.choice([f"{a} {op} {b} berapa", f"hitung {a}{op}{b}", f"berapa hasil {a} {op} {b}"])
        r = f"<pikir>Menghitung {a} {op} {b} = {res}</pikir>Hasilnya **{res}**."
        convs.append(_entry(q, r, "neutral", "sains_hitung"))
    return convs


def gen_indonesia_budaya():
    convs = []
    items = [
        ("apa itu batik", "Batik = kain bermotif tradisional Indonesia, warisan budaya UNESCO."),
        ("apa itu wayang", "Wayang = seni pertunjukan boneka kulit/panggung, cerita biasanya Ramayana/Mahabharata."),
        ("ibu kota indonesia", "Ibu kota Indonesia adalah Nusantara (IKN); Jakarta tetap pusat ekonomi besar."),
        ("berapa provinsi di indonesia", "Indonesia memiliki 38 provinsi (termasuk pemekaran terbaru)."),
        ("apa bahasa resmi indonesia", "Bahasa resmi: Bahasa Indonesia. Daerah punya bahasa daerah masing-masing."),
        ("hari kemerdekaan indonesia", "17 Agustus 1945 — Hari Kemerdekaan Republik Indonesia."),
    ]
    for q, a in items:
        convs.append(_entry(q, f"<pikir>Membahas budaya Indonesia...</pikir>{a}", "joy", "budaya_id"))

    makanan = [
        "rendang", "gudeg", "rawon", "pempek", "soto", "gado-gado", "tempe", "tahu",
        "es campur", "klepon", "serabi", "martabak",
    ]
    for m in makanan:
        q = random.choice([f"ceritain tentang {m}", f"{m} itu apa", f"asal {m} dari mana"])
        r = f"{m.capitalize()} adalah kuliner khas Indonesia yang populer. Rasanya unik dan jadi kebanggaan lokal!"
        convs.append(_entry(q, f"<pikir>Membahas kuliner...</pikir>{r}", "joy", "budaya_id"))
    return convs


def gen_coding_massive():
    """Ratusan variasi pertanyaan coding ringan."""
    convs = []
    topics = [
        ("list", "List Python: `buah = ['apel','jeruk']`, akses `buah[0]`, tambah `buah.append('mangga')`."),
        ("dict", "Dictionary: `data = {'nama':'Budi','umur':20}` — akses `data['nama']`."),
        ("tuple", "Tuple immutable: `titik = (10, 20)` — cocok untuk koordinat tetap."),
        ("set", "Set menghilangkan duplikat: `set([1,1,2])` → `{1,2}`."),
        ("if else", "Kondisi: `if skor >= 75: print('lulus') else: print('ulang')`."),
        ("for loop", "For loop: `for i in range(5): print(i)` mencetak 0..4."),
        ("while", "While: `n=0; while n<3: print(n); n+=1`."),
        ("function", "Fungsi: `def tambah(a,b): return a+b`."),
        ("class", "Class OOP: atribut di `__init__`, method untuk perilaku objek."),
        ("import", "Import modul: `import math` lalu `math.sqrt(16)`."),
        ("numpy", "NumPy array: operasi vektor lebih cepat dari list Python murni."),
        ("pandas", "Pandas DataFrame seperti tabel Excel di Python."),
        ("matplotlib", "Matplotlib untuk plot grafik: line, bar, scatter."),
        ("json", "JSON: `json.loads()` baca string, `json.dumps()` ke string."),
        ("requests", "Requests HTTP: `requests.get(url)` ambil halaman web."),
        ("venv", "Virtual env mengisolasi dependensi proyek Python."),
        ("pip", "pip install paket dari PyPI: `pip install torch`."),
        ("git commit", "Git: `git add .` lalu `git commit -m 'pesan'`."),
        ("async", "async/await untuk I/O non-blocking di Python 3."),
        ("decorator", "Decorator membungkus fungsi untuk logging, auth, dll."),
    ]
    prefixes = [
        "apa itu",
        "jelaskan",
        "contoh",
        "buat contoh",
        "kasih contoh",
        "gimana cara pakai",
        "bedanya",
    ]
    for topic, answer in topics:
        for pref in prefixes:
            q = f"{pref} {topic} python" if "python" not in pref else f"{pref} {topic}"
            convs.append(
                _entry(q, f"<pikir>Menjelaskan {topic}...</pikir>{answer}", "neutral", "coding_massive")
            )

    errors = [
        ("IndentationError", "Samakan indent 4 spasi setelah if/for/def."),
        ("SyntaxError", "Cek tanda kurung, kutip, dan titik dua yang hilang."),
        ("KeyError", "Key tidak ada di dict — pakai `.get(key, default)`."),
        ("IndexError", "Indeks di luar range list — cek `len(list)`."),
        ("AttributeError", "Objek tidak punya atribut/method — cek tipe & nama."),
    ]
    for err, fix in errors:
        for pref in ["kenapa", "cara fix", "solusi"]:
            q = f"{pref} {err} python"
            convs.append(_entry(q, f"<pikir>Mendiagnosis {err}...</pikir>{fix}", "neutral", "coding_massive"))

    for _ in range(200):
        x = random.randint(1, 100)
        q = random.choice([f"print bilangan {x} python", f"kode python cetak {x}"])
        r = f"<pikir>Contoh print...</pikir>```python\nprint({x})\n```"
        convs.append(_entry(q, r, "neutral", "coding_massive"))
    return convs


def gen_greeting_massive():
    convs = []
    opens = ["halo", "hai", "hey", "pagi", "siang", "sore", "malam", "assalamualaikum"]
    tails = ["", " kak", " bang", " mas", " mbak", " bro", " gan"]
    replies = [
        "Halo! Senang ketemu kamu — mau ngobrol apa?",
        "Hai! Aku SpaceAx AI, siap bantu.",
        "Hey! Ceritain aja topiknya.",
    ]
    for o in opens:
        for t in tails:
            q = (o + t).strip()
            r = random.choice(replies)
            convs.append(_entry(q, f"<pikir>Menyapa...</pikir>{r}", "joy", "greeting_massive"))
    return convs


def gen_emosi_massive():
    convs = []
    stems = ["sedih", "marah", "takut", "cemas", "bahagia", "lega", "kecewa", "bangga"]
    modifiers = ["banget", "sedikit", "hari ini", "karena kerjaan", "karena ujian"]
    responses = {
        "sedih": "Aku dengerin. Mau cerita pelan-pelan?",
        "marah": "Wajar marah. Tarik napas dulu — mau venting?",
        "takut": "Takut itu normal. Kita pecah masalahnya jadi kecil ya.",
        "cemas": "Coba fokus satu hal yang bisa dikontrol sekarang.",
        "bahagia": "Wah keren! Bagi-bagi cerita senangnya dong!",
        "lega": "Syukur lega! Semoga lancar terus.",
        "kecewa": "Kecewa boleh; jangan dipendam sendiri.",
        "bangga": "Mantap! Kamu pantas bangga pada progresmu.",
    }
    for s in stems:
        for m in modifiers:
            q = f"aku {s} {m}"
            convs.append(
                _entry(
                    q,
                    f"<pikir>Merespons emosi {s}...</pikir>{responses[s]}",
                    s if s in ("bahagia", "lega", "bangga") else "sadness",
                    "emosi_massive",
                )
            )
    return convs


def gen_semua_extra():
    """Gabungkan semua generator tambahan."""
    generators = [
        gen_bahasa_formal_santai,
        gen_pendidikan,
        gen_teknologi,
        gen_spaceax_bantuan,
        gen_kesehatan_motivasi,
        gen_followup_konteks,
        gen_sains_umum,
        gen_indonesia_budaya,
        gen_coding_massive,
        gen_greeting_massive,
        gen_emosi_massive,
    ]
    out = []
    for fn in generators:
        out.extend(fn())
    return out
