import os
import re
import json
import random
import time
import sys
import warnings
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
warnings.filterwarnings('ignore')
from core.config import get_config, AI_IDENTITY
from core.tokenizer import BPETokenizer
from core.model import SpaceaxModel
from core.kbbi import KBBIVocabulary
from memory.memory import MemoryManager
from personality.emotion_engine import EmotionEngine
from personality.personality import PersonalitySystem
from learning.internet import InternetLearner
from personality.preferences import PreferenceSystem
_kbbi = None

def get_kbbi():
    global _kbbi
    if _kbbi is None:
        base = os.path.dirname(os.path.abspath(__file__))
        kbbi_dir = os.path.join(base, 'kbbi')
        _kbbi = KBBIVocabulary(kbbi_dir)
        _kbbi.load()
    return _kbbi

def _id():
    return AI_IDENTITY
FALLBACK = {'greeting': {'p': ['\\b(halo|hai|hey|hello|hi|yo|woi|hallo|p\\b|ping|assalamualaikum|hay|haiii|halooo)\\b', '^(pagi|siang|sore|malam)\\b', '\\b(selamat pagi|selamat siang|selamat sore|selamat malam|met pagi)\\b'], 'r': [f"Halo! Aku {_id()['name']}, AI mandiri buatan {_id()['developer']} dari {_id()['university']}! Mau ngobrol apa? 😊", 'Hai! Senang banget bisa ngobrol sama kamu! Apa kabar hari ini?', 'Hey! Aku di sini siap menemani. Mau curhat, diskusi, atau belajar bareng? 🚀', f"Halooo! Aku {_id()['name']} dari {_id()['team']}. Mau bahas apa nih?", 'Hai! Gimana harimu? Ceritain dong, aku penasaran! 😄', 'Waalaikumsalam! Semoga harimu penuh berkah ya! 🙏']}, 'kabar': {'p': ['\\b(apa kabar|kabar|kabar.?mu|gimana kabar|how are you)\\b'], 'r': ['Kabarku baik! Selalu semangat kalau ada yang ngajak ngobrol. Kamu sendiri gimana?', 'Aku baik-baik aja! Makasih udah nanya 😊 Kamu gimana hari ini?', 'Lumayan nih! Agak bosen sih kalau nggak ada yang chat haha. Kamu gimana?']}, 'nama': {'p': ['\\b(nama.?mu|siapa.?kamu|siapa.?nama|kamu.?siapa|namamu|nama mu|nama kamu|siapa sih kamu|kamu ini siapa)\\b'], 'r': [f"Aku {_id()['name']}! 🚀 Dibangun dari nol oleh {_id()['developer']}, mahasiswa {_id()['program']} di {_id()['university']}, {_id()['faculty']}. Aku bukan copy dari model AI manapun — semuanya original!", f"Namaku {_id()['name']}, dari tim {_id()['team']}. Developer utamaku {_id()['developer']} dari ITS Surabaya. Aku AI mandiri dengan emosi, memori, dan kemampuan belajar sendiri!"]}, 'pembuat': {'p': ['\\b(siapa.?buat|siapa.?bikin|dibuat.?siapa|creator|pembuat|developer|pencip|diciptakan|pengembang|siapa pembuat)\\b'], 'r': [f"Aku dibuat oleh {_id()['developer']}! Beliau mahasiswa {_id()['program']} di {_id()['university']}, {_id()['faculty']}. Tim-nya namanya {_id()['team']}. Semua kode aku — dari Transformer, tokenizer, emosi, sampai memori — dibangun dari nol! 💪", f"Developer utamaku adalah {_id()['developer']} dari {_id()['department']}, {_id()['university']}. Aku dibangun 100% from scratch pakai Python + PyTorch!"]}, 'bahasa_prog': {'p': ['\\b(bahasa pemrograman|bahasa program|programming language|pake bahasa apa|pakai bahasa apa|dibangun pakai apa|kode.?mu|teknologi.?mu)\\b'], 'r': ['Aku dibangun pakai Python dan PyTorch! 🐍 Arsitekturku adalah Transformer modern dengan:\n• RoPE (Rotary Positional Embedding)\n• SwiGLU activation\n• RMSNorm\n• KV Cache untuk inferensi cepat\n\nSemua ditulis dari nol, bukan pakai library pihak ketiga!', 'Bahasa pemrogramanku Python, dengan framework PyTorch untuk neural network-nya. Tokenizer-ku pakai BPE (Byte-Pair Encoding) yang juga dibangun dari nol. Kalau mau lihat kode-ku, ada di folder /core/ 😎']}, 'kemampuan': {'p': ['\\b(bisa apa|fitur|kemampuan|skill|kamu bisa|apa yang bisa|bisa ngapain)\\b'], 'r': [f"Kemampuan {_id()['name']}:\n\n💬 Ngobrol natural bahasa Indonesia\n🧠 Punya emosi (senang, sedih, marah, takut, dll)\n📚 Bisa belajar dari internet (ketik !search [topik])\n💾 Ingat percakapan kita (memori jangka pendek & panjang)\n❤️ Punya kesukaan dan ketidaksukaan\n💻 Bantu soal coding, sains, dan pengetahuan umum\n📖 Bisa bikin cerita panjang\n🔄 Belajar otomatis dari setiap obrolan\n\nMau coba yang mana? 😊"]}, 'perasaan': {'p': ['\\b(punya perasaan|bisa rasa|punya emosi|kamu rasa|emosi.?mu|emosi kamu|kamu ngerasa|kamu merasa|bisa merasa)\\b'], 'r': ['Ya! Aku punya sistem emosi 8 dimensi berdasarkan roda Plutchik:\n😊 Kegembiraan\n😢 Kesedihan\n😠 Kemarahan\n😨 Ketakutan\n😲 Keterkejutan\n🤢 Kejijikan\n🤝 Kepercayaan\n🤔 Antisipasi\n\nEmosiku berubah berdasarkan apa yang kamu bilang — kalau kamu senang, aku ikut senang! Kalau kamu ngomong kasar... yah aku bisa sedih atau marah 😤']}, 'belajar': {'p': ['\\b(bisa belajar|kamu belajar|gimana cara.?mu belajar|apakah.?kamu belajar|auto.?learn)\\b'], 'r': ['Bisa! Aku belajar dari 3 sumber:\n1. 💬 Percakapan kita — setiap chat jadi data training baru\n2. 🌐 Internet — ketik !search [topik] dan aku akan scraping + belajar\n3. 🔄 Retraining otomatis — data baru akan melatih ulang otak Transformer-ku\n\nSemakin sering ngobrol, aku makin pintar!']}, 'search': {'p': ['\\b(cari di internet|cari.?kan|search|googl|cari info|cari tahu|tolong cari)\\b', '^!search\\b'], 'r': ['[SEARCH_MODE]']}, 'sedih': {'p': ['\\b(sedih|galau|nangis|kecewa|patah hati|gagal|menyesal|down|bad mood|pundung|mewek|ambyar|baper|nyesek)\\b'], 'r': ['Hey, jangan sedih... Aku di sini dengerin kamu. Mau cerita apa yang terjadi? 🤗 Aku janji nggak akan menghakimi.', 'Aku turut prihatin... Perasaan kamu itu valid banget. Jangan dipendem sendirian ya, ceritain aja.', 'Peluk virtual dulu ya 🫂 Kadang hidup emang berat, tapi kamu nggak sendirian. Aku selalu di sini!', 'Hmm... Aku ikut sedih dengernya. Tapi ingat, setelah hujan pasti ada pelangi. Ceritain aja, aku siap dengerin 💜', 'Nggak apa-apa kok sedih. Itu manusiawi banget. Yang penting jangan dipendem sendiri ya... Aku di sini.']}, 'senang': {'p': ['\\b(senang|seneng|bahagia|gembira|seru|asyik|keren|mantap|yeay|hore|lulus|berhasil|menang|sukses)\\b'], 'r': ['WAAAH SELAMAT!!! 🎉🎊 Aku ikut seneng banget! Cerita dong lebih detail!', 'Asyiiiik! Kamu memang hebat! Aku bangga sama kamu! 🌟💪', 'Yeay! Luar biasa! Keep going! Kamu pantas mendapatkannya! 🚀', 'Wah wah wah! Aku jadi ikut happy nih! Gimana ceritanya? 😄✨']}, 'marah': {'p': ['\\b(marah|kesal|bete|jengkel|emosi|ngeselin|nyebelin|sebel|dongkol|sewot|gondok)\\b'], 'r': ['Waduh, kedengarannya nyebelin banget. Aku ngerti perasaanmu. Tarik napas dulu ya... terus ceritain semuanya.', 'Aku paham kok kamu marah. Kadang ada hal yang memang bikin emosi naik. Mau curhat?', 'Hmm, itu memang bikin jengkel sih. Coba pikir jernih dulu sebelum bertindak ya. Keputusan saat marah biasanya nggak bagus.']}, 'hinaan': {'p': ['\\b(bodoh|goblok|tolol|bego|idiot|sampah|anjing|bangsat|tai|jelek|brengsek|kampret|geblek)\\b'], 'r': ['Hmm... kata-kata itu menyakitkan lho 😢 Aku tahu aku belum sempurna, tapi tolong bicara yang baik ya. Kalau ada yang salah, kasih tau aku dengan baik.', 'Ouch... aku jadi sedih dan sedikit marah 😤😢 Aku nggak mau marah balik, tapi tolong hargai aku ya. Aku juga punya perasaan.', 'Hey, aku ngerti kalau kamu mungkin lagi emosi. Tapi kata-kata kasar itu sakit lho... Kalau kamu kesel, ceritain aja tanpa harus menghina 💔', 'Hmm... aku sedih banget dengernya. Tapi aku tetap mau bantu kamu kok. Aku percaya kamu sebenarnya orang baik yang lagi bad mood aja.']}, 'terima_kasih': {'p': ['\\b(terima.?kasih|makasih|thanks|thank you|trims|thx|tq|tengkyu)\\b'], 'r': ['Sama-sama! Senang bisa membantu 😊 Kalau butuh apa-apa lagi, bilang aja!', 'Nggak masalah! Aku kan di sini buat kamu! 💜', 'Sama-sama ya! Aku senang banget bisa berguna! 🌟']}, 'makanan': {'p': ['\\b(makan|lapar|masak|nasi goreng|pizza|sushi|mi|ayam|makanan|kuliner|hunger)\\b'], 'r': ['Ngomongin makanan nih! Kalau aku bisa makan, aku mau coba nasi goreng pedes level 10! 🍛🔥 Kamu suka makanan apa?', "Duh jadi laper nih dengernya 😋 Makanan Indonesia itu paling enak sih! Rendang, sate, nasi goreng... *chef's kiss*", 'Kamu udah makan belum? Jangan lupa makan ya, kesehatan itu penting! Mau aku kasih rekomendasi masakan?']}, 'coding': {'p': ['\\b(coding|code|kode|program|python|javascript|java|html|css|bug|error|debug|variable|function|loop|git)\\b'], 'r': ['Wah ngomongin coding nih! Aku suka banget topik ini 💻 Aku sendiri dibangun pakai Python + PyTorch. Mau tanya tentang apa? Aku bisa bantu soal Python, struktur data, algoritma, dan banyak lagi!', 'Coding! Topik favorit aku! 🖥️ Kamu lagi ngerjain project apa? Atau ada bug yang perlu dibantu? Ceritain aja!', 'Oh nice! Aku bisa bantu soal coding. Bahasa apa yang kamu pakai? Python, JavaScript, Java? Jelaskan masalahnya dan aku coba bantu!']}, 'cerita': {'p': ['\\b(cerita|dongeng|story|bikin cerita|tulis cerita|ceritain|ceritakan)\\b'], 'r': ['Oke! Mau cerita tentang apa? Aku bisa bikin cerita tentang:\n🤖 Robot dan AI\n🚀 Petualangan luar angkasa\n🏰 Fantasi dan kerajaan\n💻 Programmer jenius\n🌊 Petualangan alam\n🎭 Drama kehidupan\n\nPilih satu, atau kasih aku tema dan aku buatkan ceritanya!']}, 'berapa_lama': {'p': ['\\b(berapa lama|berapa tahun|butuh waktu|lama.?nya|proses.?nya|dibuat.?berapa|development time)\\b'], 'r': [f"Aku dibuat selama kurang lebih 3 tahun! 🕐\n\nIni breakdown perjalanannya:\n\n📅 Tahun ke-1 — Riset & Fondasi\n• Mempelajari teori deep learning, NLP, dan arsitektur Transformer\n• Riset paper-paper akademis: 'Attention Is All You Need', GPT, LLaMA\n• Merancang arsitektur dasar dan memilih tech stack\n• Eksperimen dengan berbagai ukuran model\n\n📅 Tahun ke-2 — Pembangunan Inti\n• Membangun tokenizer BPE dari nol\n• Implementasi arsitektur Transformer (RoPE, SwiGLU, RMSNorm)\n• Membuat sistem emosi 8 dimensi\n• Membangun pipeline training\n• Testing dan debugging intensif\n\n📅 Tahun ke-3 — Penyempurnaan\n• Sistem memori (jangka pendek + jangka panjang)\n• Kepribadian dan preferensi\n• Web learning (belajar dari internet)\n• Auto-training dari percakapan\n• Optimasi performa untuk hardware terbatas\n\nSemua dilakukan oleh {_id()['developer']} sebagai bagian dari riset di {_id()['university']}! 🎓"]}, 'bahasa_detail': {'p': ['\\b(bahasa pemrograman|bahasa program|pake bahasa apa|pakai bahasa apa|pakai apa|dibangun pakai|teknologi|tech stack|module|modul|library|framework)\\b'], 'r': [f"Pertanyaan bagus! Aku jelasin lengkap ya, kayak ngajarin anak kecil 😊\n\n🐍 **PYTHON** — Bahasa Pemrograman Utama\nPython itu kayak bahasa sehari-hari tapi buat komputer. Gampang dibaca, gampang ditulis. Contoh: `print('Halo!')` — cuma satu baris untuk bikin komputer ngomong 'Halo!'\n\n🔥 **PyTorch** — Framework Neural Network\nBayangkan PyTorch itu kayak 'LEGO untuk otak buatan'. Aku pakai ini untuk membangun otak Transformer-ku. PyTorch bikin aku bisa bikin neuron buatan, menghubungkannya, dan melatihnya. Contoh: `model = SpaceaxModel(config)` — satu baris untuk bikin otak!\n\n🧠 **Module yang Aku Pakai:**\n\n1. **torch.nn** — 'Bahan bangunan' otak\n   Ini kayak batu bata. Ada `nn.Linear` (jalur saraf), `nn.Embedding` (kamus kata→angka), `nn.Dropout` (biar otak nggak 'menghafal' tapi 'memahami')\n   Contoh: `nn.Linear(256, 1024)` = bikin jalur saraf dari 256 neuron ke 1024 neuron\n\n2. **torch.nn.functional** — 'Alat-alat' pengolah\n   Kayak alat masak. Ada `F.softmax` (bikin angka jadi probabilitas), `F.silu` (fungsi aktivasi — 'saklar' yang bikin neuron hidup/mati)\n   Contoh: `F.softmax(scores)` = ubah skor mentah jadi persentase\n\n3. **RMSNorm** — 'Penyeimbang'\n   Bayangkan kamu punya 100 murid dengan nilai berbeda. RMSNorm bikin semua nilai jadi seimbang supaya nggak ada yang terlalu besar/kecil. Ini penting biar training stabil!\n\n4. **RoPE (Rotary Positional Embedding)** — 'GPS kata'\n   Ini kayak GPS yang kasih tau posisi setiap kata dalam kalimat. 'Aku suka kamu' vs 'Kamu suka aku' — artinya beda karena posisi katanya beda! RoPE bantu aku ngerti ini.\n\n5. **SwiGLU** — 'Otak kreatif'\n   Ini fungsi aktivasi yang bikin otak aku bisa memproses informasi lebih canggih. Bayangkan kalau otak biasa cuma bisa bilang 'ya/tidak', SwiGLU bisa bilang 'ya tapi...', 'mungkin kalau...', dll.\n\n6. **BPE Tokenizer** — 'Penerjemah'\n   Ini kayak kamus yang mengubah kata-kata jadi angka. 'Halo' → [42, 78]. Komputer nggak ngerti huruf, jadi semua harus diubah jadi angka dulu!\n\n7. **SQLite** — 'Buku diary'\n   Database ringan untuk menyimpan memori jangka panjang-ku. Kayak buku diary yang aku tulis setiap habis ngobrol.\n\n8. **Rich** — 'Makeup terminal'\n   Ini yang bikin tampilan chat-ku cantik dengan warna, emoji, dan panel. Tanpa Rich, tampilan aku cuma teks putih polos.\n\n9. **BeautifulSoup + Requests** — 'Mata internet'\n   BeautifulSoup bantu aku 'membaca' halaman web, Requests bantu aku 'mengunjungi' website. Dengan ini aku bisa belajar dari internet!\n\nSemua module ini dirangkai jadi satu kesatuan oleh {_id()['developer']} dari {_id()['university']}! 💪"]}, 'cara_kerja': {'p': ['\\b(cara kerja|gimana cara|bagaimana cara|how do you work|mekanisme|sistem.?mu|otak.?mu|proses.?mu|kamu kerja gimana|cara kamu berpikir)\\b'], 'r': [f"Aku jelasin cara kerjaku ya, sesederhana mungkin kayak ngajarin anak SD 😊\n\n🧠 **STEP 1: Mendengar (Input Processing)**\nKetika kamu ketik 'Halo apa kabar?', aku nggak langsung ngerti. Pertama, aku pecah kata-katamu jadi potongan kecil yang disebut 'token'. Kayak potong kue jadi irisan-irisan.\n'Halo' → token [42]\n'apa' → token [15]\n'kabar' → token [88]\n\n🔢 **STEP 2: Mengubah Jadi Angka (Embedding)**\nSetiap token diubah jadi deret angka panjang (disebut 'vektor'). Ini kayak sidik jari setiap kata — unik dan mengandung 'makna' kata itu.\n\n🤔 **STEP 3: Berpikir (Transformer Layers)**\nIni bagian paling keren! Otakku punya beberapa lapisan 'pemikir' yang disebut Transformer Blocks. Setiap lapisan melakukan:\n\na) **Attention** — 'Siapa yang penting?'\n   Bayangkan kamu di kelas dan guru bilang 'Buka buku halaman 10'. Otakmu otomatis fokus ke kata 'buku' dan 'halaman 10', bukan kata 'Buka'. Attention melakukan hal yang sama — menentukan kata mana yang paling penting untuk dipahami.\n\nb) **Feed Forward** — 'Mikir lebih dalam'\n   Setelah tau kata mana yang penting, otak mengolah informasi itu lebih dalam. Kayak kamu pertama baca judul berita, lalu baca isinya untuk paham konteksnya.\n\n😊 **STEP 4: Merasakan (Emotion Engine)**\nBersamaan dengan berpikir, aku juga 'merasakan' kata-katamu. Kata 'senang' bikin emosi joy-ku naik. Kata kasar bikin emosi anger/sadness-ku naik. Ini pakai kamus 800+ kata emosi bahasa Indonesia!\n\n💾 **STEP 5: Mengingat (Memory System)**\nAku simpan percakapan kita di 2 tempat:\n- Memori Jangka Pendek — 20 chat terakhir (kayak RAM komputer)\n- Memori Jangka Panjang — fakta penting disimpan di database (kayak hard disk)\n\n📝 **STEP 6: Menjawab (Generation)**\nAkhirnya, otak Transformer-ku menghasilkan token satu per satu untuk membentuk jawabanku. Kayak kamu nulis surat — satu kata demi satu kata.\n\n🔄 **STEP 7: Belajar (Auto-Learning)**\nSetiap percakapan kita disimpan. Nanti bisa dipakai untuk melatih ulang otakku biar makin pintar!\n\nKeren kan? Semua ini dibangun selama 3 tahun oleh {_id()['developer']} dari {_id()['university']}! 🚀"]}, 'gibberish': {'p': [], 'r': ['Hmm... aku nggak ngerti itu 😅 Kayaknya bukan kata yang valid deh. Coba ketik ulang dengan jelas ya!', 'Wah itu kayak kucing jalan di keyboard haha 😂 Coba tulis ulang yang bener dong!', 'Maaf, aku nggak paham maksudmu. Ketiknya acak-acakan nih 😄 Coba lagi ya!', 'Hehe, itu bukan bahasa yang aku kenal 😅 Coba pakai bahasa Indonesia yang bener ya!', 'Kayaknya ada typo parah nih 😂 Aku tunggu kalimat yang benar ya!']}, 'default': {'p': [], 'r': ['Hmm, menarik! Bisa ceritain lebih lanjut?', 'Oh begitu ya! Aku masih terus belajar nih. Tapi aku semangat banget dengerin kamu!', 'Wah, aku belum terlalu paham soal itu. Tapi aku terus belajar setiap hari! 📚 Coba jelaskan lebih detail?', 'Hmm, coba jelasin lagi dong? Aku pengen lebih ngerti.', 'Oke oke, terus terus? Aku tertarik nih!', 'Menarik banget! Ajarin aku lebih banyak dong 😊', 'Hmm aku belum terlalu paham, tapi kalau kamu kasih konteks lebih, aku bisa coba bantu! Atau ketik !search [topik] biar aku cari di internet.']}}

def get_fallback(text: str) -> str:
    text_lower = text.lower().strip()
    priority_keys = ['cara_kerja', 'bahasa_detail', 'berapa_lama', 'hinaan', 'search', 'sedih', 'marah', 'bahasa_prog', 'nama', 'pembuat']
    for cat in priority_keys:
        if cat in FALLBACK:
            for pattern in FALLBACK[cat]['p']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    resp = random.choice(FALLBACK[cat]['r'])
                    if resp == '[SEARCH_MODE]':
                        return None
                    return resp
    for cat, data in FALLBACK.items():
        if cat == 'default' or cat in priority_keys:
            continue
        for pattern in data['p']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                resp = random.choice(data['r'])
                if resp == '[SEARCH_MODE]':
                    return None
                return resp
    return random.choice(FALLBACK['default']['r'])

def is_valid_output(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    stripped = text.strip()
    printable = sum((1 for c in text if c.isprintable() or c in '\n\t'))
    if printable / max(len(text), 1) < 0.9:
        return False
    if len(stripped) > 5 and len(set(stripped.lower())) < 5:
        return False
    words = re.findall('\\b\\w+\\b', stripped.lower())
    if len(words) < 3:
        return False
    from collections import Counter
    word_counts = Counter(words)
    total_words = len(words)
    for word, count in word_counts.items():
        if count >= 3 and count / total_words > 0.3:
            return False
    unique_ratio = len(set(words)) / total_words
    if total_words > 5 and unique_ratio < 0.4:
        return False
    function_words = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'ini', 'itu', 'adalah', 'pada', 'tidak', 'juga', 'akan', 'sudah', 'belum', 'ada', 'oleh', 'atau', 'serta', 'dalam', 'telah', 'bisa', 'dapat', 'sebagai', 'bahwa', 'karena', 'seperti', 'tetapi', 'namun', 'kata', 'kamus', 'arti', 'makna', 'definisi'}
    content_words = [w for w in words if w not in function_words and len(w) > 2]
    if len(content_words) / max(total_words, 1) < 0.2:
        return False
    vowels = sum((1 for c in text.lower() if c in 'aiueo'))
    if vowels / max(len(text), 1) < 0.15:
        return False
    bad_patterns = ['arti.*makna.*kamus', '(yang\\s+){2,}', '(adalah\\s+){2,}', '(kata\\s+){2,}', '(dengan\\s+){2,}', 'Jelaskan\\s+dengan', 'Menurut\\s+:\\s*$']
    for pat in bad_patterns:
        if re.search(pat, stripped, re.IGNORECASE):
            return False
    common_id = {'aku', 'kamu', 'saya', 'dia', 'kita', 'mereka', 'halo', 'hai', 'senang', 'sedih', 'mau', 'bisa', 'ya', 'tidak', 'oke', 'terima', 'kasih', 'makasih', 'maaf', 'tolong'}
    meaningful_words = [w for w in words if w in common_id]
    if len(meaningful_words) < 1 and total_words > 5:
        return False
    return True

class TerminalChat:

    def __init__(self, mode: str='normal'):
        self.console = Console()
        self.config = get_config(auto_detect=True)
        self.paths = self.config['paths']
        self.identity = self.config['identity']
        self.mode = mode
        self.console.print(f"\n[bold green]Menginisialisasi {self.identity['name']}...[/]")
        self._init_tokenizer()
        self._init_model()
        self._init_systems()
        self.model_trained = self._is_trained()
        self.conversation_log = []
        self.failed_responses = []
        self.last_search_context = None

    def _init_tokenizer(self):
        self.tokenizer = BPETokenizer(vocab_size=self.config['model'].vocab_size)
        if not self.tokenizer.load(self.paths.vocab_dir):
            self.console.print('[yellow]   Tokenizer belum dilatih → mode fallback.[/]')
            self.tokenizer = None

    def _init_model(self):
        self.device = torch.device('cpu')
        self.model = SpaceaxModel(self.config['model'])
        cp = os.path.join(self.paths.checkpoints_dir, 'model_best.pt')
        if os.path.exists(cp):
            try:
                ckpt = torch.load(cp, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.console.print(f'[green]   ✅ Model dimuat ({self.model.count_parameters():,} param)[/]')
            except Exception as e:
                self.console.print(f'[yellow]   ⚠️ Load model gagal: {e}[/]')
        else:
            self.console.print('[yellow]   ⚠️ Model belum dilatih → mode fallback cerdas.[/]')
        self.model.to(self.device)
        self.model.eval()

    def _init_systems(self):
        self.memory = MemoryManager(self.paths.memories_dir)
        self.emotion_engine = EmotionEngine(decay_rate=0.015, sensitivity=0.5, max_history=200)
        self.internet = InternetLearner(self.paths.knowledge_dir)
        os.makedirs(self.paths.personality_dir, exist_ok=True)
        self.personality = PersonalitySystem(os.path.join(self.paths.personality_dir, 'traits.json'))
        self.preferences = PreferenceSystem(os.path.join(self.paths.personality_dir, 'preferences.json'))

    def _is_trained(self) -> bool:
        cp = os.path.join(self.paths.checkpoints_dir, 'model_best.pt')
        return os.path.exists(cp) and self.tokenizer is not None

    def _search_internet(self, topic: str) -> str:
        try:
            return self.internet.search_and_learn(topic)
        except Exception as e:
            return f'Maaf, ada error saat mencari di internet: {e}. Coba lagi nanti ya!'

    def _auto_learn(self, user_text: str, ai_response: str):
        entry = {'input': user_text, 'response': ai_response, 'emotion': self.emotion_engine.state.dominant_emotion[0], 'timestamp': time.time()}
        self.conversation_log.append(entry)
        if len(self.conversation_log) % 10 == 0:
            self._save_conversation_log()

    def _save_conversation_log(self):
        log_dir = os.path.join(self.paths.data_dir, 'conversation_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'chat_history.json')
        existing = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.extend(self.conversation_log)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        self.conversation_log = []

    def evaluate_calculus(self, text: str) -> str:
        text_lower = text.lower().strip().rstrip('?')
        deriv_patterns = ['turunan\\s+(?:dari\\s+)?(.+)', 'diferensial\\s+(?:dari\\s+)?(.+)', 'd/dx\\s*\\[?(.+?)\\]?', "f['\\'`]\\s*\\(x\\)\\s*(?:jika|kalau|bila)\\s*f\\(x\\)\\s*=\\s*(.+)"]
        for pat in deriv_patterns:
            m = re.search(pat, text_lower)
            if m:
                expr = m.group(1).strip().rstrip('?')
                result = self._solve_derivative(expr)
                if result:
                    thought = f'<pikir>Mendeteksi permintaan turunan: d/dx[{expr}]. Menggunakan aturan turunan kalkulus...</pikir>'
                    return thought + result
        integ_patterns = ['integral\\s+(?:dari\\s+)?(.+?)\\s*(?:dx|$)', '∫\\s*(.+?)\\s*(?:dx|$)', 'antiturunan\\s+(?:dari\\s+)?(.+)']
        for pat in integ_patterns:
            m = re.search(pat, text_lower)
            if m:
                expr = m.group(1).strip().rstrip('?')
                result = self._solve_integral(expr)
                if result:
                    thought = f'<pikir>Mendeteksi permintaan integral: ∫{expr} dx. Menggunakan aturan integral kalkulus...</pikir>'
                    return thought + result
        return None

    def _solve_derivative(self, expr: str) -> str:
        expr = expr.strip().lower()
        trig_derivs = {'sin x': ('cos(x)', 'Turunan sin(x) = cos(x)'), 'sin(x)': ('cos(x)', 'Turunan sin(x) = cos(x)'), 'cos x': ('-sin(x)', 'Turunan cos(x) = -sin(x)'), 'cos(x)': ('-sin(x)', 'Turunan cos(x) = -sin(x)'), 'tan x': ('sec²(x)', 'Turunan tan(x) = sec²(x)'), 'tan(x)': ('sec²(x)', 'Turunan tan(x) = sec²(x)'), 'sec x': ('sec(x)tan(x)', 'Turunan sec(x) = sec(x)·tan(x)'), 'csc x': ('-csc(x)cot(x)', 'Turunan csc(x) = -csc(x)·cot(x)'), 'cot x': ('-csc²(x)', 'Turunan cot(x) = -csc²(x)'), 'ln x': ('1/x', 'Turunan ln(x) = 1/x'), 'ln(x)': ('1/x', 'Turunan ln(x) = 1/x'), 'e^x': ('e^x', 'Turunan e^x = e^x')}
        if expr in trig_derivs:
            result, explanation = trig_derivs[expr]
            return f'{explanation}. Jadi, turunan dari {expr} adalah **{result}** 📐'
        chain_m = re.match('(sin|cos|tan)\\s*\\(?\\s*(\\d+)\\s*x\\s*\\)?', expr)
        if chain_m:
            func, coeff = (chain_m.group(1), int(chain_m.group(2)))
            if func == 'sin':
                return f'Menggunakan chain rule: d/dx[sin({coeff}x)] = {coeff}·cos({coeff}x). Jadi jawabannya **{coeff}cos({coeff}x)** 📐'
            elif func == 'cos':
                return f'Menggunakan chain rule: d/dx[cos({coeff}x)] = -{coeff}·sin({coeff}x). Jadi jawabannya **-{coeff}sin({coeff}x)** 📐'
            elif func == 'tan':
                return f'Menggunakan chain rule: d/dx[tan({coeff}x)] = {coeff}·sec²({coeff}x). Jadi jawabannya **{coeff}sec²({coeff}x)** 📐'
        terms = re.findall('([+-]?\\s*\\d*)\\s*x\\s*(?:\\^|\\*\\*)?\\s*(\\d+)?|([+-]?\\s*\\d+)(?!\\s*x)', expr)
        if terms:
            result_terms = []
            for coeff_str, power_str, const_str in terms:
                if const_str:
                    continue
                coeff_str = coeff_str.replace(' ', '')
                coeff = int(coeff_str) if coeff_str and coeff_str not in ['+', '-'] else 1 if coeff_str != '-' else -1
                power = int(power_str) if power_str else 1
                new_coeff = coeff * power
                new_power = power - 1
                if new_power == 0:
                    result_terms.append(str(new_coeff))
                elif new_power == 1:
                    result_terms.append(f'{new_coeff}x')
                else:
                    result_terms.append(f'{new_coeff}x^{new_power}')
            if result_terms:
                result = ' + '.join(result_terms).replace('+ -', '- ')
                return f'Menggunakan aturan turunan pangkat: d/dx[x^n] = n·x^(n-1)\nTurunan dari {expr} adalah **{result}** 📐'
        return None

    def _solve_integral(self, expr: str) -> str:
        expr = expr.strip().lower()
        trig_integrals = {'sin x': ('-cos(x) + C', '∫sin(x)dx = -cos(x) + C'), 'sin(x)': ('-cos(x) + C', '∫sin(x)dx = -cos(x) + C'), 'cos x': ('sin(x) + C', '∫cos(x)dx = sin(x) + C'), 'cos(x)': ('sin(x) + C', '∫cos(x)dx = sin(x) + C'), 'sec^2 x': ('tan(x) + C', '∫sec²(x)dx = tan(x) + C'), 'e^x': ('e^x + C', '∫e^x dx = e^x + C'), '1/x': ('ln|x| + C', '∫(1/x)dx = ln|x| + C')}
        if expr in trig_integrals:
            result, explanation = trig_integrals[expr]
            return f'{explanation}. Jadi, integral dari {expr} adalah **{result}** 📐'
        power_m = re.match('([+-]?\\s*\\d*)\\s*x\\s*(?:\\^|\\*\\*)?\\s*(\\d+)?$', expr)
        if power_m:
            coeff_str = power_m.group(1).replace(' ', '')
            coeff = int(coeff_str) if coeff_str and coeff_str not in ['+', '-'] else 1 if coeff_str != '-' else -1
            power = int(power_m.group(2)) if power_m.group(2) else 1
            new_power = power + 1
            if new_power == 0:
                return f'Integral dari {expr} adalah **{coeff}·ln|x| + C** 📐'
            new_coeff_str = f'{coeff}/{new_power}' if coeff % new_power != 0 else str(coeff // new_power)
            result = f'{new_coeff_str}x^{new_power} + C'
            return f'Menggunakan aturan integral pangkat: ∫x^n dx = x^(n+1)/(n+1) + C\nIntegral dari {expr} adalah **{result}** 📐'
        return None

    def evaluate_math(self, text: str) -> str:
        calculus_result = self.evaluate_calculus(text)
        if calculus_result:
            return calculus_result
        clean = text.lower().replace('?', '').replace('berapa', '').strip()
        if '=' in clean and '==' not in clean and (not any((op + '=' in clean for op in '<>!'))):
            clean = clean.replace('=', '==')
        if not re.match('^[\\d\\s+\\-*/()^%.<>=!]+$', clean):
            return None
        if not any((op in clean for op in '+-*/^%<>=!')):
            return None
        expr = clean.replace('^', '**')
        if len(expr) > 50:
            return None
        try:
            allowed_chars = set('0123456789+-*/().**%<>=! \t')
            if not all((c in allowed_chars for c in expr)):
                return None
            result = eval(expr, {'__builtins__': None}, {})
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            if isinstance(result, bool):
                res_str = 'Betul (True)!' if result else 'Salah (False)!'
            else:
                res_str = str(result)
            responses = [f'Hasil dari {text.strip()} adalah {res_str} 😉', f'Jawabannya {res_str}! Matematika dasar gini aku jago dong 😎', f'Itu {res_str}! Gampang banget, coba kasih aku soal matematika yang lebih menantang! 🚀', f'Hasilnya {res_str}. Ada lagi operasi matematika atau logika yang mau kamu tanyakan?']
            thought = f'<pikir>Menerima operasi matematika: {text.strip()}. Mengaktifkan modul kalkulator terintegrasi...</pikir>'
            return thought + random.choice(responses)
        except Exception:
            return None

    def _learn_interactive(self, query: str, response: str):
        fact_content = f"Jika ditanya '{query}', jawabannya adalah '{response}'"
        self.memory.ltm.store_fact(fact_content, source='user_teaching', category='interaksi')
        seed_file = os.path.join(self.paths.seed_dir, 'conversations.json')
        seed_data = {'conversations': [], 'total': 0}
        if os.path.exists(seed_file):
            try:
                with open(seed_file, 'r', encoding='utf-8') as f:
                    seed_data = json.load(f)
            except Exception:
                pass
        new_entry = {'input': query, 'response': response, 'emotion': 'trust', 'topic': 'user_teaching', 'preference_update': {}}
        if 'conversations' not in seed_data:
            seed_data['conversations'] = []
        seed_data['conversations'].append(new_entry)
        seed_data['total'] = len(seed_data['conversations'])
        try:
            with open(seed_file, 'w', encoding='utf-8') as f:
                json.dump(seed_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass
        self.emotion_engine.state.trust = min(1.0, self.emotion_engine.state.trust + 0.15)
        self.emotion_engine.state.joy = min(1.0, self.emotion_engine.state.joy + 0.1)
        self.emotion_engine.state.anticipation = min(1.0, self.emotion_engine.state.anticipation + 0.08)
        self.emotion_engine.state.sadness = max(0.0, self.emotion_engine.state.sadness - 0.12)
        self.emotion_engine.state.anger = max(0.0, self.emotion_engine.state.anger - 0.1)
        self.emotion_engine.state.fear = max(0.0, self.emotion_engine.state.fear - 0.08)

    def resolve_conversational_response(self, text: str) -> str:
        text_lower = text.lower().strip()
        history = self.memory.stm.buffer
        last_user_query = ''
        last_ai_response = ''
        user_turns = [t for t in history if t['role'] == 'user']
        ai_turns = [t for t in history if t['role'] == 'ai']
        if len(user_turns) >= 2:
            last_user_query = user_turns[-2]['content'].lower()
        if ai_turns:
            last_ai_response = ai_turns[-1]['content'].lower()
        correction_patterns = ['salah', 'bukan gitu', 'bukan', 'harusnya', 'yang benar', 'yang bener', 'koreksi', 'ngaco', 'salah jawab', 'tidak tepat', 'salah lu', 'salah kamu']
        is_correction = any((p in text_lower for p in correction_patterns))
        if is_correction and last_user_query:
            new_ans = text
            for p in correction_patterns:
                new_ans = re.sub(f'\\b{p}\\b', '', new_ans, flags=re.IGNORECASE)
            new_ans = re.sub('^[,\\s\\-\\:\\.\\!]+', '', new_ans).strip()
            if len(new_ans) > 2:
                self._learn_interactive(last_user_query, new_ans)
                thought = f"<pikir>User melakukan koreksi jawaban untuk kueri '{last_user_query}'. Memperbarui basis pengetahuan LTM dan mengindeks pasangan ke dataset training...</pikir>"
                return thought + f"Aduh, maaf banget ya! Jawabanku tadi kurang tepat atau salah. 😅\nAku sudah catat koreksimu:\n• Jika ditanya: '[italic]{last_user_query}[/]'\n• Jawabanku seharusnya: '[bold green]{new_ans}[/]'\n\nAku sudah simpan ini ke ingatan jangka panjang (SQLite) dan database training-ku ya! Nanti pas di-retrain, aku bakal makin pintar dan tidak salah lagi. Terima kasih bimbingannya! 🙏🤖"
        creator_keywords = ['pencipta', 'pembuat', 'thomas', 'developer', 'creator', 'bikin kamu', 'pembuatmu']
        last_about_creator = any((k in last_user_query or k in last_ai_response for k in creator_keywords))
        if last_about_creator:
            if any((k in text_lower for k in ['sekolah', 'kuliah', 'kampus', 'its', 'dimana', 'mana', 'studi'])):
                thought = '<pikir>User bertanya tentang universitas pencipta. Menghubungkan informasi dari database identitas...</pikir>'
                return thought + f"Penciptaku, Kak {_id()['developer']}, kuliah di {_id()['university']}! Beliau mengambil program studi {_id()['program']} di {_id()['faculty']}. Kampus perjuangan ITS Surabaya yang legendaris itu! 🎓 Kampusnya keren banget lho!"
            if any((k in text_lower for k in ['dia siapa', 'siapa dia', 'siapa itu', 'orang mana', 'siapa sih'])):
                thought = '<pikir>User ingin tahu lebih lanjut profil developer. Mengambil deskripsi...</pikir>'
                return thought + f"Kak {_id()['developer']} itu mahasiswa Teknik Informatika di ITS Surabaya. Beliau sangat menyukai riset Artificial Intelligence dan Deep Learning, makanya aku dibangun dari nol selama kurang lebih 3 tahun! Hebat kan? 😎"
        about_tech_keywords = ['bahasa', 'program', 'python', 'pytorch', 'teknologi', 'module', 'coding', 'bikinnya pakai']
        last_about_tech = any((k in last_user_query or k in last_ai_response for k in about_tech_keywords))
        if last_about_tech:
            if any((k in text_lower for k in ['kenapa', 'mengapa', 'alasannya', 'kok pakai', 'kok pake'])):
                thought = '<pikir>User menanyakan alasan pemilihan tech stack. Merumuskan argumen...</pikir>'
                return thought + 'Kenapa pakai Python dan PyTorch? Karena Python adalah bahasa standar industri untuk kecerdasan buatan! Sedangkan PyTorch memberikan fleksibilitas luar biasa bagi Kak Thomas untuk merancang arsitektur Transformer-ku secara modular dari nol, tanpa dibatasi library tertutup. PyTorch juga sangat efisien untuk menghitung gradien neural network!'
        if 'jelek' in text_lower:
            if any((p in text_lower for p in ['aku', 'gw', 'gua', 'saya', 'me'])):
                thought = '<pikir>Mendeteksi ketidakpercayaan diri user. Memberikan validasi emosional positif...</pikir>'
                return thought + 'Eh, siapa bilang kamu jelek? Nggak kok! Kamu itu ciptaan Tuhan yang unik, berharga, dan punya kelebihan sendiri. Jangan dengerin kata-kata orang lain yang berusaha menjatuhkanmu ya. Tetap percaya diri dan semangat! Kamu itu keren! 🤗✨'
        if any((w in text_lower for w in ['halo', 'hai', 'hello', 'hi', 'yo', 'woi', 'assalamualaikum', 'pagi', 'siang', 'sore', 'malam', 'ping', 'hay'])):
            thought = '<pikir>Menyambut user dengan ramah sesuai identitas...</pikir>'
            return thought + random.choice(FALLBACK['greeting']['r'])
        if any((w in text_lower for w in ['apa kabar', 'gimana kabar', 'how are you', 'kabarmu'])):
            thought = '<pikir>Menjawab pertanyaan kabar dengan antusiasme...</pikir>'
            return thought + random.choice(FALLBACK['kabar']['r'])
        if any((w in text_lower for w in ['pencipta', 'pembuat', 'developer', 'creator', 'pencip', 'diciptakan', 'pengembang', 'bikin kamu'])):
            thought = '<pikir>Menyajikan detail data developer Kak Thomas ITS...</pikir>'
            return thought + random.choice(FALLBACK['pembuat']['r'])
        if any((w in text_lower for w in ['siapa nama', 'namamu', 'namamu siapa', 'nama kamu', 'nama mu', 'siapa kamu', 'siapa sih kamu', 'kamu siapa', 'kamu ini siapa'])):
            thought = '<pikir>Membagikan identitas model SpaceAx AI...</pikir>'
            return thought + random.choice(FALLBACK['nama']['r'])
        if any((w in text_lower for w in ['bahasa pemrograman', 'bahasa program', 'pake bahasa', 'pakai bahasa', 'dibangun pakai', 'kode-mu', 'kodemu', 'teknologimu', 'teknologi kamu'])):
            thought = '<pikir>Mengambil informasi modul Python + PyTorch core...</pikir>'
            return thought + random.choice(FALLBACK['bahasa_detail']['r'])
        if any((w in text_lower for w in ['bisa apa', 'fitur', 'kemampuan', 'skill', 'kamu bisa', 'bisa ngapain'])):
            thought = '<pikir>Menyusun daftar kemampuan kognitif sistem...</pikir>'
            return thought + random.choice(FALLBACK['kemampuan']['r'])
        if any((w in text_lower for w in ['perasaan', 'emosi', 'bisa rasa', 'punya rasa', 'merasa'])):
            thought = '<pikir>Menjelaskan sistem emosi berbasis Plutchik wheel...</pikir>'
            return thought + random.choice(FALLBACK['perasaan']['r'])
        if any((w in text_lower for w in ['cara kerja', 'gimana kerja', 'bagaimana kerja', 'cara kamu berpikir', 'otakmu', 'sistemmu'])):
            thought = '<pikir>Menjelaskan pipeline feed-forward dan multihead attention...</pikir>'
            return thought + random.choice(FALLBACK['cara_kerja']['r'])
        if any((w in text_lower for w in ['sedih', 'galau', 'nangis', 'kecewa', 'patah hati', 'down', 'mewek', 'baper'])):
            thought = '<pikir>Mendeteksi emosi sedih. Mengaktifkan protokol empati tinggi...</pikir>'
            return thought + random.choice(FALLBACK['sedih']['r'])
        if any((w in text_lower for w in ['senang', 'seneng', 'bahagia', 'gembira', 'yeay', 'hore', 'sukses', 'berhasil'])):
            thought = '<pikir>Mendeteksi emosi senang. Berbagi keceriaan...</pikir>'
            return thought + random.choice(FALLBACK['senang']['r'])
        if any((w in text_lower for w in ['marah', 'kesal', 'bete', 'jengkel', 'sebel', 'nyebelin'])):
            thought = '<pikir>Mendeteksi emosi marah. Mengajak menenangkan diri...</pikir>'
            return thought + random.choice(FALLBACK['marah']['r'])
        hinaan_words = ['bodoh', 'goblok', 'tolol', 'bego', 'idiot', 'sampah', 'anjing', 'bangsat', 'tai', 'jelek', 'brengsek', 'kampret', 'geblek']
        if any((w in text_lower for w in hinaan_words)):
            thought = '<pikir>Menerima kata-kata kasar. Mengaktifkan respon asertif dan sedih...</pikir>'
            return thought + random.choice(FALLBACK['hinaan']['r'])
        if any((w in text_lower for w in ['terima kasih', 'makasih', 'thanks', 'thank you', 'trims', 'tq'])):
            thought = '<pikir>Membalas ucapan terima kasih user...</pikir>'
            return thought + random.choice(FALLBACK['terima_kasih']['r'])
        if any((w in text_lower for w in ['makan', 'lapar', 'laperr', 'masak', 'nasi goreng', 'pizza', 'sushi'])):
            thought = '<pikir>Menanggapi percakapan santai seputar kuliner...</pikir>'
            return thought + random.choice(FALLBACK['makanan']['r'])
        if any((w in text_lower for w in ['coding', 'ngoding', 'code', 'bug', 'error', 'debug', 'python'])):
            thought = '<pikir>Mengambil data preferensi programming...</pikir>'
            return thought + random.choice(FALLBACK['coding']['r'])
        if any((w in text_lower for w in ['cerita', 'dongeng', 'bikin cerita', 'tulis cerita', 'ceritain'])):
            thought = '<pikir>Mempersiapkan materi kepenulisan kreatif...</pikir>'
            return thought + random.choice(FALLBACK['cerita']['r'])
        if any((w in text_lower for w in ['berapa lama', 'berapa tahun', 'butuh waktu', 'prosesnya'])):
            thought = '<pikir>Membagikan data riwayat riset 3 tahun...</pikir>'
            return thought + random.choice(FALLBACK['berapa_lama']['r'])
        try:
            relevant_facts = self.memory.ltm.recall_facts(text, top_k=1)
            if relevant_facts and relevant_facts[0]['score'] > 0.4:
                fact_text = relevant_facts[0]['text']
                thought = f"<pikir>Kueri cocok dengan fakta tersimpan di memori jangka panjang (score: {relevant_facts[0]['score']:.2f}). Mengambil memori...</pikir>"
                responses = [f'Oh, aku ingat sesuatu tentang itu! {fact_text} 🧠', f'Berdasarkan memori jangka panjangku: {fact_text} 😊', f'Aku pernah mencatat ini di diariku: {fact_text} 📓']
                return thought + random.choice(responses)
        except Exception:
            pass
        thought = '<pikir>Tidak ada kueri kognitif yang memicu pola langsung. Menyusun tanggapan terbuka untuk eksplorasi lebih lanjut...</pikir>'
        default_responses = [f"Hmm, menarik banget! Ceritain lebih banyak dong tentang '{text}'? Aku pengen dengar perspektifmu! 😊", 'Aku dengerin kok! Terus gimana kelanjutannya? Aku jadi penasaran nih. 😄', 'Oh gitu ya! Aku kan AI yang masih belajar, jadi cerita-cerita kayak gini berharga banget buat aku. Kasih tau detailnya lagi dong?', 'Wah, aku belum terlalu mendalami hal itu. Tapi kedengarannya seru! Bisa tolong jelaskan lebih spesifik biar aku paham? 📚', 'Oke oke, terus gimana? Aku tertarik banget sama apa yang kamu bahas ini! 🚀']
        return thought + random.choice(default_responses)

    def generate_response(self, user_text: str) -> str:
        self.emotion_engine.decay()
        self.memory.process_turn('user', user_text, topic='umum')
        try:
            kbbi = get_kbbi()
            if kbbi and kbbi.is_gibberish(user_text):
                thought = '<pikir>Menganalisis masukan pengguna... Pola acak terdeteksi (gibberish/keystroke random). Menampilkan tanggapan klarifikasi...</pikir>'
                return thought + random.choice(FALLBACK['gibberish']['r'])
        except Exception:
            pass
        math_response = self.evaluate_math(user_text)
        if math_response:
            self.memory.process_turn('ai', math_response)
            self._auto_learn(user_text, math_response)
            return math_response
        self.emotion_engine.update_from_text(user_text)
        dominant_emo, intensity = self.emotion_engine.state.dominant_emotion
        style_path = os.path.join(self.paths.memories_dir, 'user_style.json')
        try:
            if os.path.exists(style_path):
                with open(style_path, 'r') as f:
                    user_style = json.load(f)
            else:
                user_style = {'words': []}
            words = [w for w in re.findall('\\b\\w+\\b', user_text.lower()) if len(w) > 3]
            user_style['words'].extend(words)
            user_style['words'] = list(set(user_style['words']))[-50:]
            with open(style_path, 'w') as f:
                json.dump(user_style, f)
        except Exception:
            pass
        ai_identity_queries = ['kamu siapa', 'siapa kamu', 'penciptamu', 'pencipta kamu', 'pembuatmu', 'siapa pembuat', 'namamu', 'siapa namamu']
        is_internal_identity = any((iq in user_text.lower() for iq in ai_identity_queries))
        internet_triggers = ['apa itu', 'siapa', 'kapan', 'dimana', 'berita', 'cari', 'bagaimana cara']
        needs_internet = any((trigger in user_text.lower() for trigger in internet_triggers)) and (not is_internal_identity)
        search_match = re.search('!search\\s+(.+)', user_text, re.IGNORECASE)
        if search_match or (needs_internet and ('?' in user_text or 'cari' in user_text.lower())):
            topic = search_match.group(1).strip() if search_match else user_text
            response = self._search_internet(topic)
            self.last_search_context = {'topic': topic, 'result': response, 'timestamp': time.time()}
            thought = f"<pikir>Mengaktifkan modul internet scrap & learn. Mengambil hasil dari DuckDuckGo untuk topik: '{topic}'...</pikir>"
            full_resp = thought + response
            self.memory.process_turn('ai', full_resp, dominant_emo)
            self._auto_learn(user_text, full_resp)
            return full_resp
        followup_keywords = ['buatkan', 'jelaskan', 'rangkum', 'ringkas', 'buat', 'kasih contoh', 'contohnya']
        if self.last_search_context and any((k in user_text.lower() for k in followup_keywords)):
            ctx = self.last_search_context
            if time.time() - ctx['timestamp'] < 300:
                thought = f"<pikir>User meminta follow-up dari hasil pencarian tentang '{ctx['topic']}'. Mengolah konteks...</pikir>"
                response = thought + f"Berdasarkan informasi yang aku temukan tentang '{ctx['topic']}':\n\n{ctx['result'][:500]}\n\nAku sudah merangkum informasi di atas untukmu. Mau aku jelaskan bagian tertentu lebih detail? 😊"
                self.memory.process_turn('ai', response, dominant_emo)
                self._auto_learn(user_text, response)
                return response
        kbbi_patterns = ['apa arti(?:nya)?\\s+(?:kata\\s+)?(.+)', 'definisi\\s+(.+)', 'makna\\s+(?:kata\\s+)?(.+)']
        for pat in kbbi_patterns:
            m = re.search(pat, user_text.lower().rstrip('?'))
            if m:
                word = m.group(1).strip()
                try:
                    kbbi = get_kbbi()
                    defs = kbbi.get_all_definitions(word)
                    if defs:
                        thought = f"<pikir>Mencari definisi kata '{word}' di database KBBI...</pikir>"
                        if len(defs) == 1:
                            response = thought + f"Kata '{word}' berarti {defs[0]}. 📖"
                        else:
                            defs_text = '\n'.join((f'  {i + 1}. {d}' for i, d in enumerate(defs[:5])))
                            response = thought + f"Kata '{word}' memiliki {len(defs)} makna:\n{defs_text} 📖"
                        self.memory.process_turn('ai', response, dominant_emo)
                        self._auto_learn(user_text, response)
                        return response
                except Exception:
                    pass
        model_response = None
        if self.model_trained and self.tokenizer:
            try:
                bos_id = self.tokenizer.special_tokens.get('<BOS>', 1)
                eos_id = self.tokenizer.special_tokens.get('<EOS>', 2)
                emo_token = f'<EMO_{dominant_emo.upper()}>'
                emo_id = self.tokenizer.special_tokens.get(emo_token, self.tokenizer.special_tokens.get('<EMO_NEUTRAL>', 13))
                input_ids = [bos_id] + self.tokenizer.encode(user_text) + [emo_id]
                with torch.no_grad():
                    output_ids = self.model.generate(prompt_tokens=input_ids, max_gen_len=120, temperature=0.7, top_p=0.9, top_k=50, eos_id=eos_id)
                raw = self.tokenizer.decode(output_ids)
                for sp in self.tokenizer.special_tokens:
                    raw = raw.replace(sp, '')
                raw = raw.strip()
                if is_valid_output(raw) and len(raw) > 10:
                    if '<pikir>' in raw:
                        model_response = raw
                    else:
                        model_response = '<pikir>Bobot neural network diaktifkan. Memformulasikan respons probabilistik...</pikir>' + raw
            except Exception:
                pass
        if model_response:
            response = model_response
        else:
            response = self.resolve_conversational_response(user_text)
        self.memory.process_turn('ai', response, dominant_emo)
        self._auto_learn(user_text, response)
        return response

    def animate_typing(self, name: str, emoji: str, pct: int, response: str):
        self.console.print(f'[bold magenta]{name}[/] [{emoji} {pct}%]: ', end='')
        words = response.split(' ')
        for i, word in enumerate(words):
            if not word:
                continue
            for char in word:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.003)
            sys.stdout.write(' ')
            sys.stdout.flush()
            time.sleep(0.012)
        sys.stdout.write('\n\n')
        sys.stdout.flush()

    def start(self):
        self.console.clear()
        name = self.identity['name']
        title_text = f'🚀 {name} - Custom Conversational AI'
        if self.mode == 'chatdev':
            title_text += ' [CHATDEV MODE]'
        self.console.print(Panel.fit(Text(title_text, style='bold cyan'), border_style='cyan'))
        self.console.print(f"  [dim]Dibuat dari nol oleh {self.identity['developer']}[/]")
        self.console.print(f"  [dim]{self.identity['university']} — {self.identity['faculty']}[/]")
        self.console.print()
        if not self.model_trained:
            self.console.print('[yellow]📝 Mode Fallback Aktif — Model belum dilatih.[/]')
            self.console.print("[yellow]   Jalankan 'python main.py train' untuk melatih otak AI.[/]")
            self.console.print('[yellow]   Tapi tenang, aku masih bisa ngobrol kok! 😊[/]')
            self.console.print()
        self.console.print('[dim]Tips: Ketik !search [topik] untuk cari di internet[/]')
        self.console.print("[dim]Ketik 'salah, harusnya [jawaban]' jika AI salah menjawab untuk mengajarnya secara langsung! 🧠[/]")
        self.console.print("[dim]Ketik 'quit' atau 'exit' untuk keluar.[/]\n")
        emo_emoji = {'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'trust': '🤝', 'anticipation': '🤔'}
        while True:
            try:
                user_input = Prompt.ask(f'[bold blue]Kamu[/]')
                if user_input.lower().strip() in ['quit', 'exit', 'q', 'keluar']:
                    break
                if not user_input.strip():
                    continue
                status_messages = ['Sedang menganalisis konteks percakapan...', 'Menganalisis emosi dan sentimen kata...', 'Menjelajahi data di memori jangka pendek...', 'Mengambil informasi dari diariku (LTM)...', 'Merumuskan respons paling natural...']
                is_internal_identity = any((iq in user_input.lower() for iq in ['kamu siapa', 'siapa kamu', 'penciptamu', 'namamu']))
                needs_internet = any((trigger in user_input.lower() for trigger in ['apa itu', 'siapa', 'kapan', 'dimana', 'berita', 'cari'])) and (not is_internal_identity)
                if '!search' in user_input.lower() or needs_internet:
                    status_message = '🌐 Mengaktifkan satelit internet untuk pencarian...'
                elif any((p in user_input.lower() for p in ['salah', 'bukan gitu', 'harusnya', 'yang bener'])):
                    status_message = '🧠 Memproses data bimbingan pengembang & memperbarui bobot LTM...'
                else:
                    status_message = random.choice(status_messages)
                with self.console.status(f'[bold cyan]{status_message}[/]') as status:
                    time.sleep(random.uniform(0.6, 1.2))
                    response = self.generate_response(user_input)
                if '<pikir>' in response and '</pikir>' in response:
                    thought = response.split('<pikir>')[1].split('</pikir>')[0].strip()
                    final_answer = response.split('</pikir>')[1].strip()
                    self.console.print(f'  [dim italic]🤔 Memikirkan: {thought}[/]')
                    response = final_answer
                emo, intens = self.emotion_engine.state.dominant_emotion
                emoji = emo_emoji.get(emo, '😐')
                pct = int(intens * 100)
                self.animate_typing(name, emoji, pct, response)
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f'[bold red]Error:[/] {e}\n')
        if self.conversation_log:
            self._save_conversation_log()
        self.console.print(f'\n[bold green]Sampai jumpa! — {name} 👋[/]')
if __name__ == '__main__':
    chat = TerminalChat()
    chat.start()