"""
SpaceAx — integrasi KBBI & leksikon Indonesia.
Sumber: kbbi_v_part*.json, indonesian-words, list_*, slang, root, stop words.
"""

import json
import os
import re
import glob
import random
from typing import Dict, List, Optional


class KBBIVocabulary:
    """Kamus dan leksikon bahasa Indonesia untuk training + validasi chat."""

    WORDLIST_FILES = (
        "indonesian-words.txt",
        "list_0.5.1.txt",
        "list_1.0.0.txt",
    )

    def __init__(self, kbbi_dir: str):
        self.kbbi_dir = kbbi_dir
        self.words = set()
        self.definitions = {}
        self.all_definitions = {}
        self.word_classes = {}
        self.kata_dasar = {}
        self.kata_turunan = {}
        self.gabungan_kata = {}
        self.idioms = {}
        self.peribahasa = {}
        self.stop_words = set()
        self.slang_to_formal: Dict[str, str] = {}
        self.root_words = set()
        self.list_words = set()
        self.aux_stats: Dict[str, int] = {}
        self._loaded = False

    def load(self) -> int:
        """Load semua kata dari file KBBI JSON. Return jumlah kata."""
        if self._loaded:
            return len(self.words)

        files = sorted(glob.glob(os.path.join(self.kbbi_dir, "kbbi_v_part*.json")))
        if not files:
            print(f"⚠️ Tidak ada file KBBI di {self.kbbi_dir}")
            return 0

        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for key, val in data.items():
                    # Ambil kata dasar dari key
                    clean = self._clean_word(key)
                    if clean and len(clean) >= 2:
                        self.words.add(clean)

                    # Ambil dari data entri
                    if isinstance(val, dict) and val.get("status") == "success" and "data" in val:
                        entries = val["data"].get("entri", [])
                        for entry in entries:
                            nama = entry.get("nama", "")
                            clean_nama = self._clean_word(nama)
                            if not clean_nama or len(clean_nama) < 2:
                                continue
                            
                            self.words.add(clean_nama)

                            # Kumpulkan SEMUA definisi
                            defs = []
                            classes = []
                            for makna in entry.get("makna", []):
                                for sub in makna.get("submakna", []):
                                    if sub:
                                        defs.append(sub)
                                for kelas in makna.get("kelas", []):
                                    kode = kelas.get("kode", "")
                                    nama_kelas = kelas.get("nama", "")
                                    if nama_kelas:
                                        classes.append(nama_kelas)

                            if defs:
                                self.all_definitions[clean_nama] = defs
                                self.definitions[clean_nama] = defs[0][:200]
                            if classes:
                                self.word_classes[clean_nama] = list(set(classes))

                            # Kata dasar
                            kd_list = entry.get("kata_dasar", [])
                            if kd_list:
                                self.kata_dasar[clean_nama] = [
                                    self._clean_word(kd) for kd in kd_list 
                                    if self._clean_word(kd)
                                ]

                            # Kata turunan
                            kt_list = entry.get("kata_turunan", [])
                            if kt_list:
                                self.kata_turunan[clean_nama] = [
                                    self._clean_word(kt) for kt in kt_list
                                    if self._clean_word(kt)
                                ]

                            # Gabungan kata
                            gk_list = entry.get("gabungan_kata", [])
                            if gk_list:
                                self.gabungan_kata[clean_nama] = [
                                    self._clean_word(gk) for gk in gk_list
                                    if self._clean_word(gk)
                                ]

                            # Idiom
                            for idiom in entry.get("idiom", []):
                                if isinstance(idiom, str) and idiom.strip():
                                    self.idioms[idiom.strip()] = clean_nama

                            # Peribahasa
                            for pb in entry.get("peribahasa", []):
                                if isinstance(pb, str) and pb.strip():
                                    self.peribahasa[pb.strip()] = clean_nama

            except Exception as e:
                print(f"⚠️ Error loading {fpath}: {e}")

        self._load_auxiliary_lexicons()
        self._loaded = True
        print(f"📚 Leksikon dimuat: {len(self.words):,} kata unik")
        print(f"   📝 Definisi KBBI: {len(self.all_definitions):,}")
        print(f"   📋 Daftar kata (txt): {len(self.list_words):,}")
        print(f"   🌱 Kata dasar (root): {len(self.root_words):,}")
        print(f"   💬 Slang→baku: {len(self.slang_to_formal):,}")
        print(f"   ⏹️ Stop words: {len(self.stop_words):,}")
        print(f"   📜 Idiom: {len(self.idioms):,} | Peribahasa: {len(self.peribahasa):,}")
        return len(self.words)

    def _load_auxiliary_lexicons(self) -> None:
        """Muat file txt/json tambahan di folder kbbi/."""
        added_list = 0
        for fname in self.WORDLIST_FILES:
            path = os.path.join(self.kbbi_dir, fname)
            if not os.path.isfile(path):
                continue
            n = self._ingest_wordlist_file(path)
            added_list += n
            self.aux_stats[fname] = n

        stop_path = os.path.join(self.kbbi_dir, "combined_stop_words.txt")
        if os.path.isfile(stop_path):
            with open(stop_path, "r", encoding="utf-8") as f:
                for line in f:
                    for w in line.strip().lower().split():
                        cw = self._clean_word(w)
                        if cw:
                            self.stop_words.add(cw)
            self.aux_stats["combined_stop_words.txt"] = len(self.stop_words)

        root_path = os.path.join(self.kbbi_dir, "combined_root_words.txt")
        if os.path.isfile(root_path):
            with open(root_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = self._clean_word(line.strip())
                    if w and len(w) >= 2:
                        self.root_words.add(w)
                        self.words.add(w)

        slang_path = os.path.join(self.kbbi_dir, "combined_slang_words.txt")
        if os.path.isfile(slang_path):
            try:
                with open(slang_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                data = json.loads(raw) if raw.startswith("{") else {}
                for slang, formal in data.items():
                    s = self._clean_word(str(slang))
                    if not s or len(s) < 2:
                        continue
                    ftxt = str(formal).strip()
                    self.slang_to_formal[s] = ftxt
                    self.words.add(s)
                    for part in re.findall(r"[a-z]{2,}", ftxt.lower()):
                        self.words.add(part)
            except Exception as e:
                print(f"⚠️ Gagal memuat slang: {e}")
            self.aux_stats["combined_slang_words.txt"] = len(self.slang_to_formal)

        self.list_words = set(self.words) - set(self.all_definitions.keys())

    def _ingest_wordlist_file(self, path: str) -> int:
        count = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = self._clean_word(line.strip())
                if w and len(w) >= 2 and re.match(r"^[a-z][a-z\-]*[a-z]$|^[a-z]{2,}$", w):
                    if w not in self.words:
                        count += 1
                    self.words.add(w)
        return count

    @staticmethod
    def should_refresh_seed(seed_file: str, kbbi_dir: str) -> bool:
        """True jika file leksikon lebih baru dari seed atau belum ada cukup pasangan kbbi."""
        if os.environ.get("SPACEAX_KBBI_SYNC", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return True
        if not os.path.isfile(seed_file):
            return True
        seed_mtime = os.path.getmtime(seed_file)
        watch = [
            os.path.join(kbbi_dir, f)
            for f in (
                list(KBBIVocabulary.WORDLIST_FILES)
                + [
                    "combined_slang_words.txt",
                    "combined_stop_words.txt",
                    "combined_root_words.txt",
                ]
            )
        ]
        watch.extend(sorted(glob.glob(os.path.join(kbbi_dir, "kbbi_v_part*.json"))))
        for p in watch:
            if os.path.isfile(p) and os.path.getmtime(p) > seed_mtime:
                return True
        try:
            with open(seed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            kbbi_n = sum(
                1
                for c in data.get("conversations", [])
                if str(c.get("topic", "")).startswith("kbbi")
            )
            if kbbi_n < 800:
                return True
        except Exception:
            return True
        return False

    @staticmethod
    def strip_kbbi_topics(conversations: list) -> list:
        return [
            c
            for c in conversations
            if not str(c.get("topic", "")).startswith("kbbi")
        ]

    def _clean_word(self, text: str) -> str:
        """Bersihkan kata dari karakter khusus."""
        if not text:
            return ""
        # Hapus karakter dalam kurung, angka, tanda baca khusus
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'[0-9]', '', text)
        # Hapus titik pemisah suku kata
        text = text.replace('.', '')
        text = text.strip().lower()
        # Hanya simpan kata alfabet + spasi + tanda hubung
        text = re.sub(r'[^a-z\s\-]', '', text)
        return text.strip()

    def is_valid_word(self, word: str) -> bool:
        """Cek apakah kata ada di KBBI."""
        if not self._loaded:
            self.load()
        return word.lower().strip() in self.words

    def is_gibberish(self, text: str) -> bool:
        """Deteksi apakah teks adalah gibberish (kata acak tanpa makna)."""
        if not self._loaded:
            self.load()

        clean_text = text.strip()
        if not clean_text:
            return True

        # 1. Izinkan ekspresi matematika / angka dasar
        if re.match(r"^[\d\s+\-*/()^%.<>=!?]+$", clean_text):
            return False

        # 2. Input sangat pendek
        if len(clean_text) <= 3:
            return False

        # 3. Whitelist pola tawa/ekspresi
        laugh_patterns = [
            r'[hw]a[ha]+', r'he[he]+', r'hi[hi]+', r'hu[hu]+',
            r'w+k+w+k*', r'xi[xi]+', r'lo+l+', r'hm+',
            r'ah+', r'oh+', r'uh+', r'eh+', r'la+h+', r'ya+h+',
            r'du+h+', r'hu+f+t?', r'hi+k+s*', r'wo+w+', r'wa+h+',
            r'ih+', r'ew+',
        ]
        text_lower = clean_text.lower()
        remaining_text = text_lower
        for pat in laugh_patterns:
            remaining_text = re.sub(pat, '', remaining_text)
        remaining_clean = re.sub(r'[\s.,!?]+', '', remaining_text)
        if not remaining_clean:
            return False

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not words:
            return False

        tech_words = {
            "html", "css", "xml", "json", "sql", "git", "cpp", "bash",
            "linux", "unix", "struct", "class", "void", "main", "func",
            "const", "rust", "torch", "numpy", "pandas", "flask",
            "django", "react", "node", "npm", "pip", "conda",
            "tensorflow", "pytorch", "sklearn", "matplotlib",
        }

        expression_words = {
            "haha", "hahaha", "hahahaha", "hehe", "hehehe",
            "hihi", "hihihi", "huhu", "huhuhu", "wkwk", "wkwkwk",
            "lol", "lmao", "rofl", "hmm", "hmmm", "wow", "wahh",
            "ahh", "ohh", "ehh", "hiks", "hikss", "huft", "hufft",
            "duh", "aduh", "yah", "yahh", "lahh", "lahhh",
        }
        words_to_check = [w for w in words if w not in expression_words]

        if not words_to_check:
            return False

        valid = sum(1 for w in words_to_check if self._is_word_like(w) or w in tech_words)
        ratio = valid / len(words_to_check) if words_to_check else 1.0

        if ratio < 0.2 and len(words_to_check) >= 2:
            return True

        if len(words_to_check) == 1 and len(words_to_check[0]) > 8:
            w = words_to_check[0]
            if w not in self.words and not self._is_word_like(w) and w not in tech_words:
                return True

        for w in words_to_check:
            if w in self.words or w in tech_words or self._is_word_like(w):
                continue
            is_expression = any(re.fullmatch(pat, w) for pat in laugh_patterns)
            if is_expression:
                continue
            consonants_streak = 0
            for c in w:
                if c not in 'aiueo':
                    consonants_streak += 1
                    if consonants_streak >= 5:
                        return True
                else:
                    consonants_streak = 0

        return False

    def _is_word_like(self, word: str) -> bool:
        """Cek apakah kata 'mirip' kata valid."""
        w = word.lower()
        if w in self.words:
            return True
        if w in self.slang_to_formal:
            return True
        if w in self.root_words:
            return True
        slang = {
            "gue", "gw", "gua", "lo", "lu", "elo", "elu", "nggak", "gak", "ga",
            "udah", "udh", "aja", "doang", "banget", "bgt", "emang", "emg",
            "gimana", "gmn", "kalo", "kayak", "kayaknya", "kyk", "ngomong",
            "ngapain", "ngeliat", "ngerasa", "nyari", "nanya", "nggak",
            "ya", "sih", "dong", "deh", "nih", "loh", "lho", "wkwk",
            "haha", "hihi", "hehe", "xixi", "wkwkwk", "awkwk",
            "btw", "otw", "fyi", "asap", "lol", "omg", "wtf",
            "ok", "oke", "yoi", "sip", "mantap", "mantul",
            "gabut", "mager", "ngoding", "coding", "nge", "ngerjain",
            "ngomongin", "ngerti", "ngga", "enggak", "gpp",
            "tertawa", "ketawa", "ngakak", "cakep", "gokil",
            "anjir", "anjay", "dah", "bener", "bgt",
            "lahh", "lahhh", "yahh", "yahhh",
            "turunan", "integral", "kalkulus", "diferensial", "logaritma",
            "trigonometri", "sin", "cos", "tan", "limit",
        }
        if w in slang:
            return True
        if len(w) > 8 and w not in self.words and w not in slang:
            return False
        vowels = sum(1 for c in w if c in 'aiueo')
        if len(w) > 0 and vowels / len(w) >= 0.3:
            return True
        return False

    def get_definition(self, word: str) -> str:
        """Ambil definisi kata dari KBBI."""
        if not self._loaded:
            self.load()
        return self.definitions.get(word.lower().strip(), "")

    def get_all_definitions(self, word: str) -> list:
        """Ambil semua definisi kata dari KBBI."""
        if not self._loaded:
            self.load()
        return self.all_definitions.get(word.lower().strip(), [])

    # ========================================================================
    # Training Data Generation — Menghasilkan ribuan pasangan Q&A dari KBBI
    # ========================================================================

    def generate_rich_training_data(self, max_pairs: int = 15000) -> list:
        """Generate training data berkualitas tinggi dari KBBI.
        
        Menghasilkan ribuan variasi pertanyaan dan jawaban yang sangat beragam
        agar model belajar merangkai kata secara organik, bukan sekadar
        menghafal template.
        
        Returns:
            List of dicts
        """
        if not self._loaded:
            self.load()

        pairs = []
        words_with_defs = list(self.all_definitions.keys())
        # Stress test: gunakan lebih banyak kata
        sampled_words = random.sample(words_with_defs, min(max_pairs, len(words_with_defs)))

        pikir_prefixes = [
            "Mencari makna kata '{word}'...",
            "Mengingat definisi untuk '{word}'...",
            "Menganalisis kata '{word}'...",
            "Kata '{word}' memiliki arti...",
            "Menurut kamus, '{word}' berarti...",
            "Hmm, '{word}' itu...",
        ]

        # Variasi gaya bahasa (formal, santai, analitik, singkat)
        def generate_organic_response(word, defs, classes, kd):
            style = random.choice(["formal", "santai", "analitik", "singkat"])
            pikir = f"<pikir>{random.choice(pikir_prefixes).format(word=word)}</pikir>"
            
            arti_utama = defs[0]
            
            if style == "formal":
                ans = f"Menurut kamus bahasa Indonesia, kata '{word}' berarti {arti_utama}."
                if len(defs) > 1:
                    ans += f" Selain itu, kata ini juga dapat bermakna {defs[1]}."
                if classes:
                    ans += f" Dari segi kelas kata, '{word}' termasuk golongan {', '.join(classes)}."
            elif style == "santai":
                ans = f"Kata '{word}' itu artinya {arti_utama}. "
                if classes:
                    ans += f"Oh ya, ini tuh termasuk kata {classes[0]} lho."
                if kd:
                    ans += f" Kata dasarnya dari '{kd[0]}'."
            elif style == "analitik":
                ans = f"Analisis kata '{word}':\n"
                ans += f"- Makna utama: {arti_utama}\n"
                if len(defs) > 1:
                    ans += f"- Makna sekunder: {defs[1]}\n"
                if classes:
                    ans += f"- Kelas kata: {', '.join(classes)}\n"
                if kd:
                    ans += f"- Akar kata: {kd[0]}"
            else: # singkat
                ans = f"'{word}': {arti_utama}."

            # Kadang tambahkan emoji acak untuk variasi ekspresi
            if random.random() < 0.3:
                ans += " " + random.choice(["📚", "💡", "🤓", "✨", "📖", "📝"])
                
            return pikir + ans

        # 1. Definisi Campuran
        for word in sampled_words:
            defs = self.all_definitions.get(word, [])
            if not defs: continue
            classes = self.word_classes.get(word, [])
            kd = self.kata_dasar.get(word, [])

            # Generate random question variants
            q_variants = [
                f"Apa arti {word}?",
                f"{word} artinya apa?",
                f"Definisi {word} dong",
                f"Jelaskan makna kata {word}",
                f"Kamu tahu arti {word} nggak?",
                f"Maksud dari {word} itu apa sih?",
                f"Apa yang dimaksud {word}?",
                f"kata {word} maksudnya apa?",
                f"makna {word}",
            ]
            
            q = random.choice(q_variants)
            
            # Tambahkan noise/typo pada input kadang-kadang untuk melatih robustness
            if random.random() < 0.2:
                q = q.lower()
            if random.random() < 0.1:
                q = q.replace("?", "")

            a = generate_organic_response(word, defs, classes, kd)
            
            pairs.append({
                "input": q, "response": a,
                "emotion": random.choice(["neutral", "anticipation", "joy"]), 
                "topic": "kbbi_mixed",
                "preference_update": {"belajar": 1}
            })

        # 2. Idiom & peribahasa (dibatasi agar tidak mendominasi dataset)
        idiom_items = list(self.idioms.items())
        random.shuffle(idiom_items)
        for idiom, base_word in idiom_items[:800]:
            q_variants = [
                f"Apa arti idiom '{idiom}'?",
                f"Maksud ungkapan '{idiom}' apa?",
                f"Jelaskan idiom {idiom}",
                f"ungkapan {idiom} artinya?"
            ]
            q = random.choice(q_variants)
            
            pikir = f"<pikir>Menganalisis kiasan '{idiom}'...</pikir>"
            ans = f"Idiom '{idiom}' adalah kiasan dalam bahasa Indonesia. Karena mengandung kata '{base_word}', maknanya berkaitan dengan hal tersebut, biasanya digunakan untuk menggambarkan situasi tertentu secara kiasan."
            
            pairs.append({
                "input": q, "response": pikir + ans,
                "emotion": "anticipation", "topic": "kbbi_idiom",
                "preference_update": {}
            })

        peribahasa_items = list(self.peribahasa.items())
        random.shuffle(peribahasa_items)
        for pb, base_word in peribahasa_items[:800]:
            q_variants = [
                f"Apa arti peribahasa '{pb}'?",
                f"Makna pepatah '{pb}'",
                f"Jelaskan peribahasa {pb}",
                f"Tahu arti peribahasa {pb} nggak?"
            ]
            q = random.choice(q_variants)
            
            pikir = f"<pikir>Menerjemahkan makna tersirat dari '{pb}'...</pikir>"
            styles = [
                f"Peribahasa '{pb}' ini mengajarkan kita tentang suatu kebijaksanaan hidup. Kata kuncinya '{base_word}'.",
                f"Makna dari pepatah '{pb}' sangat dalam, ini adalah perumpamaan tradisional yang membawa pesan moral.",
                f"Dalam budaya kita, '{pb}' dipakai untuk menasihati seseorang melalui perumpamaan."
            ]
            ans = random.choice(styles)
            
            pairs.append({
                "input": q, "response": pikir + ans,
                "emotion": "anticipation", "topic": "kbbi_peribahasa",
                "preference_update": {}
            })

        # 3. Tantangan Kalimat (Membuat kalimat dari kata)
        words_for_sentences = random.sample(sampled_words, min(800, len(sampled_words)))
        for word in words_for_sentences:
            defs = self.all_definitions.get(word, [""])[0]
            if len(defs) < 5: continue
            
            q_variants = [
                f"Buatkan kalimat pakai kata '{word}'",
                f"Contoh kalimat dengan kata {word}",
                f"Gunakan {word} dalam kalimat",
                f"Gimana cara pakai kata {word}?"
            ]
            q = random.choice(q_variants)
            
            subjects = ["Saya", "Mereka", "Dia", "Budi", "Pemerintah", "Masyarakat", "Kita"]
            contexts = ["kemarin", "dengan sangat hati-hati", "di masa depan", "dalam rapat tersebut", "sehari-hari"]
            
            kalimat = f"{random.choice(subjects)} menggunakan prinsip {word} {random.choice(contexts)}."
            
            pikir = f"<pikir>Merangkai kalimat organik dengan kata '{word}' yang berarti '{defs[:50]}...'</pikir>"
            ans = f"Tentu! Mengingat '{word}' artinya '{defs}', berikut contoh penggunaannya:\n\n\"{kalimat}\""
            
            pairs.append({
                "input": q, "response": pikir + ans,
                "emotion": "joy", "topic": "kbbi_kalimat",
                "preference_update": {}
            })

        print(f"  📝 Generated {len(pairs)} pasangan definisi KBBI")
        return pairs

    def generate_slang_training_data(self, max_pairs: int = 2500) -> list:
        """Pasangan gaul→baku agar model paham percakapan Indonesia nyata."""
        if not self._loaded:
            self.load()
        if not self.slang_to_formal:
            return []

        items = list(self.slang_to_formal.items())
        random.shuffle(items)
        items = items[:max_pairs]
        pairs = []
        q_templates = [
            "apa arti {slang}",
            "{slang} itu apa",
            "gaul {slang} artinya",
            "bahasa baku dari {slang}",
            "{slang} benernya apa",
        ]
        for slang, formal in items:
            q = random.choice(q_templates).format(slang=slang)
            pikir = f"<pikir>Menerjemahkan gaul '{slang}' ke bahasa baku...</pikir>"
            ans = (
                f"'{slang}' dalam percakapan santai biasanya berarti **{formal}**. "
                f"Kalau nulis formal, pakai: {formal}."
            )
            pairs.append({
                "input": q,
                "response": pikir + ans,
                "emotion": "neutral",
                "topic": "kbbi_slang",
                "preference_update": {},
            })
            if random.random() < 0.35:
                q2 = f"kalau orang bilang {slang} maksudnya apa"
                pairs.append({
                    "input": q2,
                    "response": pikir + f"Maksudnya: {formal}.",
                    "emotion": "neutral",
                    "topic": "kbbi_slang",
                    "preference_update": {},
                })
        return pairs

    def generate_lexicon_training_data(self, max_pairs: int = 2000) -> list:
        """Kata dari daftar leksikon (tanpa entri KBBI penuh) — tetap valid secara bahasa."""
        if not self._loaded:
            self.load()
        candidates = [
            w for w in self.list_words
            if len(w) >= 4 and w.isalpha() and w not in self.stop_words
        ]
        if not candidates:
            return []
        random.shuffle(candidates)
        sampled = candidates[:max_pairs]
        pairs = []
        for word in sampled:
            q = random.choice([
                f"apakah '{word}' kata yang valid",
                f"kata {word} ada nggak di bahasa indonesia",
                f"{word} itu kata bener?",
            ])
            pikir = f"<pikir>Memeriksa leksikon untuk '{word}'...</pikir>"
            ans = (
                f"Ya, '{word}' termasuk kosakata bahasa Indonesia yang dikenali leksikon. "
                f"Kalau butuh definisi lengkap, tanya: apa arti {word}?"
            )
            pairs.append({
                "input": q,
                "response": pikir + ans,
                "emotion": "neutral",
                "topic": "kbbi_lexicon",
                "preference_update": {},
            })
        return pairs

    def generate_grammar_training_data(self) -> list:
        """Partikel & stop word — kaidah dasar (bukan definisi kamus)."""
        if not self._loaded:
            self.load()
        pairs = []
        particles = [
            ("yang", "Partikel penghubung: 'buku yang baru' — yang menghubungkan frasa."),
            ("di", "Partikel depan tempat/waktu: 'di rumah', 'di pagi hari'."),
            ("ke", "Partikel arah: 'ke sekolah'."),
            ("tidak", "Negasi: 'saya tidak makan'."),
            ("sudah", "Aspek waktu selesai: 'sudah makan'."),
            ("akan", "Aspek waktu rencana: 'akan pergi'."),
            ("sedang", "Aspek berlangsung: 'sedang belajar'."),
        ]
        for word, expl in particles:
            if word in self.stop_words or word in self.words:
                q = f"apa fungsi kata {word}"
                pairs.append({
                    "input": q,
                    "response": f"<pikir>Menjelaskan kaidah tata bahasa...</pikir>{expl}",
                    "emotion": "neutral",
                    "topic": "kbbi_grammar",
                    "preference_update": {},
                })
        return pairs

    def enrich_all_training_data(
        self,
        *,
        max_def_pairs: int = 5000,
        max_slang_pairs: int = 2500,
        max_lexicon_pairs: int = 1500,
    ) -> list:
        """Gabungan semua generator KBBI + leksikon untuk seed dataset."""
        if not self._loaded:
            self.load()
        all_pairs = []
        all_pairs.extend(self.generate_rich_training_data(max_pairs=max_def_pairs))
        all_pairs.extend(self.generate_slang_training_data(max_pairs=max_slang_pairs))
        all_pairs.extend(self.generate_lexicon_training_data(max_pairs=max_lexicon_pairs))
        all_pairs.extend(self.generate_grammar_training_data())
        random.shuffle(all_pairs)
        print(f"  📚 Total pasangan KBBI+leksikon: {len(all_pairs):,}")
        return all_pairs

    def generate_corpus(self, max_chars: int = 12_000_000) -> str:
        """Corpus untuk melatih tokenizer BPE (sampel besar, dibatasi memori)."""
        if not self._loaded:
            self.load()

        chunks: List[str] = []

        def_words = list(self.all_definitions.items())
        random.shuffle(def_words)
        for word, defs in def_words[: min(25000, len(def_words))]:
            for d in defs[:2]:
                chunks.append(f"{word} adalah {d}")
                chunks.append(f"arti {word}: {d}")

        slang_items = list(self.slang_to_formal.items())
        random.shuffle(slang_items)
        for slang, formal in slang_items[: min(8000, len(slang_items))]:
            chunks.append(f"{slang} artinya {formal}")
            chunks.append(f"dalam bahasa baku {slang} disebut {formal}")

        list_sample = list(self.words)
        random.shuffle(list_sample)
        for w in list_sample[:80000]:
            if len(w) >= 3:
                chunks.append(w)
                chunks.append(f"kata {w}")

        for idiom, base in list(self.idioms.items())[:500]:
            chunks.append(f"idiom {idiom} berkaitan dengan {base}")
        for pb, base in list(self.peribahasa.items())[:500]:
            chunks.append(f"peribahasa {pb}")

        corpus = " ".join(chunks)
        if len(corpus) > max_chars:
            corpus = corpus[:max_chars]
        return corpus

    def get_all_words(self) -> list:
        """Return semua kata KBBI sebagai list."""
        if not self._loaded:
            self.load()
        return sorted(list(self.words))
