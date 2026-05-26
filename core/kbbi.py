import json
import os
import re
import glob
import random

class KBBIVocabulary:

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
        self._loaded = False

    def load(self) -> int:
        if self._loaded:
            return len(self.words)
        files = sorted(glob.glob(os.path.join(self.kbbi_dir, 'kbbi_v_part*.json')))
        if not files:
            print(f'⚠️ Tidak ada file KBBI di {self.kbbi_dir}')
            return 0
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, val in data.items():
                    clean = self._clean_word(key)
                    if clean and len(clean) >= 2:
                        self.words.add(clean)
                    if isinstance(val, dict) and val.get('status') == 'success' and ('data' in val):
                        entries = val['data'].get('entri', [])
                        for entry in entries:
                            nama = entry.get('nama', '')
                            clean_nama = self._clean_word(nama)
                            if not clean_nama or len(clean_nama) < 2:
                                continue
                            self.words.add(clean_nama)
                            defs = []
                            classes = []
                            for makna in entry.get('makna', []):
                                for sub in makna.get('submakna', []):
                                    if sub:
                                        defs.append(sub)
                                for kelas in makna.get('kelas', []):
                                    kode = kelas.get('kode', '')
                                    nama_kelas = kelas.get('nama', '')
                                    if nama_kelas:
                                        classes.append(nama_kelas)
                            if defs:
                                self.all_definitions[clean_nama] = defs
                                self.definitions[clean_nama] = defs[0][:200]
                            if classes:
                                self.word_classes[clean_nama] = list(set(classes))
                            kd_list = entry.get('kata_dasar', [])
                            if kd_list:
                                self.kata_dasar[clean_nama] = [self._clean_word(kd) for kd in kd_list if self._clean_word(kd)]
                            kt_list = entry.get('kata_turunan', [])
                            if kt_list:
                                self.kata_turunan[clean_nama] = [self._clean_word(kt) for kt in kt_list if self._clean_word(kt)]
                            gk_list = entry.get('gabungan_kata', [])
                            if gk_list:
                                self.gabungan_kata[clean_nama] = [self._clean_word(gk) for gk in gk_list if self._clean_word(gk)]
                            for idiom in entry.get('idiom', []):
                                if isinstance(idiom, str) and idiom.strip():
                                    self.idioms[idiom.strip()] = clean_nama
                            for pb in entry.get('peribahasa', []):
                                if isinstance(pb, str) and pb.strip():
                                    self.peribahasa[pb.strip()] = clean_nama
            except Exception as e:
                print(f'⚠️ Error loading {fpath}: {e}')
        self._loaded = True
        print(f'📚 KBBI dimuat: {len(self.words):,} kata unik')
        print(f'   📝 Definisi: {len(self.all_definitions):,}')
        print(f'   🏷️ Kelas kata: {len(self.word_classes):,}')
        print(f'   🔗 Kata dasar: {len(self.kata_dasar):,}')
        print(f'   📜 Idiom: {len(self.idioms):,}')
        print(f'   💬 Peribahasa: {len(self.peribahasa):,}')
        return len(self.words)

    def _clean_word(self, text: str) -> str:
        if not text:
            return ''
        text = re.sub('\\([^)]*\\)', '', text)
        text = re.sub('[0-9]', '', text)
        text = text.replace('.', '')
        text = text.strip().lower()
        text = re.sub('[^a-z\\s\\-]', '', text)
        return text.strip()

    def is_valid_word(self, word: str) -> bool:
        if not self._loaded:
            self.load()
        return word.lower().strip() in self.words

    def is_gibberish(self, text: str) -> bool:
        if not self._loaded:
            self.load()
        clean_text = text.strip()
        if not clean_text:
            return True
        if re.match('^[\\d\\s+\\-*/()^%.<>=!?]+$', clean_text):
            return False
        if len(clean_text) <= 3:
            return False
        laugh_patterns = ['[hw]a[ha]+', 'he[he]+', 'hi[hi]+', 'hu[hu]+', 'w+k+w+k*', 'xi[xi]+', 'lo+l+', 'hm+', 'ah+', 'oh+', 'uh+', 'eh+', 'la+h+', 'ya+h+', 'du+h+', 'hu+f+t?', 'hi+k+s*', 'wo+w+', 'wa+h+', 'ih+', 'ew+']
        text_lower = clean_text.lower()
        remaining_text = text_lower
        for pat in laugh_patterns:
            remaining_text = re.sub(pat, '', remaining_text)
        remaining_clean = re.sub('[\\s.,!?]+', '', remaining_text)
        if not remaining_clean:
            return False
        words = re.findall('\\b[a-zA-Z]{3,}\\b', text.lower())
        if not words:
            return False
        tech_words = {'html', 'css', 'xml', 'json', 'sql', 'git', 'cpp', 'bash', 'linux', 'unix', 'struct', 'class', 'void', 'main', 'func', 'const', 'rust', 'torch', 'numpy', 'pandas', 'flask', 'django', 'react', 'node', 'npm', 'pip', 'conda', 'tensorflow', 'pytorch', 'sklearn', 'matplotlib'}
        expression_words = {'haha', 'hahaha', 'hahahaha', 'hehe', 'hehehe', 'hihi', 'hihihi', 'huhu', 'huhuhu', 'wkwk', 'wkwkwk', 'lol', 'lmao', 'rofl', 'hmm', 'hmmm', 'wow', 'wahh', 'ahh', 'ohh', 'ehh', 'hiks', 'hikss', 'huft', 'hufft', 'duh', 'aduh', 'yah', 'yahh', 'lahh', 'lahhh'}
        words_to_check = [w for w in words if w not in expression_words]
        if not words_to_check:
            return False
        valid = sum((1 for w in words_to_check if self._is_word_like(w) or w in tech_words))
        ratio = valid / len(words_to_check) if words_to_check else 1.0
        if ratio < 0.2 and len(words_to_check) >= 2:
            return True
        if len(words_to_check) == 1 and len(words_to_check[0]) > 8:
            w = words_to_check[0]
            if w not in self.words and (not self._is_word_like(w)) and (w not in tech_words):
                return True
        for w in words_to_check:
            if w in self.words or w in tech_words or self._is_word_like(w):
                continue
            is_expression = any((re.fullmatch(pat, w) for pat in laugh_patterns))
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
        w = word.lower()
        if w in self.words:
            return True
        slang = {'gue', 'gw', 'gua', 'lo', 'lu', 'elo', 'elu', 'nggak', 'gak', 'ga', 'udah', 'udh', 'aja', 'doang', 'banget', 'bgt', 'emang', 'emg', 'gimana', 'gmn', 'kalo', 'kayak', 'kayaknya', 'kyk', 'ngomong', 'ngapain', 'ngeliat', 'ngerasa', 'nyari', 'nanya', 'nggak', 'ya', 'sih', 'dong', 'deh', 'nih', 'loh', 'lho', 'wkwk', 'haha', 'hihi', 'hehe', 'xixi', 'wkwkwk', 'awkwk', 'btw', 'otw', 'fyi', 'asap', 'lol', 'omg', 'wtf', 'ok', 'oke', 'yoi', 'sip', 'mantap', 'mantul', 'gabut', 'mager', 'ngoding', 'coding', 'nge', 'ngerjain', 'ngomongin', 'ngerti', 'ngga', 'enggak', 'gpp', 'tertawa', 'ketawa', 'ngakak', 'cakep', 'gokil', 'anjir', 'anjay', 'dah', 'bener', 'bgt', 'lahh', 'lahhh', 'yahh', 'yahhh', 'turunan', 'integral', 'kalkulus', 'diferensial', 'logaritma', 'trigonometri', 'sin', 'cos', 'tan', 'limit'}
        if w in slang:
            return True
        if len(w) > 8 and w not in self.words and (w not in slang):
            return False
        vowels = sum((1 for c in w if c in 'aiueo'))
        if len(w) > 0 and vowels / len(w) >= 0.3:
            return True
        return False

    def get_definition(self, word: str) -> str:
        if not self._loaded:
            self.load()
        return self.definitions.get(word.lower().strip(), '')

    def get_all_definitions(self, word: str) -> list:
        if not self._loaded:
            self.load()
        return self.all_definitions.get(word.lower().strip(), [])

    def generate_rich_training_data(self, max_pairs: int=15000) -> list:
        if not self._loaded:
            self.load()
        pairs = []
        words_with_defs = list(self.all_definitions.keys())
        sampled_words = random.sample(words_with_defs, min(max_pairs, len(words_with_defs)))
        pikir_prefixes = ["Mencari makna kata '{word}'...", "Mengingat definisi KBBI untuk '{word}'...", "Menganalisis kata '{word}'...", "Kata '{word}' memiliki arti...", "Berdasarkan kamus, '{word}' berarti...", "Hmm, '{word}' itu...", "Membuka database KBBI untuk '{word}'..."]

        def generate_organic_response(word, defs, classes, kd):
            style = random.choice(['formal', 'santai', 'analitik', 'singkat'])
            pikir = f'<pikir>{random.choice(pikir_prefixes).format(word=word)}</pikir>'
            arti_utama = defs[0]
            if style == 'formal':
                ans = f"Menurut Kamus Besar Bahasa Indonesia (KBBI), kata '{word}' didefinisikan sebagai {arti_utama}."
                if len(defs) > 1:
                    ans += f' Selain itu, kata ini juga dapat bermakna {defs[1]}.'
                if classes:
                    ans += f" Dari segi kelas kata, '{word}' termasuk golongan {', '.join(classes)}."
            elif style == 'santai':
                ans = f"Kata '{word}' itu artinya {arti_utama}. "
                if classes:
                    ans += f'Oh ya, ini tuh termasuk kata {classes[0]} lho.'
                if kd:
                    ans += f" Kata dasarnya dari '{kd[0]}'."
            elif style == 'analitik':
                ans = f"Analisis kata '{word}':\n"
                ans += f'- Makna utama: {arti_utama}\n'
                if len(defs) > 1:
                    ans += f'- Makna sekunder: {defs[1]}\n'
                if classes:
                    ans += f"- Kelas kata: {', '.join(classes)}\n"
                if kd:
                    ans += f'- Akar kata: {kd[0]}'
            else:
                ans = f"'{word}': {arti_utama}."
            if random.random() < 0.3:
                ans += ' ' + random.choice(['📚', '💡', '🤓', '✨', '📖', '📝'])
            return pikir + ans
        for word in sampled_words:
            defs = self.all_definitions.get(word, [])
            if not defs:
                continue
            classes = self.word_classes.get(word, [])
            kd = self.kata_dasar.get(word, [])
            q_variants = [f'Apa arti {word}?', f'{word} artinya apa?', f'Definisi {word} dong', f'Jelaskan makna kata {word}', f'Tolong carikan arti kata {word} di KBBI', f'Kamu tahu arti {word} nggak?', f'Maksud dari {word} itu apa sih?', f'Apa yang dimaksud {word}?', f'Berikan arti kata {word}', f'kata {word} maksudnya apa?', f'{word}?', f'makna {word}', f'kbbi {word}']
            q = random.choice(q_variants)
            if random.random() < 0.2:
                q = q.lower()
            if random.random() < 0.1:
                q = q.replace('?', '')
            a = generate_organic_response(word, defs, classes, kd)
            pairs.append({'input': q, 'response': a, 'emotion': random.choice(['neutral', 'anticipation', 'joy']), 'topic': 'kbbi_mixed', 'preference_update': {'belajar': 1}})
        for idiom, base_word in self.idioms.items():
            q_variants = [f"Apa arti idiom '{idiom}'?", f"Maksud ungkapan '{idiom}' apa?", f'Jelaskan idiom {idiom}', f'ungkapan {idiom} artinya?']
            q = random.choice(q_variants)
            pikir = f"<pikir>Menganalisis kiasan '{idiom}'...</pikir>"
            ans = f"Idiom '{idiom}' adalah kiasan dalam bahasa Indonesia. Karena mengandung kata '{base_word}', maknanya berkaitan dengan hal tersebut, biasanya digunakan untuk menggambarkan situasi tertentu secara kiasan."
            pairs.append({'input': q, 'response': pikir + ans, 'emotion': 'anticipation', 'topic': 'kbbi_idiom', 'preference_update': {}})
        for pb, base_word in self.peribahasa.items():
            q_variants = [f"Apa arti peribahasa '{pb}'?", f"Makna pepatah '{pb}'", f'Jelaskan peribahasa {pb}', f'Tahu arti peribahasa {pb} nggak?']
            q = random.choice(q_variants)
            pikir = f"<pikir>Menerjemahkan makna tersirat dari '{pb}'...</pikir>"
            styles = [f"Peribahasa '{pb}' ini mengajarkan kita tentang suatu kebijaksanaan hidup. Kata kuncinya '{base_word}'.", f"Makna dari pepatah '{pb}' sangat dalam, ini adalah perumpamaan tradisional yang membawa pesan moral.", f"Dalam budaya kita, '{pb}' dipakai untuk menasihati seseorang melalui perumpamaan."]
            ans = random.choice(styles)
            pairs.append({'input': q, 'response': pikir + ans, 'emotion': 'anticipation', 'topic': 'kbbi_peribahasa', 'preference_update': {}})
        words_for_sentences = random.sample(sampled_words, min(2000, len(sampled_words)))
        for word in words_for_sentences:
            defs = self.all_definitions.get(word, [''])[0]
            if len(defs) < 5:
                continue
            q_variants = [f"Buatkan kalimat pakai kata '{word}'", f'Contoh kalimat dengan kata {word}', f'Gunakan {word} dalam kalimat', f'Gimana cara pakai kata {word}?']
            q = random.choice(q_variants)
            subjects = ['Saya', 'Mereka', 'Dia', 'Budi', 'Pemerintah', 'Masyarakat', 'Kita']
            contexts = ['kemarin', 'dengan sangat hati-hati', 'di masa depan', 'dalam rapat tersebut', 'sehari-hari']
            kalimat = f'{random.choice(subjects)} menggunakan prinsip {word} {random.choice(contexts)}.'
            pikir = f"<pikir>Merangkai kalimat organik dengan kata '{word}' yang berarti '{defs[:50]}...'</pikir>"
            ans = f'''Tentu! Mengingat '{word}' artinya '{defs}', berikut contoh penggunaannya:\n\n"{kalimat}"'''
            pairs.append({'input': q, 'response': pikir + ans, 'emotion': 'joy', 'topic': 'kbbi_kalimat', 'preference_update': {}})
        print(f'  📝 Generated {len(pairs)} KBBI training pairs (organic stress-test)')
        return pairs

    def generate_corpus(self) -> str:
        if not self._loaded:
            self.load()
        texts = []
        for word, definition in self.definitions.items():
            texts.append(f'{word} adalah {definition}')
        for word, defs in self.all_definitions.items():
            for d in defs:
                texts.append(f'{word}: {d}')
        return ' '.join(texts)

    def get_all_words(self) -> list:
        if not self._loaded:
            self.load()
        return sorted(list(self.words))