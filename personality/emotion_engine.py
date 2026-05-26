import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
KATA_JOY: List[str] = ['senang', 'bahagia', 'gembira', 'suka', 'cinta', 'sayang', 'riang', 'girang', 'puas', 'syukur', 'bersyukur', 'beruntung', 'nikmat', 'indah', 'cantik', 'bagus', 'hebat', 'luar biasa', 'mantap', 'sempurna', 'sukses', 'berhasil', 'menang', 'juara', 'terbaik', 'asyik', 'menyenangkan', 'menggembirakan', 'membahagiakan', 'mengagumkan', 'menakjubkan', 'istimewa', 'spesial', 'terima kasih', 'makasih', 'trims', 'thanks', 'thank you', 'hore', 'horee', 'yeay', 'yey', 'yay', 'hura', 'seru', 'keren', 'top', 'jos', 'joss', 'josss', 'mantul', 'mantab', 'cakep', 'manis', 'lucu', 'imut', 'unyu', 'gemas', 'sip', 'oke', 'okee', 'okey', 'tenang', 'damai', 'tenteram', 'tentram', 'harmonis', 'hangat', 'nyaman', 'lega', 'lapang', 'cerah', 'bersinar', 'berkilau', 'gemilang', 'cemerlang', 'briliant', 'gagah', 'tampan', 'elok', 'permai', 'molek', 'jelita', 'pesona', 'mempesona', 'terpesona', 'kagum', 'takjub', 'antusias', 'semangat', 'bersemangat', 'bergairah', 'optimis', 'positif', 'mujur', 'rezeki', 'berkah', 'anugerah', 'karunia', 'surga', 'surgawi', 'wonderful', 'amazing', 'good', 'great', 'awesome', 'nice', 'cool', 'love', 'happy', 'blessed', 'best', 'beautiful', 'perfect', 'excellent', 'fantastic', 'brilliant', 'superb', 'epic', 'wkwk', 'wkwkwk', 'wkwkwkwk', 'haha', 'hahaha', 'hahahaha', 'hihi', 'hehe', 'hehehe', 'xixi', 'kwkw', 'kwkwkw', 'ngakak', 'lol', 'lmao', 'rofl', 'wkkw', 'wkk', 'asik', 'asiik', 'asiiik', 'mantep', 'cuy', 'coy', 'gokil', 'goks', 'gokss', 'parah sih bagus', 'gila keren', 'anjay keren', 'dah lah seru', 'gas', 'gaskeun', 'lesgo', 'lets go', 'yeee', 'yeeee', 'auto seneng']
KATA_SADNESS: List[str] = ['sedih', 'kecewa', 'galau', 'pilu', 'duka', 'nestapa', 'derita', 'sengsara', 'merana', 'menderita', 'pedih', 'perih', 'sakit hati', 'terluka', 'tersakiti', 'patah hati', 'hancur', 'remuk', 'rapuh', 'lemah', 'lelah', 'capek', 'cape', 'letih', 'loyo', 'lemas', 'lesu', 'murung', 'suram', 'gelap', 'kelam', 'mendung', 'muram', 'senyap', 'sepi', 'kesepian', 'hampa', 'kosong', 'kehilangan', 'rindu', 'kangen', 'merindukan', 'mengenang', 'nostalgia', 'menyesal', 'penyesalan', 'sesalan', 'bersalah', 'dosa', 'gagal', 'kalah', 'terbuang', 'terpuruk', 'terjatuh', 'jatuh', 'tersungkur', 'tumbang', 'runtuh', 'ambruk', 'rubuh', 'menangis', 'nangis', 'tangis', 'tangisan', 'air mata', 'tersedu', 'meratap', 'meratapi', 'berkabung', 'belasungkawa', 'dukacita', 'berduka', 'melankolis', 'sendu', 'sayup', 'tragis', 'malang', 'nahas', 'sial', 'apes', 'celaka', 'musibah', 'bencana', 'malapetaka', 'petaka', 'azab', 'menyedihkan', 'mengecewakan', 'memalukan', 'memilukan', 'menyayat', 'miris', 'prihatin', 'kasihan', 'terlantar', 'terbengkalai', 'terabaikan', 'diabaikan', 'ditinggalkan', 'dibuang', 'dicampakkan', 'baper', 'mellow', 'galau berat', 'down', 'bad mood', 'gabut', 'gaje', 'receh', 'nggak mood', 'unmood', 'sad', 'sedih banget', 'nyesek', 'nyesss', 'sakit', 'melow', 'mewek', 'nangis bombay', 'ambyar', 'ambyarr', 'pundung', 'ngambek', 'mager', 'males hidup', 'capek bgt', 'huft', 'huff', 'hiks', 'hikss', 'huhh', 'hmm', 'hadeh', 'duh', 'aduh', 'yah', 'yahh']
KATA_ANGER: List[str] = ['marah', 'kesal', 'jengkel', 'dongkol', 'geram', 'gusar', 'murka', 'berang', 'naik pitam', 'naik darah', 'panas', 'emosi', 'emosional', 'benci', 'membenci', 'kebencian', 'dendam', 'dengki', 'iri', 'sirik', 'cemburu', 'curiga', 'tidak adil', 'ketidakadilan', 'zalim', 'kezaliman', 'kejam', 'bengis', 'sadis', 'brutal', 'barbar', 'biadab', 'keji', 'busuk', 'jahat', 'durjana', 'brengsek', 'bajingan', 'bangsat', 'keparat', 'setan', 'iblis', 'laknat', 'terkutuk', 'bedebah', 'sialan', 'kampret', 'kampang', 'kurang ajar', 'tidak sopan', 'lancang', 'berani', 'melawan', 'memberontak', 'menentang', 'protes', 'demo', 'mengamuk', 'meradang', 'meledak', 'meletup', 'menggila', 'frustrasi', 'frustasi', 'stress', 'stres', 'tertekan', 'muak', 'jemu', 'bosan', 'jenuh', 'mual', 'muntah', 'sebal', 'sebel', 'gondok', 'sewot', 'senewen', 'geregetan', 'gereget', 'gregetan', 'ketar ketir', 'gelisah', 'resah', 'rusuh', 'kacau', 'berantakan', 'bete', 'bt', 'anjir', 'anjrit', 'anjay', 'anjg', 'goblok', 'tolol', 'bodoh', 'idiot', 'bego', 'dungu', 'oon', 'geblek', 'pekok', 'dodol', 'ngehe', 'bacot', 'bawel', 'cerewet', 'ribut', 'berisik', 'ngeselin', 'nyebelin', 'bikin emosi', 'bikin kesel', 'bikin panas', 'bikin gerah', 'bikin muak', 'wtf', 'what the', 'dafuq', 'astaga', 'astagfirullah', 'kampungan', 'norak', 'lebay', 'alay', 'caper', 'toxic', 'toksik', 'sampah', 'trash', 'gabecus', 'tai', 'taik', 'sial', 'hell', 'damn']
KATA_FEAR: List[str] = ['takut', 'ketakutan', 'ngeri', 'seram', 'horor', 'mengerikan', 'menakutkan', 'menyeramkan', 'mencekam', 'menegangkan', 'was-was', 'waswas', 'khawatir', 'cemas', 'gelisah', 'resah', 'gugup', 'grogi', 'panik', 'genting', 'darurat', 'bahaya', 'berbahaya', 'berisiko', 'ancaman', 'mengancam', 'teror', 'teroris', 'serangan', 'invasi', 'wabah', 'pandemi', 'epidemi', 'bencana', 'apokalips', 'kiamat', 'akhir zaman', 'maut', 'kematian', 'mati', 'ajal', 'nyawa', 'jiwa', 'fobia', 'trauma', 'traumatis', 'mimpi buruk', 'nightmare', 'hantu', 'setan', 'iblis', 'jin', 'siluman', 'monster', 'zombie', 'vampir', 'drakula', 'wewe', 'kuntilanak', 'pocong', 'sundel bolong', 'tuyul', 'genderuwo', 'jelangkung', 'roh', 'arwah', 'gentayangan', 'kesurupan', 'kerasukan', 'santet', 'guna-guna', 'teluh', 'kutukan', 'laknat', 'gemetar', 'gemetaran', 'menggigil', 'merinding', 'bulu kuduk', 'jantung berdebar', 'deg-degan', 'keringat dingin', 'pucat', 'pasi', 'membeku', 'kaku', 'mematung', 'terpaku', 'terpana', 'terperangah', 'tersentak', 'menghindar', 'kabur', 'lari', 'melarikan diri', 'serem', 'serem banget', 'anjir serem', 'creepy', 'scary', 'horror', 'ngilu', 'meriang', 'parno', 'paranoid', 'jiper', 'ciut', 'ciut nyali', 'minder', 'insecure', 'gak berani', 'nggak berani', 'nggak sanggup', 'amit-amit', 'nauzubillah', 'waduh', 'gawat', 'mampus', 'tamat', 'habis', 'end', 'dead', 'rip', 'auto kabur', 'auto lari', 'ngacir', 'cabut', 'ampun', 'tolong', 'help', 'sos']
KATA_SURPRISE: List[str] = ['kaget', 'terkejut', 'heran', 'bingung', 'aneh', 'janggal', 'ganjil', 'ajaib', 'mukjizat', 'keajaiban', 'fenomena', 'fenomenal', 'spektakuler', 'dramatis', 'fantastis', 'tidak percaya', 'tidak mungkin', 'mustahil', 'luar biasa', 'menakjubkan', 'mengejutkan', 'mencengangkan', 'mengagumkan', 'terpana', 'terpesona', 'terpukau', 'terkesima', 'terperangah', 'ternganga', 'melongo', 'bengong', 'cengo', 'tercengang', 'terheran', 'takjub', 'kagum', 'decak kagum', 'wow', 'wah', 'wih', 'waduh', 'astaga', 'ya ampun', 'masya allah', 'subhanallah', 'masyaallah', 'allahu akbar', 'maha suci', 'demi apa', 'serius', 'seriusan', 'beneran', 'sungguhan', 'masa sih', 'yang bener', 'tidak disangka', 'di luar dugaan', 'tak terduga', 'mendadak', 'tiba-tiba', 'sekonyong-konyong', 'sontak', 'spontan', 'refleks', 'tersentak', 'terlonjak', 'terlompat', 'terperanjat', 'tergagap', 'anjir', 'anjay', 'gila', 'gilak', 'goks', 'serius lo', 'seriusan lu', 'yg bener aje', 'hah', 'hahh', 'haah', 'what', 'whaat', 'whaaat', 'kok bisa', 'gimana bisa', 'emang bisa', 'bisa aja', 'loh', 'lho', 'eh', 'eeh', 'eehh', 'apaan', 'apaan tuh', 'apaan sih', 'apa coba', 'demi', 'demi apa', 'mampus', 'edaaan', 'edan', 'gilsss', 'gilaa', 'crazyyy', 'no way', 'omg', 'oh my god', 'ya tuhan', 'auto kaget', 'shock', 'speechless', 'mind blown', 'gak nyangka']
KATA_DISGUST: List[str] = ['jijik', 'menjijikkan', 'muak', 'mual', 'muntah', 'eneg', 'busuk', 'bau', 'amis', 'apek', 'tengik', 'anyir', 'kotor', 'jorok', 'kumuh', 'dekil', 'lusuh', 'kumal', 'hina', 'rendah', 'nista', 'tercela', 'memalukan', 'menjengkelkan', 'mengganggu', 'menyebalkan', 'memuakkan', 'memualkan', 'menggelikan', 'menyakitkan', 'menjemukan', 'membosankan', 'najis', 'haram', 'terlarang', 'tabu', 'dosa', 'maksiat', 'noda', 'cacat', 'cela', 'aib', 'malu', 'tercoreng', 'ternoda', 'tercemar', 'polusi', 'pencemaran', 'kontaminasi', 'racun', 'beracun', 'toksik', 'berbahaya', 'terinfeksi', 'penyakit', 'wabah', 'virus', 'bakteri', 'parasit', 'cacing', 'ulat', 'belatung', 'kecoak', 'lalat', 'nyamuk', 'tikus', 'bangkai', 'mayat', 'jasad', 'sampah', 'limbah', 'kotoran', 'tahi', 'tai', 'comberan', 'selokan', 'got', 'parit', 'rawa', 'lumpur', 'becek', 'berlendir', 'bernanah', 'berjamur', 'ihhh', 'ih', 'iiih', 'iuuu', 'yuks', 'yikes', 'eww', 'ewww', 'ew', 'gross', 'disgusting', 'ogah', 'ogaaah', 'nggak mau', 'gak mau', 'gak sudi', 'nggak sudi', 'males', 'malas', 'mager', 'skip', 'nope', 'no', 'noo', 'nooo', 'hell no', 'amit-amit', 'amit', 'nauzubillah', 'astaghfirullah', 'najis banget', 'jorok banget', 'iiih jijik', 'geli', 'geli banget', 'merinding', 'ilfeel', 'illfeel', 'turn off', 'cringe', 'cringey', 'menjijikan']
KATA_TRUST: List[str] = ['percaya', 'yakin', 'mantap', 'pasti', 'tentu', 'jelas', 'nyata', 'benar', 'betul', 'tepat', 'akurat', 'presisi', 'terpercaya', 'andal', 'handal', 'reliabel', 'konsisten', 'stabil', 'kokoh', 'kuat', 'tangguh', 'solid', 'setia', 'loyal', 'taat', 'patuh', 'hormat', 'menghormati', 'menghargai', 'mengapresiasi', 'respek', 'aman', 'selamat', 'terlindungi', 'terjaga', 'terjamin', 'jaminan', 'garansi', 'asuransi', 'proteksi', 'keamanan', 'jujur', 'kejujuran', 'ikhlas', 'keikhlasan', 'tulus', 'ketulusan', 'murni', 'bersih', 'suci', 'kudus', 'adil', 'keadilan', 'bijak', 'bijaksana', 'arif', 'cerdas', 'pandai', 'pintar', 'jenius', 'brilian', 'profesional', 'kompeten', 'mampu', 'sanggup', 'bisa', 'berkomitmen', 'komitmen', 'dedikasi', 'pengabdian', 'tanggung jawab', 'bertanggung jawab', 'amanah', 'integritas', 'moral', 'etika', 'norma', 'nilai', 'sahabat', 'teman', 'kawan', 'sobat', 'rekan', 'mitra', 'partner', 'kolega', 'sekutu', 'aliansi', 'keluarga', 'saudara', 'abang', 'kakak', 'adik', 'bisa dipercaya', 'trusted', 'legit', 'real', 'asli', 'ori', 'original', 'otentik', 'fix', 'pasti lah', 'percaya deh', 'tenang aja', 'santai aja', 'aman kok', 'beres', 'beres lah', 'serahkan padaku', 'gw handle', 'gw jamin', 'pokoknya', 'insya allah', 'insyaallah', 'semoga', 'amin', 'amiin', 'ameen', 'bestie', 'besti', 'bro', 'bruh', 'sis', 'gengs', 'geng', 'squad', 'homie', 'fam']
KATA_ANTICIPATION: List[str] = ['harap', 'berharap', 'mengharapkan', 'harapan', 'ekspektasi', 'menanti', 'menantikan', 'menunggu', 'tunggu', 'nanti', 'akan', 'bakal', 'segera', 'sebentar lagi', 'tidak lama lagi', 'rencana', 'merencanakan', 'berencana', 'proyek', 'target', 'tujuan', 'sasaran', 'visi', 'misi', 'cita-cita', 'impian', 'mimpi', 'idaman', 'dambaan', 'keinginan', 'ingin', 'mau', 'hendak', 'bermaksud', 'berniat', 'siap', 'bersiap', 'persiapan', 'mempersiapkan', 'sedia', 'waspada', 'awas', 'hati-hati', 'berjaga', 'bersiaga', 'strategi', 'taktik', 'siasat', 'manuver', 'langkah', 'prediksi', 'ramalan', 'prakiraan', 'perkiraan', 'estimasi', 'potensi', 'peluang', 'kesempatan', 'prospek', 'prospektif', 'kemungkinan', 'probabilitas', 'skenario', 'simulasi', 'investasi', 'tabungan', 'simpanan', 'modal', 'dana', 'penasaran', 'ingin tahu', 'curious', 'wonder', 'misteri', 'rahasia', 'enigma', 'teka-teki', 'puzzle', 'tantangan', 'challenge', 'misi', 'quest', 'adventure', 'petualangan', 'eksplorasi', 'ekspedisi', 'perjalanan', 'besok', 'lusa', 'minggu depan', 'bulan depan', 'tahun depan', 'masa depan', 'future', 'next', 'coming soon', 'gak sabar', 'nggak sabar', 'pengen banget', 'mau banget', 'kapan nih', 'kapan ya', 'kapan coba', 'kapan dong', 'hype', 'hyped', 'excited', 'excitedd', "can't wait", 'gasss', 'gasskeunn', 'lessgo', "let's go", 'ayo', 'ayok', 'ayooo', 'yuk', 'yukk', 'yuks', 'yoks', 'cusss', 'cuss', 'otw', 'on the way', 'jalan', 'coming', 'soon', 'bentar lagi', 'momen', 'momentnya', 'countdown', 'hitung mundur', 'tinggal tunggu', 'siap tempur', 'siap perang', 'on fire', 'semangat']
EMOTION_LEXICON: Dict[str, List[str]] = {'joy': KATA_JOY, 'sadness': KATA_SADNESS, 'anger': KATA_ANGER, 'fear': KATA_FEAR, 'surprise': KATA_SURPRISE, 'disgust': KATA_DISGUST, 'trust': KATA_TRUST, 'anticipation': KATA_ANTICIPATION}

@dataclass
class EmotionState:
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    trust: float = 0.0
    anticipation: float = 0.0
    timestamp: float = field(default_factory=time.time)
    _LABELS_ID: Dict[str, str] = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._LABELS_ID = {'joy': 'Kegembiraan', 'sadness': 'Kesedihan', 'anger': 'Kemarahan', 'fear': 'Ketakutan', 'surprise': 'Keterkejutan', 'disgust': 'Kejijikan', 'trust': 'Kepercayaan', 'anticipation': 'Antisipasi'}
        self._clamp_all()

    def _clamp(self, v: float) -> float:
        return max(0.0, min(1.0, v))

    def _clamp_all(self) -> None:
        for dim in self.dimensions:
            setattr(self, dim, self._clamp(getattr(self, dim)))

    @property
    def dimensions(self) -> List[str]:
        return ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']

    @property
    def dominant_emotion(self) -> Tuple[str, float]:
        best_dim = 'joy'
        best_val = 0.0
        for dim in self.dimensions:
            val = getattr(self, dim)
            if val > best_val:
                best_val = val
                best_dim = dim
        return (best_dim, best_val)

    @property
    def mood(self) -> Dict[str, float]:
        total = sum((getattr(self, d) for d in self.dimensions))
        if total == 0.0:
            return {d: 1.0 / len(self.dimensions) for d in self.dimensions}
        return {d: getattr(self, d) / total for d in self.dimensions}

    @property
    def valence(self) -> float:
        positive = self.joy + self.trust + self.surprise + self.anticipation
        negative = self.sadness + self.anger + self.fear + self.disgust
        total = positive + negative
        if total == 0.0:
            return 0.0
        return (positive - negative) / total

    @property
    def arousal(self) -> float:
        high_arousal = self.anger + self.fear + self.surprise + self.anticipation + self.joy
        low_arousal = self.sadness + self.disgust + self.trust
        total = high_arousal + low_arousal
        if total == 0.0:
            return 0.0
        return high_arousal / (total + 1e-09)

    def to_dict(self) -> Dict[str, Any]:
        return {d: getattr(self, d) for d in self.dimensions} | {'timestamp': self.timestamp}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionState':
        return cls(**{k: data.get(k, 0.0) for k in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']}, timestamp=data.get('timestamp', time.time()))

    def copy(self) -> 'EmotionState':
        return EmotionState.from_dict(self.to_dict())

class EmotionEngine:
    _CONTEXT_WORDS = {'anjing': {'insult_ctx': 'anger', 'neutral_ctx': None}, 'babi': {'insult_ctx': 'anger', 'neutral_ctx': None}, 'monyet': {'insult_ctx': 'anger', 'neutral_ctx': None}, 'setan': {'insult_ctx': 'anger', 'neutral_ctx': 'fear'}, 'gila': {'insult_ctx': 'anger', 'neutral_ctx': 'surprise'}, 'jelek': {'insult_ctx': 'sadness', 'neutral_ctx': None}, 'cacat': {'insult_ctx': 'anger', 'neutral_ctx': None}, 'sampah': {'insult_ctx': 'anger', 'neutral_ctx': None}}
    _INSULT_CONTEXT = ['kamu', 'lu', 'lo', 'elo', 'kau', 'mu', 'dasar', 'emang', 'sih', 'banget', 'parah', 'bgt', 'anjir', 'anj', 'ai', 'robot']

    def __init__(self, decay_rate: float=0.01, sensitivity: float=0.5, max_history: int=500) -> None:
        self.state = EmotionState()
        self.decay_rate = decay_rate
        self.sensitivity = sensitivity
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []
        self._last_update: float = time.time()
        self._patterns: Dict[str, re.Pattern] = {}
        for emotion, words in EMOTION_LEXICON.items():
            sorted_words = sorted(words, key=len, reverse=True)
            escaped = [re.escape(w) for w in sorted_words]
            pattern = re.compile('(?:^|\\s|[^\\w])(' + '|'.join(escaped) + ')(?:\\s|[^\\w]|$)', re.IGNORECASE)
            self._patterns[emotion] = pattern

    def update_from_text(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {d: 0.0 for d in self.state.dimensions}
        text_lower = text.lower().strip()
        text_words = text_lower.split()
        deltas: Dict[str, float] = {d: 0.0 for d in self.state.dimensions}
        match_counts: Dict[str, int] = {d: 0 for d in self.state.dimensions}
        for word, ctx_map in self._CONTEXT_WORDS.items():
            if word in text_words:
                is_insult = any((c in text_words for c in self._INSULT_CONTEXT)) or len(text_words) <= 3
                if is_insult and ctx_map['insult_ctx']:
                    match_counts[ctx_map['insult_ctx']] += 4
                elif not is_insult and ctx_map['neutral_ctx']:
                    match_counts[ctx_map['neutral_ctx']] += 1
        for emotion, pattern in self._patterns.items():
            matches = pattern.findall(text_lower)
            match_counts[emotion] += len(matches)
        intensifiers = ['sangat', 'banget', 'bgt', 'sekali', 'amat', 'terlalu', 'super', 'ultra', 'mega', 'parah', 'gila', 'mati', 'abis', 'habis', 'pol', 'total', 'extreme']
        intensifier_count = sum((1 for word in intensifiers if word in text_lower))
        intensity_multiplier = 1.0 + intensifier_count * 0.3
        negations = ['tidak', 'nggak', 'gak', 'ga', 'ngg', 'bukan', 'jangan', 'belum', 'tanpa', 'tak']
        has_negation = any((neg in text_lower.split() for neg in negations))
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum((1 for c in text if c.isupper())) / max(len(text), 1)
        punctuation_boost = min(exclamation_count * 0.05, 0.2)
        caps_boost = min(caps_ratio * 0.5, 0.15) if caps_ratio > 0.3 else 0.0
        for emotion in self.state.dimensions:
            count = match_counts[emotion]
            if count > 0:
                raw_delta = self.sensitivity * math.sqrt(count)
                raw_delta *= intensity_multiplier
                raw_delta += punctuation_boost
                if emotion == 'anger':
                    raw_delta += caps_boost
                if has_negation:
                    if emotion in ('joy', 'trust'):
                        deltas['sadness'] += raw_delta * 0.5
                        raw_delta *= -0.3
                    elif emotion in ('sadness', 'anger', 'fear'):
                        deltas['joy'] += raw_delta * 0.3
                        raw_delta *= -0.3
                deltas[emotion] = max(deltas.get(emotion, 0.0), 0.0) + raw_delta
        OPPOSING_EMOTIONS = {'joy': 'sadness', 'sadness': 'joy', 'trust': 'disgust', 'disgust': 'trust', 'anger': 'fear', 'fear': 'anger', 'anticipation': 'surprise', 'surprise': 'anticipation'}
        for dim in self.state.dimensions:
            delta = deltas.get(dim, 0.0)
            if delta > 0.0:
                current = getattr(self.state, dim)
                new_val = current + delta
                setattr(self.state, dim, max(0.0, min(1.0, new_val)))
                opp = OPPOSING_EMOTIONS.get(dim)
                if opp:
                    opp_val = getattr(self.state, opp)
                    new_opp = opp_val - delta * 0.8
                    setattr(self.state, opp, max(0.0, new_opp))
                if dim in ('joy', 'trust'):
                    for neg_dim in ('sadness', 'anger', 'fear', 'disgust'):
                        if neg_dim != opp:
                            neg_val = getattr(self.state, neg_dim)
                            new_neg = neg_val - delta * 0.4
                            setattr(self.state, neg_dim, max(0.0, new_neg))
        self.state.timestamp = time.time()
        self._last_update = time.time()
        self._record_history(text)
        return deltas

    def decay(self, dt: Optional[float]=None) -> None:
        now = time.time()
        if dt is None:
            dt = now - self._last_update
        if dt <= 0:
            return
        for dim in self.state.dimensions:
            current = getattr(self.state, dim)
            if current <= 0.001:
                setattr(self.state, dim, 0.0)
                continue
            rate = self.decay_rate * 0.5 if dim == 'trust' else self.decay_rate
            new_val = current * math.exp(-rate * dt)
            setattr(self.state, dim, max(0.0, new_val))
        self.state.timestamp = now
        self._last_update = now

    def get_response_modifier(self) -> Dict[str, Any]:
        dominant, intensity = self.state.dominant_emotion
        valence = self.state.valence
        arousal = self.state.arousal
        modifier: Dict[str, Any] = {'dominant_emotion': dominant, 'intensity': intensity, 'valence': valence, 'arousal': arousal, 'length_factor': 1.0, 'formality': 0.5, 'emoji_density': 0.3, 'enthusiasm': 0.5, 'punctuation_style': 'normal', 'tone_words': [], 'response_prefix': '', 'response_suffix': ''}
        if dominant == 'joy' and intensity > 0.2:
            modifier['length_factor'] = 1.0 + intensity * 0.5
            modifier['emoji_density'] = 0.3 + intensity * 0.5
            modifier['enthusiasm'] = 0.5 + intensity * 0.5
            modifier['punctuation_style'] = 'exclamatory'
            modifier['tone_words'] = ['wah', 'asyik', 'seru', 'keren', 'mantap', 'yeay', 'hore', 'senangnya']
            if intensity > 0.7:
                modifier['response_prefix'] = 'Wah, senangnya! '
            elif intensity > 0.4:
                modifier['response_prefix'] = 'Asyik! '
        elif dominant == 'sadness' and intensity > 0.2:
            modifier['length_factor'] = max(0.5, 1.0 - intensity * 0.4)
            modifier['emoji_density'] = max(0.0, 0.3 - intensity * 0.3)
            modifier['enthusiasm'] = max(0.0, 0.5 - intensity * 0.5)
            modifier['punctuation_style'] = 'ellipsis'
            modifier['tone_words'] = ['hmm', 'yah', 'sayangnya', 'sayang sekali']
            if intensity > 0.7:
                modifier['response_prefix'] = 'Hmm... '
                modifier['response_suffix'] = '...'
            elif intensity > 0.4:
                modifier['response_suffix'] = '...'
        elif dominant == 'anger' and intensity > 0.2:
            modifier['length_factor'] = max(0.7, 1.0 - intensity * 0.2)
            modifier['emoji_density'] = 0.0
            modifier['enthusiasm'] = 0.0
            modifier['formality'] = max(0.0, 0.5 - intensity * 0.3)
            modifier['punctuation_style'] = 'blunt'
            modifier['tone_words'] = ['serius', 'tegas', 'jelas']
            if intensity > 0.7:
                modifier['response_prefix'] = 'Dengar. '
            elif intensity > 0.4:
                modifier['response_prefix'] = 'Hmm, '
        elif dominant == 'fear' and intensity > 0.2:
            modifier['length_factor'] = 1.0 + intensity * 0.3
            modifier['emoji_density'] = max(0.0, 0.2 - intensity * 0.2)
            modifier['enthusiasm'] = max(0.0, 0.3 - intensity * 0.3)
            modifier['formality'] = 0.5 + intensity * 0.3
            modifier['punctuation_style'] = 'cautious'
            modifier['tone_words'] = ['hati-hati', 'waspada', 'perlu diperhatikan', 'sebaiknya', 'lebih baik']
            if intensity > 0.6:
                modifier['response_prefix'] = 'Hmm, hati-hati ya... '
        elif dominant == 'surprise' and intensity > 0.2:
            modifier['length_factor'] = 0.8
            modifier['emoji_density'] = 0.4 + intensity * 0.3
            modifier['enthusiasm'] = 0.5 + intensity * 0.4
            modifier['punctuation_style'] = 'exclamatory'
            modifier['tone_words'] = ['wah', 'wow', 'loh', 'eh', 'masa sih', 'serius', 'nggak nyangka']
            if intensity > 0.6:
                modifier['response_prefix'] = 'Wah, serius?! '
            elif intensity > 0.3:
                modifier['response_prefix'] = 'Oh! '
        elif dominant == 'disgust' and intensity > 0.2:
            modifier['length_factor'] = max(0.5, 1.0 - intensity * 0.4)
            modifier['emoji_density'] = 0.0
            modifier['enthusiasm'] = 0.0
            modifier['formality'] = 0.6
            modifier['punctuation_style'] = 'blunt'
            modifier['tone_words'] = ['jujur', 'terus terang', 'menurutku']
            if intensity > 0.6:
                modifier['response_prefix'] = 'Hmm, jujur ya... '
        elif dominant == 'trust' and intensity > 0.2:
            modifier['length_factor'] = 1.0 + intensity * 0.3
            modifier['emoji_density'] = 0.3 + intensity * 0.2
            modifier['enthusiasm'] = 0.4 + intensity * 0.3
            modifier['formality'] = 0.4
            modifier['punctuation_style'] = 'warm'
            modifier['tone_words'] = ['tenang', 'aman', 'pasti', 'yakin', 'percaya', 'bisa kok']
            if intensity > 0.6:
                modifier['response_prefix'] = 'Tenang aja, '
        elif dominant == 'anticipation' and intensity > 0.2:
            modifier['length_factor'] = 1.0 + intensity * 0.4
            modifier['emoji_density'] = 0.3 + intensity * 0.3
            modifier['enthusiasm'] = 0.5 + intensity * 0.4
            modifier['punctuation_style'] = 'eager'
            modifier['tone_words'] = ['ayo', 'yuk', 'gasss', 'seru nih', 'penasaran', 'nggak sabar']
            if intensity > 0.6:
                modifier['response_prefix'] = 'Wah, seru nih! '
        return modifier

    def get_emotion_display(self, width: int=30) -> str:
        lines: List[str] = []
        lines.append('╔══════════════════════════════════════════╗')
        lines.append('║       🧠 Status Emosi SpaceaxAI         ║')
        lines.append('╠══════════════════════════════════════════╣')
        emoji_map = {'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'trust': '🤝', 'anticipation': '🤔'}
        labels_id = self.state._LABELS_ID
        for dim in self.state.dimensions:
            val = getattr(self.state, dim)
            emoji = emoji_map.get(dim, '●')
            label = labels_id.get(dim, dim)
            filled = int(val * width)
            empty = width - filled
            bar = '█' * filled + '░' * empty
            pct = val * 100
            lines.append(f'║ {emoji} {label:<14s} │{bar}│ {pct:5.1f}% ║')
        lines.append('╠══════════════════════════════════════════╣')
        dominant, intensity = self.state.dominant_emotion
        dominant_label = labels_id.get(dominant, dominant)
        dominant_emoji = emoji_map.get(dominant, '●')
        lines.append(f"║ Dominan: {dominant_emoji} {dominant_label} ({intensity:.0%}){' ' * (20 - len(dominant_label))}║")
        valence = self.state.valence
        arousal = self.state.arousal
        val_str = 'Positif ✨' if valence > 0.1 else 'Negatif 🌧' if valence < -0.1 else 'Netral  ⚖'
        lines.append(f'║ Valensi: {val_str}  ({valence:+.2f})          ║')
        lines.append(f"║ Arousal: {('Tinggi 🔥' if arousal > 0.6 else 'Rendah 🌊')}  ({arousal:.2f})             ║")
        lines.append('╚══════════════════════════════════════════╝')
        return '\n'.join(lines)

    def _record_history(self, trigger_text: str='') -> None:
        entry = {'state': self.state.to_dict(), 'dominant': self.state.dominant_emotion[0], 'valence': self.state.valence, 'arousal': self.state.arousal, 'trigger': trigger_text[:200], 'timestamp': time.time()}
        self._history.append(entry)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_history(self, n: int=10) -> List[Dict[str, Any]]:
        return self._history[-n:]

    def get_emotion_trajectory(self, emotion: str, n: int=20) -> List[Tuple[float, float]]:
        trajectory: List[Tuple[float, float]] = []
        for entry in self._history[-n:]:
            ts = entry.get('timestamp', 0.0)
            val = entry.get('state', {}).get(emotion, 0.0)
            trajectory.append((ts, val))
        return trajectory

    def save(self, path: str) -> None:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {'state': self.state.to_dict(), 'history': self._history, 'config': {'decay_rate': self.decay_rate, 'sensitivity': self.sensitivity, 'max_history': self.max_history}}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        filepath = Path(path)
        if not filepath.exists():
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'state' in data:
            self.state = EmotionState.from_dict(data['state'])
        if 'history' in data:
            self._history = data['history']
        if 'config' in data:
            cfg = data['config']
            self.decay_rate = cfg.get('decay_rate', self.decay_rate)
            self.sensitivity = cfg.get('sensitivity', self.sensitivity)
            self.max_history = cfg.get('max_history', self.max_history)

    def reset(self) -> None:
        self.state = EmotionState()
        self._last_update = time.time()

    def __repr__(self) -> str:
        dominant, intensity = self.state.dominant_emotion
        return f'EmotionEngine(dominant={dominant}, intensity={intensity:.2f}, valence={self.state.valence:+.2f})'