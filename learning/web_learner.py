import json
import os
import re
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse, urljoin
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None
logger = logging.getLogger('spaceax.web_learner')

class ContentExtractor:
    CONTENT_TAGS = ['article', 'main', 'section', 'div']
    REMOVE_TAGS = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript', 'form', 'button', 'input', 'select', 'textarea', 'svg', 'canvas', 'video', 'audio', 'ad', 'advertisement', 'sidebar', 'menu', 'toolbar', 'banner']
    REMOVE_CLASSES = ['nav', 'menu', 'sidebar', 'footer', 'header', 'ad', 'advertisement', 'comment', 'social', 'share', 'related', 'recommended', 'popup', 'modal', 'cookie', 'banner']

    def extract(self, html: str, url: str='') -> dict:
        if BeautifulSoup is None:
            raise ImportError('beautifulsoup4 belum terinstall. Jalankan: pip install beautifulsoup4')
        soup = BeautifulSoup(html, 'html.parser')
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        for cls in self.REMOVE_CLASSES:
            for tag in soup.find_all(class_=re.compile(cls, re.I)):
                tag.decompose()
        title = ''
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text(strip=True)
        content_text = ''
        main_content = soup.find('article') or soup.find('main')
        if main_content:
            content_text = self._extract_text(main_content)
        else:
            best_div = None
            best_length = 0
            for div in soup.find_all('div'):
                text = div.get_text(strip=True)
                if len(text) > best_length:
                    best_length = len(text)
                    best_div = div
            if best_div and best_length > 200:
                content_text = self._extract_text(best_div)
            else:
                body = soup.find('body')
                if body:
                    content_text = self._extract_text(body)
        content_text = self._clean_text(content_text)
        sentences = re.split('[.!?]+', content_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        summary = '. '.join(sentences[:3]) + '.' if sentences else ''
        language = self._detect_language(content_text)
        return {'title': title, 'content': content_text, 'summary': summary, 'word_count': len(content_text.split()), 'language': language, 'url': url, 'extracted_at': datetime.now(timezone.utc).isoformat()}

    def _extract_text(self, element) -> str:
        texts = []
        for child in element.children:
            if hasattr(child, 'name'):
                if child.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote']:
                    text = child.get_text(strip=True)
                    if text:
                        texts.append(text)
                elif child.name in ['br', 'hr']:
                    texts.append('\n')
                elif child.name in ['div', 'section', 'article']:
                    texts.append(self._extract_text(child))
            elif hasattr(child, 'strip'):
                text = child.strip()
                if text:
                    texts.append(text)
        return '\n'.join(texts)

    def _clean_text(self, text: str) -> str:
        text = re.sub('\\s+', ' ', text)
        text = re.sub('\\n{3,}', '\n\n', text)
        text = re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]', '', text)
        text = text.strip()
        return text

    def _detect_language(self, text: str) -> str:
        text_lower = text.lower()
        id_words = ['yang', 'dan', 'dari', 'untuk', 'dengan', 'adalah', 'pada', 'ini', 'itu', 'dalam', 'tidak', 'akan', 'sudah', 'bisa', 'juga', 'atau', 'lebih', 'karena', 'ada', 'mereka', 'telah', 'bahwa', 'oleh', 'setelah', 'seperti', 'saya']
        en_words = ['the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'been', 'will', 'are', 'was', 'were', 'not', 'but', 'they', 'can', 'which', 'about', 'would']
        words = text_lower.split()
        if not words:
            return 'unknown'
        id_count = sum((1 for w in words if w in id_words))
        en_count = sum((1 for w in words if w in en_words))
        if id_count > en_count:
            return 'id'
        elif en_count > id_count:
            return 'en'
        return 'id'

class WebScraper:
    DEFAULT_HEADERS = {'User-Agent': 'SpaceaxAI-Learner/1.0 (Educational AI; +https://github.com/spaceaxai)', 'Accept': 'text/html,application/xhtml+xml', 'Accept-Language': 'id-ID,id;q=0.9,en;q=0.8'}

    def __init__(self, delay: float=2.0, timeout: int=15):
        if requests is None:
            raise ImportError('requests belum terinstall. Jalankan: pip install requests')
        self.delay = delay
        self.timeout = timeout
        self.last_request_time: dict[str, float] = {}
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self.extractor = ContentExtractor()

    def _respect_delay(self, domain: str):
        now = time.time()
        last = self.last_request_time.get(domain, 0)
        wait = self.delay - (now - last)
        if wait > 0:
            time.sleep(wait)
        self.last_request_time[domain] = time.time()

    def fetch(self, url: str) -> Optional[dict]:
        try:
            domain = urlparse(url).netloc
            self._respect_delay(domain)
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                logger.warning(f'Bukan HTML: {content_type} dari {url}')
                return None
            result = self.extractor.extract(response.text, url)
            if result['word_count'] < 50:
                logger.warning(f"Konten terlalu pendek ({result['word_count']} kata) dari {url}")
                return None
            return result
        except Exception as e:
            logger.error(f'Gagal fetch {url}: {e}')
            return None

    def search_and_fetch(self, query: str, max_results: int=5) -> list[dict]:
        results = []
        wiki_results = self._search_wikipedia(query, lang='id', limit=max_results)
        for url in wiki_results:
            content = self.fetch(url)
            if content:
                content['source'] = 'wikipedia_id'
                results.append(content)
        if len(results) < max_results:
            wiki_en = self._search_wikipedia(query, lang='en', limit=max_results - len(results))
            for url in wiki_en:
                content = self.fetch(url)
                if content:
                    content['source'] = 'wikipedia_en'
                    results.append(content)
        return results[:max_results]

    def _search_wikipedia(self, query: str, lang: str='id', limit: int=5) -> list[str]:
        try:
            api_url = f'https://{lang}.wikipedia.org/w/api.php'
            params = {'action': 'query', 'list': 'search', 'srsearch': query, 'srlimit': limit, 'format': 'json', 'utf8': 1}
            domain = urlparse(api_url).netloc
            self._respect_delay(domain)
            response = self.session.get(api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            urls = []
            for item in data.get('query', {}).get('search', []):
                title = item['title'].replace(' ', '_')
                urls.append(f'https://{lang}.wikipedia.org/wiki/{title}')
            return urls
        except Exception as e:
            logger.error(f'Gagal search Wikipedia: {e}')
            return []

class WebLearner:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.knowledge_dir = os.path.join(data_dir, 'knowledge')
        self.learned_file = os.path.join(self.knowledge_dir, 'learned_articles.json')
        self.training_additions_file = os.path.join(data_dir, 'seed', 'web_learned.json')
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'seed'), exist_ok=True)
        self.scraper = WebScraper(delay=2.0)
        self.learned_hashes: set[str] = set()
        self._load_learned_hashes()

    def _load_learned_hashes(self):
        if os.path.exists(self.learned_file):
            try:
                with open(self.learned_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.learned_hashes = set(data.get('hashes', []))
            except Exception:
                self.learned_hashes = set()

    def _save_learned_hash(self, content_hash: str):
        self.learned_hashes.add(content_hash)
        data = {'hashes': list(self.learned_hashes), 'last_updated': datetime.now(timezone.utc).isoformat()}
        with open(self.learned_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _content_hash(self, text: str) -> str:
        return hashlib.md5(text[:500].encode('utf-8')).hexdigest()

    def learn_topic(self, topic: str, max_articles: int=3) -> list[dict]:
        logger.info(f'🌐 Mempelajari topik: {topic}')
        articles = self.scraper.search_and_fetch(topic, max_results=max_articles)
        knowledge_entries = []
        for article in articles:
            content_hash = self._content_hash(article['content'])
            if content_hash in self.learned_hashes:
                logger.info(f"⏭️ Sudah dipelajari: {article['title']}")
                continue
            entry = self._process_article(article, topic)
            if entry:
                knowledge_entries.append(entry)
                self._save_learned_hash(content_hash)
                self._save_knowledge_entry(entry)
                training_pairs = self._generate_training_pairs(entry)
                if training_pairs:
                    self._append_training_data(training_pairs)
                logger.info(f"✅ Dipelajari: {article['title']} ({article['word_count']} kata)")
        return knowledge_entries

    def learn_url(self, url: str) -> Optional[dict]:
        logger.info(f'🌐 Mempelajari URL: {url}')
        article = self.scraper.fetch(url)
        if not article:
            logger.error(f'❌ Gagal fetch: {url}')
            return None
        content_hash = self._content_hash(article['content'])
        if content_hash in self.learned_hashes:
            logger.info(f"⏭️ Sudah dipelajari: {article['title']}")
            return None
        entry = self._process_article(article, '')
        if entry:
            self._save_learned_hash(content_hash)
            self._save_knowledge_entry(entry)
            training_pairs = self._generate_training_pairs(entry)
            if training_pairs:
                self._append_training_data(training_pairs)
            logger.info(f"✅ Dipelajari: {article['title']}")
        return entry

    def _process_article(self, article: dict, topic: str) -> Optional[dict]:
        content = article['content']
        if len(content) > 5000:
            content = content[:5000]
        key_facts = self._extract_key_facts(content)
        categories = self._categorize_content(content, topic)
        entry = {'id': self._content_hash(content), 'title': article['title'], 'content': content, 'summary': article['summary'], 'key_facts': key_facts, 'categories': categories, 'source_url': article.get('url', ''), 'source_type': article.get('source', 'web'), 'language': article.get('language', 'id'), 'word_count': article.get('word_count', 0), 'learned_at': datetime.now(timezone.utc).isoformat(), 'topic': topic}
        return entry

    def _extract_key_facts(self, content: str, max_facts: int=10) -> list[str]:
        sentences = re.split('(?<=[.!?])\\s+', content)
        sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 300]
        if not sentences:
            return []
        scored = []
        for i, sent in enumerate(sentences):
            score = 0.0
            if i < 3:
                score += 2.0
            if re.search('\\d+', sent):
                score += 1.0
            definitive_words = ['adalah', 'merupakan', 'yaitu', 'disebut', 'berarti', 'artinya', 'definisi', 'fungsi', 'tujuan', 'manfaat', 'is', 'are', 'was', 'means', 'defined', 'refers']
            for word in definitive_words:
                if word in sent.lower():
                    score += 1.5
                    break
            word_count = len(sent.split())
            if word_count < 5:
                score -= 1.0
            elif word_count > 40:
                score -= 0.5
            if re.search('[A-Z][a-z]{2,}', sent):
                score += 0.5
            scored.append((score, sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sent for _, sent in scored[:max_facts]]

    def _categorize_content(self, content: str, topic: str) -> list[str]:
        content_lower = content.lower()
        categories = []
        category_keywords = {'sains': ['ilmu', 'penelitian', 'sains', 'science', 'teori', 'eksperimen', 'fisika', 'kimia', 'biologi', 'matematika', 'atom', 'molekul'], 'teknologi': ['teknologi', 'komputer', 'internet', 'digital', 'software', 'hardware', 'robot', 'ai', 'kecerdasan buatan', 'program'], 'sejarah': ['sejarah', 'tahun', 'abad', 'kerajaan', 'perang', 'revolusi', 'kemerdekaan', 'kolonial', 'kuno', 'masa lalu', 'history'], 'geografi': ['pulau', 'benua', 'negara', 'kota', 'sungai', 'gunung', 'lautan', 'iklim', 'wilayah', 'daerah', 'provinsi'], 'budaya': ['budaya', 'tradisi', 'adat', 'seni', 'musik', 'tari', 'upacara', 'festival', 'kuliner', 'bahasa', 'sastra'], 'alam': ['hewan', 'tumbuhan', 'hutan', 'laut', 'ekosistem', 'lingkungan', 'cuaca', 'bumi', 'planet', 'alam'], 'kesehatan': ['kesehatan', 'penyakit', 'obat', 'dokter', 'rumah sakit', 'vitamin', 'gizi', 'nutrisi', 'olahraga', 'tubuh'], 'sosial': ['masyarakat', 'sosial', 'komunitas', 'pendidikan', 'ekonomi', 'politik', 'hukum', 'pemerintah', 'demokrasi'], 'filosofi': ['filsafat', 'filosofi', 'makna', 'eksistensi', 'moral', 'etika', 'kebenaran', 'bijak', 'pemikiran']}
        for category, keywords in category_keywords.items():
            if any((kw in content_lower for kw in keywords)):
                categories.append(category)
        if topic and topic.lower() not in categories:
            categories.append(topic.lower())
        if not categories:
            categories.append('umum')
        return categories

    def _generate_training_pairs(self, entry: dict) -> list[dict]:
        pairs = []
        title = entry['title']
        facts = entry['key_facts']
        summary = entry['summary']
        categories = entry['categories']
        if title and summary:
            pairs.append({'input': f'Apa yang kamu tahu tentang {title}?', 'response': f'Oh, tentang {title} ya! {summary}', 'emotion': 'anticipation', 'topic': categories[0] if categories else 'umum', 'preference_update': {'belajar': 1}})
        for fact in facts[:3]:
            question_templates = [f'Ceritakan fakta menarik tentang {title}.', f'Apa yang menarik dari {title}?', f'Bisa jelaskan tentang {title}?']
            for i, template in enumerate(question_templates):
                if i < len(facts):
                    pairs.append({'input': template, 'response': f'Ini menarik nih! {facts[i]}', 'emotion': 'anticipation', 'topic': categories[0] if categories else 'umum', 'preference_update': {'belajar': 1}})
                    break
        if facts:
            pairs.append({'input': f'Kamu tahu tentang {title} nggak?', 'response': f'Tahu dong! {facts[0]} Mau tahu lebih lanjut?', 'emotion': 'joy', 'topic': categories[0] if categories else 'umum', 'preference_update': {'belajar': 1}})
        return pairs

    def _save_knowledge_entry(self, entry: dict):
        filename = f"knowledge_{entry['id']}.json"
        filepath = os.path.join(self.knowledge_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

    def _append_training_data(self, pairs: list[dict]):
        existing = []
        if os.path.exists(self.training_additions_file):
            try:
                with open(self.training_additions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing = data.get('conversations', [])
            except Exception:
                existing = []
        existing.extend(pairs)
        with open(self.training_additions_file, 'w', encoding='utf-8') as f:
            json.dump({'conversations': existing, 'last_updated': datetime.now(timezone.utc).isoformat(), 'total': len(existing)}, f, ensure_ascii=False, indent=2)

    def get_learning_stats(self) -> dict:
        knowledge_files = [f for f in os.listdir(self.knowledge_dir) if f.startswith('knowledge_') and f.endswith('.json')] if os.path.exists(self.knowledge_dir) else []
        training_count = 0
        if os.path.exists(self.training_additions_file):
            try:
                with open(self.training_additions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    training_count = data.get('total', 0)
            except Exception:
                pass
        return {'articles_learned': len(self.learned_hashes), 'knowledge_entries': len(knowledge_files), 'training_pairs_generated': training_count}

class ScheduledLearner:
    DEFAULT_TOPICS = ['Indonesia', 'Jakarta', 'Budaya Indonesia', 'Sejarah Indonesia', 'Masakan Indonesia', 'Pulau Jawa', 'Bahasa Indonesia', 'Kecerdasan buatan', 'Robot', 'Luar angkasa', 'Planet', 'Fisika kuantum', 'Biologi sel', 'Internet', 'Fotosintesis', 'Evolusi', 'Tata surya', 'Iklim bumi', 'Lautan', 'Gunung berapi', 'Dinosaurus', 'Musik', 'Film', 'Sastra', 'Seni lukis', 'Filsafat', 'Etika', 'Logika', 'Kesehatan', 'Nutrisi', 'Olahraga', 'Pendidikan']

    def __init__(self, web_learner: WebLearner, state_file: str=''):
        self.learner = web_learner
        self.state_file = state_file or os.path.join(web_learner.data_dir, 'knowledge', 'scheduler_state.json')
        self.current_index = 0
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.current_index = data.get('current_index', 0)
            except Exception:
                pass

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({'current_index': self.current_index, 'last_run': datetime.now(timezone.utc).isoformat()}, f, indent=2)

    def learn_next(self, count: int=1) -> list[dict]:
        all_entries = []
        for _ in range(count):
            topic = self.DEFAULT_TOPICS[self.current_index % len(self.DEFAULT_TOPICS)]
            logger.info(f'📚 Scheduled learning: {topic}')
            entries = self.learner.learn_topic(topic, max_articles=2)
            all_entries.extend(entries)
            self.current_index += 1
            self._save_state()
        return all_entries

    def learn_batch(self, topics: list[str]) -> list[dict]:
        all_entries = []
        for topic in topics:
            entries = self.learner.learn_topic(topic, max_articles=2)
            all_entries.extend(entries)
        return all_entries