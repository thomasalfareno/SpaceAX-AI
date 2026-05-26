import json
import os
import requests
import re
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
from ddgs import DDGS
from datetime import datetime, timezone

class InternetLearner:
    TRUSTED_DOMAINS = ['wikipedia.org', 'id.wikipedia.org', 'kompas.com', 'detik.com', 'liputan6.com', 'tribunnews.com', 'cnnindonesia.com', 'tempo.co', 'bbc.com', 'bbc.co.uk', 'britannica.com']
    SPAM_KEYWORDS = ['citation', 'bibliography', 'apa style', 'mla style', 'chicago style', 'apa format', 'mla format', 'apa citation', 'mla citation', 'cite this', 'citation generator', 'citation machine', 'works cited', 'in-text citation', 'reference list', 'grammarly', 'plagiarism', 'paraphrase', 'paraphrasing', 'rewrite', 'rewriter', 'turnitin', 'quillbot', 'wordtune', 'plagiarism checker', 'grammar checker', 'ads', 'sponsored', 'adwords']
    SPAM_DOMAINS = ['scribbr.com', 'easybib.com', 'bibme.com', 'citationmachine.net', 'citethisforme.com', 'mybib.com', 'citefast.com', 'formatically.com', 'grafiati.com', 'zbib.org', 'quillbot.com', 'grammarly.com', 'turnitin.com', 'wordtune.com']

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.db_path = os.path.join(knowledge_dir, 'internet_db.json')
        self.knowledge_base = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_db(self):
        os.makedirs(self.knowledge_dir, exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)

    def clean_search_query(self, query: str) -> str:
        q = query.lower().strip()
        q = re.sub('^!search\\s+', '', q)
        phrases_to_remove = ['apa arti dari', 'apa arti', 'arti dari', 'apa itu', 'siapa itu', 'apa sih', 'tolong carikan tentang', 'tolong cari tentang', 'cari tentang', 'carikan tentang', 'tolong cari', 'tolong carikan', 'cari info', 'cari tahu', 'tahukah kamu', 'siapakah', 'jelaskan tentang', 'jelaskan maksud']
        for phrase in phrases_to_remove:
            q = q.replace(phrase, '')
        q = re.sub('[?!\\.]+$', '', q)
        return q.strip()

    def _is_spam(self, href: str, body: str) -> bool:
        href_lower = href.lower()
        body_lower = body.lower()
        for domain in self.SPAM_DOMAINS:
            if domain in href_lower:
                return True
        for keyword in self.SPAM_KEYWORDS:
            if keyword in href_lower or keyword in body_lower:
                return True
        return False

    def _score_result(self, result: dict, query_keywords: list) -> int:
        score = 0
        href = result.get('href', '').lower()
        body = result.get('body', '').lower()
        title = result.get('title', '').lower()
        for domain in self.TRUSTED_DOMAINS:
            if domain in href:
                score += 20
                break
        for kw in query_keywords:
            if kw in body:
                score += 5
            if kw in title:
                score += 3
        body_len = len(result.get('body', ''))
        if body_len > 150:
            score += 5
        elif body_len > 80:
            score += 2
        return score

    def search_and_learn(self, query: str) -> str:
        cleaned_query = self.clean_search_query(query)
        if not cleaned_query:
            cleaned_query = query
        query_key = query.lower().strip()
        if query_key in self.knowledge_base:
            return self.knowledge_base[query_key]['summary']
        print(f"\n[🌐 AI mengakses Internet untuk: '{cleaned_query}']...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with DDGS() as ddgs:
                    raw_results = list(ddgs.text(cleaned_query, max_results=8, region='id-id'))
            if not raw_results:
                return 'Maaf, aku tidak menemukan informasi relevan di internet saat ini.'
            filtered_results = []
            for r in raw_results:
                href = r.get('href', '')
                body = r.get('body', '')
                if self._is_spam(href, body):
                    continue
                filtered_results.append(r)
            if not filtered_results:
                filtered_results = raw_results[:2]
            query_keywords = [w for w in cleaned_query.lower().split() if len(w) > 2]
            filtered_results.sort(key=lambda r: self._score_result(r, query_keywords), reverse=True)
            results = filtered_results[:3]
            summaries = []
            sources = []
            for i, r in enumerate(results):
                title = r.get('title', 'Sumber Informasi')
                href = r.get('href', '')
                body = r.get('body', '').strip()
                if body:
                    summaries.append(body)
                if href:
                    clean_title = re.sub('[^\\w\\s\\-]', '', title)[:40].strip()
                    sources.append(f'[{clean_title}]({href})')
            combined_text = ' '.join(summaries)
            max_len = 500
            if len(combined_text) > max_len:
                truncated = combined_text[:max_len]
                last_dot = truncated.rfind('.')
                if last_dot > 100:
                    summary_text = truncated[:last_dot + 1]
                else:
                    summary_text = truncated + '...'
            else:
                summary_text = combined_text
            source_links = ', '.join(sources)
            summary = f'Berdasarkan informasi terbaru dari internet:\n\n{summary_text}\n\n🌐 Sumber: {source_links}'
            self.knowledge_base[query_key] = {'query': query, 'summary': summary, 'sources': [r.get('href', '') for r in results], 'learned_at': datetime.now(timezone.utc).isoformat()}
            self._save_db()
            return summary
        except Exception as e:
            print(f'[Error Internet] {e}')
            return 'Koneksi internetku sedang bermasalah atau DuckDuckGo membatasi pencarianku sementara. Coba beberapa saat lagi ya!'