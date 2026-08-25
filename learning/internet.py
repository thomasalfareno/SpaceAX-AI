import json
import os
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

from ddgs import DDGS
from datetime import datetime, timezone


class InternetLearner:
    TRUSTED_DOMAINS = [
        "wikipedia.org", "id.wikipedia.org",
        "kompas.com", "detik.com", "liputan6.com",
        "tribunnews.com", "cnnindonesia.com", "tempo.co",
        "bbc.com", "bbc.co.uk", "britannica.com",
    ]

    SPAM_KEYWORDS = [
        "citation", "bibliography", "apa style", "mla style", "chicago style",
        "apa format", "mla format", "apa citation", "mla citation",
        "cite this", "citation generator", "citation machine",
        "works cited", "in-text citation", "reference list",
        "grammarly", "plagiarism", "paraphrase", "paraphrasing",
        "rewrite", "rewriter", "turnitin", "quillbot", "wordtune",
        "plagiarism checker", "grammar checker",
        "ads", "sponsored", "adwords",
    ]

    SPAM_DOMAINS = [
        "scribbr.com", "easybib.com", "bibme.com", "citationmachine.net",
        "citethisforme.com", "mybib.com", "citefast.com",
        "formatically.com", "grafiati.com", "zbib.org",
        "quillbot.com", "grammarly.com", "turnitin.com", "wordtune.com",
    ]

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.db_path = os.path.join(knowledge_dir, "internet_db.json")
        self.knowledge_base = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_db(self):
        os.makedirs(self.knowledge_dir, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)

    def clean_search_query(self, query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r"^!search\s+", "", q)
        phrases_to_remove = [
            "apa arti dari", "apa arti", "arti dari", "apa itu", "siapa itu", "apa sih",
            "tolong carikan tentang", "tolong cari tentang", "cari tentang", "carikan tentang",
            "tolong cari", "tolong carikan", "cari info", "cari tahu", "tahukah kamu", "siapakah",
            "jelaskan tentang", "jelaskan maksud", "bagaimana cara",
        ]
        for phrase in phrases_to_remove:
            q = q.replace(phrase, "")
        q = re.sub(r"[?!\.]+$", "", q)
        return q.strip()

    def _query_keywords(self, query: str) -> list:
        cleaned = self.clean_search_query(query)
        return [w for w in re.findall(r"\b\w+\b", cleaned) if len(w) > 2]

    def lookup_cached(self, query: str, min_score: int = 12) -> str | None:
        """Cari jawaban tersimpan di internet_db.json (fuzzy match topik)."""
        if not self.knowledge_base:
            return None

        key = query.lower().strip()
        if key in self.knowledge_base:
            entry = self.knowledge_base[key]
            return entry.get("summary") or self.synthesize_answer(query, entry.get("raw_snippets", []))

        keywords = self._query_keywords(query)
        if not keywords:
            return None

        best_summary = None
        best_score = 0

        for stored_key, entry in self.knowledge_base.items():
            haystack = " ".join([
                stored_key,
                entry.get("query", ""),
                entry.get("topic", ""),
            ]).lower()
            score = sum(15 for kw in keywords if kw in haystack)
            topic = entry.get("topic", stored_key).lower()
            if topic and any(kw in topic for kw in keywords):
                score += 10
            if score > best_score:
                best_score = score
                best_summary = entry.get("summary")
                if not best_summary and entry.get("raw_snippets"):
                    best_summary = self.synthesize_answer(query, entry["raw_snippets"])

        if best_score >= min_score:
            return best_summary
        return None

    def synthesize_answer(self, query: str, snippets: list) -> str:
        """Rangkai jawaban profesional dari cuplikan — bukan copy-paste mentah."""
        topic = self.clean_search_query(query) or query.strip()
        if not snippets:
            return (
                f"Saya belum memiliki informasi tersimpan yang cukup tentang **{topic}**. "
                f"Silakan gunakan `!search {topic}` jika Anda ingin saya mencari sumber terbaru."
            )

        sentences = []
        seen = set()

        for snippet in snippets:
            if not snippet or not isinstance(snippet, str):
                continue
            text = re.sub(r"\s+", " ", snippet.strip())
            for part in re.split(r"(?<=[.!?])\s+", text):
                part = part.strip()
                if len(part) < 25:
                    continue
                norm = part.lower()[:80]
                if norm in seen:
                    continue
                seen.add(norm)
                sentences.append(part)
                if len(sentences) >= 4:
                    break
            if len(sentences) >= 4:
                break

        if not sentences:
            body = " ".join(str(s) for s in snippets[:2])[:400]
        else:
            body = " ".join(sentences)

        if len(body) > 520:
            cut = body[:520]
            last_dot = cut.rfind(".")
            body = cut[: last_dot + 1] if last_dot > 120 else cut + "..."

        return (
            f"Mengenai **{topic.title()}**, berikut penjelasan berdasarkan sumber yang telah saya pelajari:\n\n"
            f"{body}\n\n"
            f"Jika Anda ingin detail lebih lanjut, tanyakan bagian tertentu yang ingin didalami."
        )

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
        href = result.get("href", "").lower()
        body = result.get("body", "").lower()
        title = result.get("title", "").lower()

        for domain in self.TRUSTED_DOMAINS:
            if domain in href:
                score += 20
                break

        for kw in query_keywords:
            if kw in body:
                score += 5
            if kw in title:
                score += 3

        body_len = len(result.get("body", ""))
        if body_len > 150:
            score += 5
        elif body_len > 80:
            score += 2

        return score

    def search_and_learn(self, query: str) -> str:
        """Cek cache → cari internet jika perlu → simpan seed → jawab hasil olahan."""
        cleaned_query = self.clean_search_query(query)
        if not cleaned_query:
            cleaned_query = query.strip()

        query_key = query.lower().strip()

        cached = self.lookup_cached(query)
        if cached:
            return cached

        print(f"\n[🌐 AI mengakses Internet untuk: '{cleaned_query}']...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with DDGS() as ddgs:
                    raw_results = list(ddgs.text(cleaned_query, max_results=8, region="id-id"))

            if not raw_results:
                return "Maaf, saya tidak menemukan informasi relevan di internet saat ini."

            filtered_results = []
            for r in raw_results:
                href = r.get("href", "")
                body = r.get("body", "")
                if self._is_spam(href, body):
                    continue
                filtered_results.append(r)

            if not filtered_results:
                filtered_results = raw_results[:2]

            query_keywords = self._query_keywords(cleaned_query)
            filtered_results.sort(
                key=lambda r: self._score_result(r, query_keywords), reverse=True
            )
            results = filtered_results[:3]

            raw_snippets = []
            sources = []
            for r in results:
                body = r.get("body", "").strip()
                href = r.get("href", "")
                if body:
                    raw_snippets.append(body)
                if href:
                    title = re.sub(r"[^\w\s\-]", "", r.get("title", "Sumber"))[:40].strip()
                    sources.append(f"[{title}]({href})")

            summary = self.synthesize_answer(cleaned_query, raw_snippets)
            if sources:
                summary += f"\n\n🌐 Referensi: {', '.join(sources)}"

            self.knowledge_base[query_key] = {
                "query": query,
                "topic": cleaned_query,
                "raw_snippets": raw_snippets,
                "summary": summary,
                "sources": [r.get("href", "") for r in results],
                "learned_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_db()

            return summary

        except Exception as e:
            print(f"[Error Internet] {e}")
            return (
                "Koneksi internet sedang bermasalah atau layanan pencarian membatasi akses. "
                "Coba beberapa saat lagi ya!"
            )
