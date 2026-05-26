import json
import math
import os
import re
from typing import List, Dict, Tuple, Optional

class TFIDFVectorizer:

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.num_docs: int = 0
        self.stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'ini', 'itu', 'adalah', 'pada', 'dalam', 'tidak', 'akan', 'sebagai', 'juga', 'atau', 'oleh', 'telah', 'bisa', 'menjadi', 'sudah', 'saat', 'ada', 'mereka', 'lagi', 'baru', 'kami', 'kita', 'banyak', 'hingga', 'seperti', 'karena', 'namun', 'saya', 'kamu', 'dia', 'nya', 'sebuah', 'tentang', 'tersebut', 'bagi', 'belum', 'apa', 'siapa', 'bagaimana', 'mengapa', 'kapan', 'dimana'}

    def _preprocess(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub('[^\\w\\s]', ' ', text)
        words = text.split()
        processed = []
        for w in words:
            if len(w) <= 3 or w in self.stopwords:
                continue
            original = w
            if w.startswith('meng'):
                w = w[4:]
            elif w.startswith('men'):
                w = w[3:]
            elif w.startswith('mem'):
                w = w[3:]
            elif w.startswith('me'):
                w = w[2:]
            elif w.startswith('ber'):
                w = w[3:]
            elif w.startswith('ter'):
                w = w[3:]
            elif w.startswith('di'):
                w = w[2:]
            elif w.startswith('pe'):
                w = w[2:]
            if w.endswith('kannya'):
                w = w[:-6]
            elif w.endswith('nya'):
                w = w[:-3]
            elif w.endswith('kan'):
                w = w[:-3]
            elif w.endswith('an'):
                w = w[:-2]
            elif w.endswith('i'):
                w = w[:-1]
            if len(w) < 3:
                processed.append(original)
            else:
                processed.append(w)
        return processed

    def fit(self, documents: List[str]):
        self.num_docs = len(documents)
        self.vocab = {}
        df: Dict[str, int] = {}
        for doc in documents:
            words = set(self._preprocess(doc))
            for w in words:
                df[w] = df.get(w, 0) + 1
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        self.idf = {w: math.log(self.num_docs / (count + 1)) + 1 for w, count in df.items()}

    def transform(self, text: str) -> Dict[int, float]:
        words = self._preprocess(text)
        if not words:
            return {}
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        vector: Dict[int, float] = {}
        norm_sq = 0.0
        for w, count in tf.items():
            if w in self.vocab and w in self.idf:
                idx = self.vocab[w]
                tf_norm = 1 + math.log(count)
                val = tf_norm * self.idf[w]
                vector[idx] = val
                norm_sq += val * val
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for idx in vector:
                vector[idx] /= norm
        return vector

class VectorStore:

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.vectorizer = TFIDFVectorizer()
        self.documents: Dict[str, Dict] = {}
        self.needs_fit = False

    def add(self, doc_id: str, text: str, metadata: dict=None):
        if metadata is None:
            metadata = {}
        self.documents[doc_id] = {'text': text, 'metadata': metadata}
        self.needs_fit = True

    def _ensure_fitted(self):
        if not self.needs_fit and self.vectorizer.num_docs > 0:
            return
        if not self.documents:
            return
        texts = [doc['text'] for doc in self.documents.values()]
        self.vectorizer.fit(texts)
        for doc in self.documents.values():
            doc['vector'] = self.vectorizer.transform(doc['text'])
        self.needs_fit = False

    def _cosine_similarity(self, vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
        if len(vec1) > len(vec2):
            vec1, vec2 = (vec2, vec1)
        score = 0.0
        for idx, val in vec1.items():
            if idx in vec2:
                score += val * vec2[idx]
        return score

    def search(self, query: str, top_k: int=5) -> List[Dict]:
        self._ensure_fitted()
        if not self.documents:
            return []
        query_vec = self.vectorizer.transform(query)
        if not query_vec:
            return []
        results = []
        for doc_id, doc in self.documents.items():
            score = self._cosine_similarity(query_vec, doc.get('vector', {}))
            if score > 0:
                results.append({'id': doc_id, 'text': doc['text'], 'metadata': doc['metadata'], 'score': score})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def delete(self, doc_id: str):
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.needs_fit = True

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        data = {'documents': {doc_id: {'text': d['text'], 'metadata': d['metadata']} for doc_id, d in self.documents.items()}, 'vectorizer': {'vocab': self.vectorizer.vocab, 'idf': self.vectorizer.idf, 'num_docs': self.vectorizer.num_docs}}
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        if not os.path.exists(self.save_path):
            return False
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.documents = {}
            for doc_id, d in data.get('documents', {}).items():
                self.documents[doc_id] = {'text': d['text'], 'metadata': d['metadata'], 'vector': {}}
            vec_data = data.get('vectorizer', {})
            self.vectorizer.vocab = vec_data.get('vocab', {})
            self.vectorizer.idf = vec_data.get('idf', {})
            self.vectorizer.num_docs = vec_data.get('num_docs', 0)
            for doc in self.documents.values():
                doc['vector'] = self.vectorizer.transform(doc['text'])
            self.needs_fit = False
            return True
        except Exception as e:
            print(f'Error memuat vector store: {e}')
            return False