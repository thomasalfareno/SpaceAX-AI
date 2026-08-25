"""
SpaceaxAI - Vector Store
Implementasi Vector Store TF-IDF sederhana dari nol tanpa dependensi eksternal.
Khusus dioptimalkan untuk bahasa Indonesia.
"""

import json
import math
import os
import re
from typing import List, Dict, Tuple, Optional

class TFIDFVectorizer:
    """Implementasi TF-IDF (Term Frequency - Inverse Document Frequency) dari nol."""
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.num_docs: int = 0
        
        # Stopwords Indonesia umum
        self.stopwords = {
            "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", 
            "adalah", "pada", "dalam", "tidak", "akan", "sebagai", "juga", "atau", 
            "oleh", "telah", "bisa", "menjadi", "sudah", "saat", "ada", "mereka", 
            "lagi", "baru", "kami", "kita", "banyak", "hingga", "seperti", "karena",
            "namun", "saya", "kamu", "dia", "nya", "sebuah", "tentang", "tersebut",
            "bagi", "belum", "apa", "siapa", "bagaimana", "mengapa", "kapan", "dimana"
        }
        
    def _preprocess(self, text: str) -> List[str]:
        """Bersihkan teks dan ekstrak kata."""
        # Lowercase
        text = text.lower()
        # Hapus tanda baca
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tokenisasi
        words = text.split()
        
        # Stemming sederhana (menghapus awalan/akhiran umum Indonesia)
        processed = []
        for w in words:
            if len(w) <= 3 or w in self.stopwords:
                continue
                
            # Sangat sederhana: hapus prefix me-, di-, ber-, ter-, pe-
            # dan suffix -kan, -i, -an, -nya
            original = w
            if w.startswith("meng"): w = w[4:]
            elif w.startswith("men"): w = w[3:]
            elif w.startswith("mem"): w = w[3:]
            elif w.startswith("me"): w = w[2:]
            elif w.startswith("ber"): w = w[3:]
            elif w.startswith("ter"): w = w[3:]
            elif w.startswith("di"): w = w[2:]
            elif w.startswith("pe"): w = w[2:]
            
            if w.endswith("kannya"): w = w[:-6]
            elif w.endswith("nya"): w = w[:-3]
            elif w.endswith("kan"): w = w[:-3]
            elif w.endswith("an"): w = w[:-2]
            elif w.endswith("i"): w = w[:-1]
            
            # Jika kata menjadi terlalu pendek, gunakan aslinya
            if len(w) < 3:
                processed.append(original)
            else:
                processed.append(w)
                
        return processed

    def fit(self, documents: List[str]):
        """Bangun vocabulary dan hitung IDF dari korpus dokumen."""
        self.num_docs = len(documents)
        self.vocab = {}
        
        # Hitung Document Frequency (DF)
        df: Dict[str, int] = {}
        
        for doc in documents:
            words = set(self._preprocess(doc))
            for w in words:
                df[w] = df.get(w, 0) + 1
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                    
        # Hitung IDF: log(N / (DF + 1)) + 1 (smoothing)
        self.idf = {w: math.log(self.num_docs / (count + 1)) + 1 for w, count in df.items()}

    def transform(self, text: str) -> Dict[int, float]:
        """Ubah teks menjadi vektor TF-IDF (format sparse)."""
        words = self._preprocess(text)
        if not words:
            return {}
            
        # Term Frequency (TF)
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
            
        # Hitung TF-IDF dan simpan dalam representasi sparse (index -> value)
        vector: Dict[int, float] = {}
        norm_sq = 0.0
        
        for w, count in tf.items():
            if w in self.vocab and w in self.idf:
                idx = self.vocab[w]
                # Log normalization untuk TF
                tf_norm = 1 + math.log(count)
                val = tf_norm * self.idf[w]
                vector[idx] = val
                norm_sq += val * val
                
        # L2 Normalization (Cosine Similarity nantinya cukup dot product)
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for idx in vector:
                vector[idx] /= norm
                
        return vector

class VectorStore:
    """
    Penyimpanan Vektor TF-IDF.
    Memungkinkan pencarian semantic sederhana tanpa butuh embedding model.
    """
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.vectorizer = TFIDFVectorizer()
        
        # id -> {text, metadata, vector}
        self.documents: Dict[str, Dict] = {}
        
        # Flag untuk tahu apakah vectorizer perlu di-fit ulang
        self.needs_fit = False

    def add(self, doc_id: str, text: str, metadata: dict = None):
        """Tambahkan dokumen ke store."""
        if metadata is None:
            metadata = {}
            
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata,
            # Vector akan dihitung nanti saat fit atau jika sudah fit
        }
        self.needs_fit = True

    def _ensure_fitted(self):
        """Pastikan vectorizer sudah dilatih dan semua vektor up-to-date."""
        if not self.needs_fit and self.vectorizer.num_docs > 0:
            return
            
        if not self.documents:
            return
            
        # Latih ulang vectorizer dengan semua dokumen
        texts = [doc["text"] for doc in self.documents.values()]
        self.vectorizer.fit(texts)
        
        # Hitung ulang semua vektor
        for doc in self.documents.values():
            doc["vector"] = self.vectorizer.transform(doc["text"])
            
        self.needs_fit = False

    def _cosine_similarity(self, vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
        """Hitung cosine similarity antara 2 vektor sparse (keduanya harus sudah dinormalisasi)."""
        # Karena sparse, kita iterasi yang ukurannya lebih kecil
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1
            
        score = 0.0
        for idx, val in vec1.items():
            if idx in vec2:
                score += val * vec2[idx]
        return score

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Cari dokumen paling mirip."""
        self._ensure_fitted()
        
        if not self.documents:
            return []
            
        # Encode query
        query_vec = self.vectorizer.transform(query)
        if not query_vec:
            return []
            
        # Hitung skor untuk semua dokumen
        results = []
        for doc_id, doc in self.documents.items():
            score = self._cosine_similarity(query_vec, doc.get("vector", {}))
            if score > 0:
                results.append({
                    "id": doc_id,
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score
                })
                
        # Urutkan berdasarkan skor tertinggi
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
        
    def delete(self, doc_id: str):
        """Hapus dokumen."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.needs_fit = True

    def save(self):
        """Simpan store ke disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        data = {
            "documents": {
                doc_id: {"text": d["text"], "metadata": d["metadata"]}
                for doc_id, d in self.documents.items()
            },
            "vectorizer": {
                "vocab": self.vectorizer.vocab,
                "idf": self.vectorizer.idf,
                "num_docs": self.vectorizer.num_docs
            }
        }
        
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load(self) -> bool:
        """Muat store dari disk."""
        if not os.path.exists(self.save_path):
            return False
            
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.documents = {}
            for doc_id, d in data.get("documents", {}).items():
                self.documents[doc_id] = {
                    "text": d["text"],
                    "metadata": d["metadata"],
                    "vector": {}
                }
                
            vec_data = data.get("vectorizer", {})
            self.vectorizer.vocab = vec_data.get("vocab", {})
            self.vectorizer.idf = vec_data.get("idf", {})
            self.vectorizer.num_docs = vec_data.get("num_docs", 0)
            
            # Hitung ulang vektor
            for doc in self.documents.values():
                doc["vector"] = self.vectorizer.transform(doc["text"])
                
            self.needs_fit = False
            return True
            
        except Exception as e:
            print(f"Error memuat vector store: {e}")
            return False
