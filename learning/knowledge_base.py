"""
SpaceaxAI - Knowledge Base
Manajemen pengetahuan terstruktur dengan knowledge graph sederhana.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional


class KnowledgeEntry:
    """Satu unit pengetahuan."""

    def __init__(
        self,
        content: str,
        title: str = "",
        category: str = "umum",
        source: str = "unknown",
        confidence: float = 0.8,
        entities: list[str] = None,
        relations: list[dict] = None,
    ):
        self.content = content
        self.title = title
        self.category = category
        self.source = source
        self.confidence = confidence
        self.entities = entities or []
        self.relations = relations or []  # [{subject, predicate, object}]
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.access_count = 0
        self.last_accessed = None

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "title": self.title,
            "category": self.category,
            "source": self.source,
            "confidence": self.confidence,
            "entities": self.entities,
            "relations": self.relations,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KnowledgeEntry':
        entry = cls(
            content=data["content"],
            title=data.get("title", ""),
            category=data.get("category", "umum"),
            source=data.get("source", "unknown"),
            confidence=data.get("confidence", 0.8),
            entities=data.get("entities", []),
            relations=data.get("relations", []),
        )
        entry.created_at = data.get("created_at", entry.created_at)
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = data.get("last_accessed")
        return entry

    def access(self):
        """Catat bahwa entry ini diakses."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc).isoformat()


class SimpleKnowledgeGraph:
    """Knowledge graph sederhana menggunakan triplets (subject, predicate, object)."""

    def __init__(self):
        # Adjacency list: subject -> [(predicate, object)]
        self.graph: dict[str, list[tuple[str, str]]] = {}
        # Reverse index: object -> [(predicate, subject)]
        self.reverse: dict[str, list[tuple[str, str]]] = {}

    def add_relation(self, subject: str, predicate: str, obj: str):
        """Tambah relasi ke graph."""
        subject = subject.lower().strip()
        obj = obj.lower().strip()
        predicate = predicate.lower().strip()

        if subject not in self.graph:
            self.graph[subject] = []
        
        # Cek duplikat
        relation = (predicate, obj)
        if relation not in self.graph[subject]:
            self.graph[subject].append(relation)

        # Reverse index
        if obj not in self.reverse:
            self.reverse[obj] = []
        rev_relation = (predicate, subject)
        if rev_relation not in self.reverse[obj]:
            self.reverse[obj].append(rev_relation)

    def get_relations(self, entity: str) -> list[dict]:
        """Dapatkan semua relasi dari sebuah entitas."""
        entity = entity.lower().strip()
        results = []

        # Forward relations
        for pred, obj in self.graph.get(entity, []):
            results.append({
                "subject": entity,
                "predicate": pred,
                "object": obj,
                "direction": "forward"
            })

        # Reverse relations
        for pred, subj in self.reverse.get(entity, []):
            results.append({
                "subject": subj,
                "predicate": pred,
                "object": entity,
                "direction": "reverse"
            })

        return results

    def find_path(self, start: str, end: str, max_depth: int = 3) -> list[dict]:
        """Cari jalur antara dua entitas (BFS)."""
        start = start.lower().strip()
        end = end.lower().strip()

        if start == end:
            return []

        visited = {start}
        queue = [(start, [])]

        while queue:
            current, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            # Cek forward relations
            for pred, obj in self.graph.get(current, []):
                step = {"from": current, "predicate": pred, "to": obj}
                if obj == end:
                    return path + [step]
                if obj not in visited:
                    visited.add(obj)
                    queue.append((obj, path + [step]))

            # Cek reverse relations
            for pred, subj in self.reverse.get(current, []):
                step = {"from": subj, "predicate": pred, "to": current}
                if subj == end:
                    return path + [step]
                if subj not in visited:
                    visited.add(subj)
                    queue.append((subj, path + [step]))

        return []  # Tidak ditemukan jalur

    def get_all_entities(self) -> list[str]:
        """Dapatkan semua entitas unik dalam graph."""
        entities = set(self.graph.keys()) | set(self.reverse.keys())
        return sorted(entities)

    def to_dict(self) -> dict:
        return {
            "graph": {k: [list(v) for v in vals] for k, vals in self.graph.items()},
            "reverse": {k: [list(v) for v in vals] for k, vals in self.reverse.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SimpleKnowledgeGraph':
        kg = cls()
        kg.graph = {k: [tuple(v) for v in vals] for k, vals in data.get("graph", {}).items()}
        kg.reverse = {k: [tuple(v) for v in vals] for k, vals in data.get("reverse", {}).items()}
        return kg


class EntityExtractor:
    """Ekstraksi entitas sederhana dari teks Indonesia."""

    # Pattern untuk mendeteksi entitas
    PATTERNS = {
        'person': [
            r'\b(?:Presiden|Pahlawan|Tokoh|Ilmuwan|Profesor|Dr\.?|Bapak|Ibu)\s+([A-Z][a-zA-Z\s]+)',
        ],
        'place': [
            r'\b(?:Kota|Kabupaten|Provinsi|Pulau|Gunung|Sungai|Danau|Desa|Negara)\s+([A-Z][a-zA-Z\s]+)',
        ],
        'organization': [
            r'\b(?:Universitas|Perusahaan|Organisasi|Lembaga|Kementerian)\s+([A-Z][a-zA-Z\s]+)',
        ],
    }

    # Relasi keyword patterns
    RELATION_PATTERNS = [
        (r'(.+?)\s+adalah\s+(.+?)\.', 'adalah'),
        (r'(.+?)\s+merupakan\s+(.+?)\.', 'merupakan'),
        (r'(.+?)\s+terletak di\s+(.+?)\.', 'terletak_di'),
        (r'(.+?)\s+berada di\s+(.+?)\.', 'berada_di'),
        (r'(.+?)\s+ditemukan oleh\s+(.+?)\.', 'ditemukan_oleh'),
        (r'(.+?)\s+dibuat oleh\s+(.+?)\.', 'dibuat_oleh'),
        (r'(.+?)\s+bagian dari\s+(.+?)\.', 'bagian_dari'),
        (r'(.+?)\s+memiliki\s+(.+?)\.', 'memiliki'),
        (r'(.+?)\s+dikenal sebagai\s+(.+?)\.', 'dikenal_sebagai'),
        (r'(.+?)\s+terdiri dari\s+(.+?)\.', 'terdiri_dari'),
    ]

    def extract_entities(self, text: str) -> list[dict]:
        """Ekstrak entitas dari teks."""
        entities = []

        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    name = match.group(1).strip()
                    if len(name) > 2 and len(name) < 50:
                        entities.append({
                            "name": name,
                            "type": entity_type,
                            "context": text[max(0, match.start()-30):match.end()+30]
                        })

        # Juga deteksi proper nouns (kata yang diawali huruf besar)
        words = text.split()
        for i, word in enumerate(words):
            if (word[0].isupper() and len(word) > 2 and 
                word not in ['Dan', 'Atau', 'Yang', 'Ini', 'Itu', 'Dengan', 'Untuk', 'Dari', 'Pada', 'Di', 'Ke', 'Se']):
                # Cek apakah bukan awal kalimat
                if i > 0 and not words[i-1].endswith(('.', '!', '?', ':')):
                    entities.append({
                        "name": word,
                        "type": "unknown",
                        "context": ' '.join(words[max(0,i-3):i+4])
                    })

        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            key = e['name'].lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    def extract_relations(self, text: str) -> list[dict]:
        """Ekstrak relasi dari teks menggunakan pattern matching."""
        relations = []

        for pattern, predicate in self.RELATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group(1).strip()
                obj = match.group(2).strip()

                # Filter yang terlalu panjang atau pendek
                if 2 < len(subject) < 50 and 2 < len(obj) < 100:
                    relations.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj
                    })

        return relations


class KnowledgeBase:
    """
    Knowledge Base utama.
    Menggabungkan storage, knowledge graph, dan entity extraction.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.knowledge_dir = os.path.join(data_dir, "knowledge")
        self.kb_file = os.path.join(self.knowledge_dir, "knowledge_base.json")
        self.graph_file = os.path.join(self.knowledge_dir, "knowledge_graph.json")
        
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        self.entries: dict[str, KnowledgeEntry] = {}
        self.graph = SimpleKnowledgeGraph()
        self.entity_extractor = EntityExtractor()
        
        self._load()

    def _load(self):
        """Load knowledge base dari disk."""
        if os.path.exists(self.kb_file):
            try:
                with open(self.kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, entry_data in data.get('entries', {}).items():
                        self.entries[key] = KnowledgeEntry.from_dict(entry_data)
            except Exception:
                self.entries = {}

        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = SimpleKnowledgeGraph.from_dict(data)
            except Exception:
                self.graph = SimpleKnowledgeGraph()

    def save(self):
        """Simpan knowledge base ke disk."""
        # Simpan entries
        data = {
            'entries': {k: v.to_dict() for k, v in self.entries.items()},
            'total': len(self.entries),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        with open(self.kb_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Simpan graph
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(self.graph.to_dict(), f, ensure_ascii=False, indent=2)

    def add_knowledge(self, content: str, title: str = "", category: str = "umum",
                      source: str = "unknown", confidence: float = 0.8) -> str:
        """
        Tambah pengetahuan baru ke KB.
        Otomatis ekstrak entitas dan relasi.
        
        Returns:
            Key/ID dari entry yang ditambahkan
        """
        # Buat entry
        entry = KnowledgeEntry(
            content=content,
            title=title,
            category=category,
            source=source,
            confidence=confidence,
        )

        # Ekstrak entitas
        entities = self.entity_extractor.extract_entities(content)
        entry.entities = [e['name'] for e in entities]

        # Ekstrak dan simpan relasi
        relations = self.entity_extractor.extract_relations(content)
        entry.relations = relations
        for rel in relations:
            self.graph.add_relation(rel['subject'], rel['predicate'], rel['object'])

        # Generate key
        import hashlib
        key = hashlib.md5(content[:200].encode()).hexdigest()[:12]
        
        self.entries[key] = entry
        self.save()
        
        return key

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Cari pengetahuan yang relevan dengan query.
        Menggunakan keyword matching sederhana.
        """
        query_words = set(query.lower().split())
        
        scored_results = []
        for key, entry in self.entries.items():
            content_lower = entry.content.lower()
            title_lower = entry.title.lower()
            
            # Skor berdasarkan keyword overlap
            score = 0.0
            for word in query_words:
                if len(word) < 3:
                    continue
                if word in title_lower:
                    score += 3.0  # Title match lebih penting
                if word in content_lower:
                    count = content_lower.count(word)
                    score += min(count, 5) * 1.0  # Cap at 5 occurrences
            
            # Boost berdasarkan confidence
            score *= entry.confidence
            
            # Boost berdasarkan akses (sering diakses = lebih relevan)
            score *= (1 + 0.1 * min(entry.access_count, 10))
            
            if score > 0:
                scored_results.append({
                    "key": key,
                    "title": entry.title,
                    "content": entry.content[:300],
                    "category": entry.category,
                    "score": score,
                    "confidence": entry.confidence,
                })
        
        # Sort by score
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Mark entries as accessed
        for result in scored_results[:top_k]:
            if result['key'] in self.entries:
                self.entries[result['key']].access()
        
        return scored_results[:top_k]

    def get_related_knowledge(self, entity: str) -> dict:
        """Dapatkan pengetahuan terkait entitas melalui knowledge graph."""
        relations = self.graph.get_relations(entity)
        
        # Cari juga entries yang mengandung entitas
        related_entries = []
        entity_lower = entity.lower()
        for key, entry in self.entries.items():
            if entity_lower in entry.content.lower() or entity_lower in [e.lower() for e in entry.entities]:
                related_entries.append({
                    "key": key,
                    "title": entry.title,
                    "category": entry.category,
                })
        
        return {
            "entity": entity,
            "relations": relations,
            "related_entries": related_entries[:10],
        }

    def get_stats(self) -> dict:
        """Dapatkan statistik knowledge base."""
        categories = {}
        for entry in self.entries.values():
            cat = entry.category
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_entries": len(self.entries),
            "total_entities": len(self.graph.get_all_entities()),
            "total_relations": sum(len(v) for v in self.graph.graph.values()),
            "categories": categories,
        }

    def ingest_from_web_learner(self, knowledge_entry: dict):
        """Ingest knowledge entry dari WebLearner."""
        self.add_knowledge(
            content=knowledge_entry.get('content', ''),
            title=knowledge_entry.get('title', ''),
            category=knowledge_entry.get('categories', ['umum'])[0],
            source=knowledge_entry.get('source_url', 'web'),
            confidence=0.7,  # Web content default confidence
        )

        # Juga tambah key facts sebagai entry terpisah
        for fact in knowledge_entry.get('key_facts', []):
            self.add_knowledge(
                content=fact,
                title=knowledge_entry.get('title', ''),
                category=knowledge_entry.get('categories', ['umum'])[0],
                source=knowledge_entry.get('source_url', 'web'),
                confidence=0.6,
            )
