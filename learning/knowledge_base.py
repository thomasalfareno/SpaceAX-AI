import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

class KnowledgeEntry:

    def __init__(self, content: str, title: str='', category: str='umum', source: str='unknown', confidence: float=0.8, entities: list[str]=None, relations: list[dict]=None):
        self.content = content
        self.title = title
        self.category = category
        self.source = source
        self.confidence = confidence
        self.entities = entities or []
        self.relations = relations or []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.access_count = 0
        self.last_accessed = None

    def to_dict(self) -> dict:
        return {'content': self.content, 'title': self.title, 'category': self.category, 'source': self.source, 'confidence': self.confidence, 'entities': self.entities, 'relations': self.relations, 'created_at': self.created_at, 'access_count': self.access_count, 'last_accessed': self.last_accessed}

    @classmethod
    def from_dict(cls, data: dict) -> 'KnowledgeEntry':
        entry = cls(content=data['content'], title=data.get('title', ''), category=data.get('category', 'umum'), source=data.get('source', 'unknown'), confidence=data.get('confidence', 0.8), entities=data.get('entities', []), relations=data.get('relations', []))
        entry.created_at = data.get('created_at', entry.created_at)
        entry.access_count = data.get('access_count', 0)
        entry.last_accessed = data.get('last_accessed')
        return entry

    def access(self):
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc).isoformat()

class SimpleKnowledgeGraph:

    def __init__(self):
        self.graph: dict[str, list[tuple[str, str]]] = {}
        self.reverse: dict[str, list[tuple[str, str]]] = {}

    def add_relation(self, subject: str, predicate: str, obj: str):
        subject = subject.lower().strip()
        obj = obj.lower().strip()
        predicate = predicate.lower().strip()
        if subject not in self.graph:
            self.graph[subject] = []
        relation = (predicate, obj)
        if relation not in self.graph[subject]:
            self.graph[subject].append(relation)
        if obj not in self.reverse:
            self.reverse[obj] = []
        rev_relation = (predicate, subject)
        if rev_relation not in self.reverse[obj]:
            self.reverse[obj].append(rev_relation)

    def get_relations(self, entity: str) -> list[dict]:
        entity = entity.lower().strip()
        results = []
        for pred, obj in self.graph.get(entity, []):
            results.append({'subject': entity, 'predicate': pred, 'object': obj, 'direction': 'forward'})
        for pred, subj in self.reverse.get(entity, []):
            results.append({'subject': subj, 'predicate': pred, 'object': entity, 'direction': 'reverse'})
        return results

    def find_path(self, start: str, end: str, max_depth: int=3) -> list[dict]:
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
            for pred, obj in self.graph.get(current, []):
                step = {'from': current, 'predicate': pred, 'to': obj}
                if obj == end:
                    return path + [step]
                if obj not in visited:
                    visited.add(obj)
                    queue.append((obj, path + [step]))
            for pred, subj in self.reverse.get(current, []):
                step = {'from': subj, 'predicate': pred, 'to': current}
                if subj == end:
                    return path + [step]
                if subj not in visited:
                    visited.add(subj)
                    queue.append((subj, path + [step]))
        return []

    def get_all_entities(self) -> list[str]:
        entities = set(self.graph.keys()) | set(self.reverse.keys())
        return sorted(entities)

    def to_dict(self) -> dict:
        return {'graph': {k: [list(v) for v in vals] for k, vals in self.graph.items()}, 'reverse': {k: [list(v) for v in vals] for k, vals in self.reverse.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> 'SimpleKnowledgeGraph':
        kg = cls()
        kg.graph = {k: [tuple(v) for v in vals] for k, vals in data.get('graph', {}).items()}
        kg.reverse = {k: [tuple(v) for v in vals] for k, vals in data.get('reverse', {}).items()}
        return kg

class EntityExtractor:
    PATTERNS = {'person': ['\\b(?:Presiden|Pahlawan|Tokoh|Ilmuwan|Profesor|Dr\\.?|Bapak|Ibu)\\s+([A-Z][a-zA-Z\\s]+)'], 'place': ['\\b(?:Kota|Kabupaten|Provinsi|Pulau|Gunung|Sungai|Danau|Desa|Negara)\\s+([A-Z][a-zA-Z\\s]+)'], 'organization': ['\\b(?:Universitas|Perusahaan|Organisasi|Lembaga|Kementerian)\\s+([A-Z][a-zA-Z\\s]+)']}
    RELATION_PATTERNS = [('(.+?)\\s+adalah\\s+(.+?)\\.', 'adalah'), ('(.+?)\\s+merupakan\\s+(.+?)\\.', 'merupakan'), ('(.+?)\\s+terletak di\\s+(.+?)\\.', 'terletak_di'), ('(.+?)\\s+berada di\\s+(.+?)\\.', 'berada_di'), ('(.+?)\\s+ditemukan oleh\\s+(.+?)\\.', 'ditemukan_oleh'), ('(.+?)\\s+dibuat oleh\\s+(.+?)\\.', 'dibuat_oleh'), ('(.+?)\\s+bagian dari\\s+(.+?)\\.', 'bagian_dari'), ('(.+?)\\s+memiliki\\s+(.+?)\\.', 'memiliki'), ('(.+?)\\s+dikenal sebagai\\s+(.+?)\\.', 'dikenal_sebagai'), ('(.+?)\\s+terdiri dari\\s+(.+?)\\.', 'terdiri_dari')]

    def extract_entities(self, text: str) -> list[dict]:
        entities = []
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    name = match.group(1).strip()
                    if len(name) > 2 and len(name) < 50:
                        entities.append({'name': name, 'type': entity_type, 'context': text[max(0, match.start() - 30):match.end() + 30]})
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2 and (word not in ['Dan', 'Atau', 'Yang', 'Ini', 'Itu', 'Dengan', 'Untuk', 'Dari', 'Pada', 'Di', 'Ke', 'Se']):
                if i > 0 and (not words[i - 1].endswith(('.', '!', '?', ':'))):
                    entities.append({'name': word, 'type': 'unknown', 'context': ' '.join(words[max(0, i - 3):i + 4])})
        seen = set()
        unique = []
        for e in entities:
            key = e['name'].lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def extract_relations(self, text: str) -> list[dict]:
        relations = []
        for pattern, predicate in self.RELATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                if 2 < len(subject) < 50 and 2 < len(obj) < 100:
                    relations.append({'subject': subject, 'predicate': predicate, 'object': obj})
        return relations

class KnowledgeBase:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.knowledge_dir = os.path.join(data_dir, 'knowledge')
        self.kb_file = os.path.join(self.knowledge_dir, 'knowledge_base.json')
        self.graph_file = os.path.join(self.knowledge_dir, 'knowledge_graph.json')
        os.makedirs(self.knowledge_dir, exist_ok=True)
        self.entries: dict[str, KnowledgeEntry] = {}
        self.graph = SimpleKnowledgeGraph()
        self.entity_extractor = EntityExtractor()
        self._load()

    def _load(self):
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
        data = {'entries': {k: v.to_dict() for k, v in self.entries.items()}, 'total': len(self.entries), 'last_updated': datetime.now(timezone.utc).isoformat()}
        with open(self.kb_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(self.graph.to_dict(), f, ensure_ascii=False, indent=2)

    def add_knowledge(self, content: str, title: str='', category: str='umum', source: str='unknown', confidence: float=0.8) -> str:
        entry = KnowledgeEntry(content=content, title=title, category=category, source=source, confidence=confidence)
        entities = self.entity_extractor.extract_entities(content)
        entry.entities = [e['name'] for e in entities]
        relations = self.entity_extractor.extract_relations(content)
        entry.relations = relations
        for rel in relations:
            self.graph.add_relation(rel['subject'], rel['predicate'], rel['object'])
        import hashlib
        key = hashlib.md5(content[:200].encode()).hexdigest()[:12]
        self.entries[key] = entry
        self.save()
        return key

    def search(self, query: str, top_k: int=5) -> list[dict]:
        query_words = set(query.lower().split())
        scored_results = []
        for key, entry in self.entries.items():
            content_lower = entry.content.lower()
            title_lower = entry.title.lower()
            score = 0.0
            for word in query_words:
                if len(word) < 3:
                    continue
                if word in title_lower:
                    score += 3.0
                if word in content_lower:
                    count = content_lower.count(word)
                    score += min(count, 5) * 1.0
            score *= entry.confidence
            score *= 1 + 0.1 * min(entry.access_count, 10)
            if score > 0:
                scored_results.append({'key': key, 'title': entry.title, 'content': entry.content[:300], 'category': entry.category, 'score': score, 'confidence': entry.confidence})
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        for result in scored_results[:top_k]:
            if result['key'] in self.entries:
                self.entries[result['key']].access()
        return scored_results[:top_k]

    def get_related_knowledge(self, entity: str) -> dict:
        relations = self.graph.get_relations(entity)
        related_entries = []
        entity_lower = entity.lower()
        for key, entry in self.entries.items():
            if entity_lower in entry.content.lower() or entity_lower in [e.lower() for e in entry.entities]:
                related_entries.append({'key': key, 'title': entry.title, 'category': entry.category})
        return {'entity': entity, 'relations': relations, 'related_entries': related_entries[:10]}

    def get_stats(self) -> dict:
        categories = {}
        for entry in self.entries.values():
            cat = entry.category
            categories[cat] = categories.get(cat, 0) + 1
        return {'total_entries': len(self.entries), 'total_entities': len(self.graph.get_all_entities()), 'total_relations': sum((len(v) for v in self.graph.graph.values())), 'categories': categories}

    def ingest_from_web_learner(self, knowledge_entry: dict):
        self.add_knowledge(content=knowledge_entry.get('content', ''), title=knowledge_entry.get('title', ''), category=knowledge_entry.get('categories', ['umum'])[0], source=knowledge_entry.get('source_url', 'web'), confidence=0.7)
        for fact in knowledge_entry.get('key_facts', []):
            self.add_knowledge(content=fact, title=knowledge_entry.get('title', ''), category=knowledge_entry.get('categories', ['umum'])[0], source=knowledge_entry.get('source_url', 'web'), confidence=0.6)