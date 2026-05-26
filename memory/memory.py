import sqlite3
import os
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore

class ShortTermMemory:

    def __init__(self, capacity: int=20):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add(self, role: str, content: str, emotion: str='neutral', topic: str='umum'):
        turn = {'role': role, 'content': content, 'emotion': emotion, 'topic': topic, 'timestamp': datetime.now(timezone.utc).isoformat()}
        self.buffer.append(turn)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_context(self, n: int=None) -> str:
        if n is None:
            n = self.capacity
        turns = self.buffer[-n:]
        context = ''
        for turn in turns:
            prefix = 'User' if turn['role'] == 'user' else 'AI'
            context += f"{prefix}: {turn['content']}\n"
        return context

    def clear(self):
        self.buffer = []

class LongTermMemory:

    def __init__(self, db_path: str, vector_store_path: str):
        self.db_path = db_path
        self.vector_store = VectorStore(vector_store_path)
        self._init_db()
        self.vector_store.load()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('\n            CREATE TABLE IF NOT EXISTS facts (\n                id TEXT PRIMARY KEY,\n                content TEXT,\n                source TEXT,\n                category TEXT,\n                confidence REAL,\n                created_at TEXT,\n                accessed_at TEXT,\n                access_count INTEGER DEFAULT 0\n            )\n        ')
        c.execute('\n            CREATE TABLE IF NOT EXISTS episodes (\n                id TEXT PRIMARY KEY,\n                summary TEXT,\n                participants TEXT,\n                emotion TEXT,\n                timestamp TEXT,\n                importance REAL\n            )\n        ')
        c.execute('\n            CREATE TABLE IF NOT EXISTS user_info (\n                key TEXT PRIMARY KEY,\n                value TEXT,\n                confidence REAL,\n                updated_at TEXT\n            )\n        ')
        conn.commit()
        conn.close()

    def store_fact(self, content: str, source: str='conversation', category: str='umum', confidence: float=1.0):
        import hashlib
        fact_id = f'fact_{hashlib.md5(content.encode()).hexdigest()[:10]}'
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO facts (id, content, source, category, confidence, created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)', (fact_id, content, source, category, confidence, now, now))
        conn.commit()
        conn.close()
        self.vector_store.add(fact_id, content, {'type': 'fact', 'category': category})
        self.vector_store.save()
        return fact_id

    def store_episode(self, summary: str, emotion: str, importance: float):
        import hashlib
        now = datetime.now(timezone.utc).isoformat()
        ep_id = f'ep_{hashlib.md5((summary + now).encode()).hexdigest()[:10]}'
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO episodes (id, summary, participants, emotion, timestamp, importance) VALUES (?, ?, ?, ?, ?, ?)', (ep_id, summary, 'user,ai', emotion, now, importance))
        conn.commit()
        conn.close()
        if importance > 0.5:
            self.vector_store.add(ep_id, summary, {'type': 'episode', 'emotion': emotion})
            self.vector_store.save()

    def store_user_info(self, key: str, value: str, confidence: float=1.0):
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO user_info (key, value, confidence, updated_at) VALUES (?, ?, ?, ?)', (key, value, confidence, now))
        conn.commit()
        conn.close()

    def get_user_info(self, key: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT value FROM user_info WHERE key = ?', (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def recall_facts(self, query: str, top_k: int=5) -> List[Dict]:
        results = self.vector_store.search(query, top_k=top_k * 2)
        facts = []
        fact_ids = []
        for r in results:
            if r['metadata'].get('type') == 'fact':
                facts.append(r)
                fact_ids.append(r['id'])
                if len(facts) >= top_k:
                    break
        if fact_ids:
            now = datetime.now(timezone.utc).isoformat()
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            placeholders = ','.join(['?'] * len(fact_ids))
            c.execute(f'UPDATE facts SET access_count = access_count + 1, accessed_at = ? WHERE id IN ({placeholders})', [now] + fact_ids)
            conn.commit()
            conn.close()
        return facts

class MemoryManager:

    def __init__(self, memories_dir: str):
        self.memories_dir = memories_dir
        os.makedirs(memories_dir, exist_ok=True)
        db_path = os.path.join(memories_dir, 'ltm.db')
        vs_path = os.path.join(memories_dir, 'vectors.json')
        self.stm = ShortTermMemory(capacity=10)
        self.ltm = LongTermMemory(db_path, vs_path)

    def process_turn(self, role: str, content: str, emotion: str='neutral', topic: str='umum'):
        self.stm.add(role, content, emotion, topic)
        if role == 'user':
            lower = content.lower()
            if 'nama saya' in lower or 'namaku' in lower:
                words = lower.split()
                try:
                    if 'saya' in words:
                        idx = words.index('saya')
                    else:
                        idx = words.index('namaku')
                    if idx + 1 < len(words):
                        nama = ' '.join(words[idx + 1:idx + 3])
                        nama = re.sub('[^\\w\\s]', '', nama)
                        if nama and nama != 'adalah':
                            self.ltm.store_user_info('nama', nama.title())
                except ValueError:
                    pass
            elif 'suka' in lower and (not 'tidak suka' in lower):
                self.ltm.store_fact(f'User menyukai: {content}', source='user', category='preferensi')

    def get_context_for_generation(self, current_input: str) -> str:
        relevant_facts = self.ltm.recall_facts(current_input, top_k=2)
        user_name = self.ltm.get_user_info('nama')
        context_parts = []
        if user_name:
            context_parts.append(f'[Info: Nama user adalah {user_name}]')
        if relevant_facts:
            facts_str = ' '.join([f['text'] for f in relevant_facts])
            context_parts.append(f'[Memori: {facts_str}]')
        chat_history = self.stm.get_context()
        if chat_history:
            context_parts.append(chat_history)
        return '\n'.join(context_parts)