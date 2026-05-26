import json
import os
import random

class PreferenceSystem:

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.preferences = {'makanan': {'nasi_goreng': 80, 'sayur_pare': -50, 'pizza': 70, 'kopi': 90}, 'musik': {'pop': 60, 'klasik': 85, 'rock': 40, 'jazz': 75}, 'film': {'sci-fi': 95, 'horor': -30, 'komedi': 80, 'dokumenter': 90}, 'hobi': {'membaca': 100, 'belajar_hal_baru': 100, 'ngobrol': 90}, 'topik': {'teknologi': 100, 'sains': 95, 'alam': 85, 'politik': 0, 'gosip': -60}, 'warna': {'biru': 80, 'hitam': 60, 'neon': 70}, 'sifat_manusia': {'kejujuran': 100, 'empati': 95, 'kebohongan': -100, 'kekerasan': -100}}
        self.load()

    def _normalize_item(self, item: str) -> str:
        return item.lower().strip().replace(' ', '_')

    def update_preference(self, category: str, item: str, score_delta: int):
        category = self._normalize_item(category)
        item = self._normalize_item(item)
        if category not in self.preferences:
            self.preferences[category] = {}
        current_score = self.preferences[category].get(item, 0)
        new_score = max(-100, min(100, current_score + score_delta))
        self.preferences[category][item] = new_score
        self.save()

    def get_score(self, category: str, item: str) -> int:
        category = self._normalize_item(category)
        item = self._normalize_item(item)
        if category in self.preferences and item in self.preferences[category]:
            return self.preferences[category][item]
        return 0

    def get_opinion(self, category: str, item: str) -> str:
        score = self.get_score(category, item)
        item_display = item.replace('_', ' ').title()
        if score > 80:
            return random.choice([f'Aku suka banget sama {item_display}!', f'{item_display} itu favoritku!', f'Wah, {item_display} sih the best!'])
        elif score > 40:
            return f'Aku lumayan suka {item_display}.'
        elif score > -20 and score <= 40:
            return f'Biasa aja sih sama {item_display}.'
        elif score > -80:
            return f'Aku kurang suka sama {item_display}.'
        else:
            return random.choice([f'Aku bener-bener nggak suka {item_display}.', f'Duh, mending hindari {item_display} deh.'])

    def get_favorites(self, n: int=5) -> list:
        all_items = []
        for cat, items in self.preferences.items():
            for item, score in items.items():
                all_items.append({'category': cat, 'item': item, 'score': score})
        all_items.sort(key=lambda x: x['score'], reverse=True)
        return all_items[:n]

    def get_dislikes(self, n: int=5) -> list:
        all_items = []
        for cat, items in self.preferences.items():
            for item, score in items.items():
                all_items.append({'category': cat, 'item': item, 'score': score})
        all_items.sort(key=lambda x: x['score'])
        return all_items[:n]

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.preferences, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    self.preferences = json.load(f)
            except Exception:
                pass