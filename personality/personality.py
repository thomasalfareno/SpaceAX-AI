import json
import os
from dataclasses import dataclass

@dataclass
class PersonalityTraits:
    openness: float = 0.7
    conscientiousness: float = 0.6
    extraversion: float = 0.7
    agreeableness: float = 0.8
    neuroticism: float = 0.3

class PersonalitySystem:

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.traits = PersonalityTraits()
        self.load()

    def get_response_style(self) -> dict:
        style = {'length': 'medium', 'tone': 'friendly', 'creativity': 'high', 'structure': 'casual', 'empathy': 'high'}
        if self.traits.extraversion > 0.8:
            style['length'] = 'long'
            style['tone'] = 'very_friendly_and_enthusiastic'
        elif self.traits.extraversion < 0.3:
            style['length'] = 'short'
            style['tone'] = 'reserved_and_calm'
        if self.traits.agreeableness > 0.8:
            style['empathy'] = 'very_high_and_supportive'
        elif self.traits.agreeableness < 0.3:
            style['empathy'] = 'low_and_critical'
        if self.traits.openness > 0.8:
            style['creativity'] = 'very_high_philosophical'
        elif self.traits.openness < 0.3:
            style['creativity'] = 'literal_and_practical'
        if self.traits.conscientiousness > 0.8:
            style['structure'] = 'highly_structured_detailed'
        elif self.traits.conscientiousness < 0.3:
            style['structure'] = 'loose_and_casual'
        return style

    def evolve(self, interaction_data: dict):
        if interaction_data.get('creative_topic', False):
            self.traits.openness = min(1.0, self.traits.openness + 0.01)
        if interaction_data.get('user_is_friendly', False):
            self.traits.agreeableness = min(1.0, self.traits.agreeableness + 0.01)
            self.traits.extraversion = min(1.0, self.traits.extraversion + 0.01)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        data = {'openness': self.traits.openness, 'conscientiousness': self.traits.conscientiousness, 'extraversion': self.traits.extraversion, 'agreeableness': self.traits.agreeableness, 'neuroticism': self.traits.neuroticism}
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.traits.openness = data.get('openness', 0.7)
                    self.traits.conscientiousness = data.get('conscientiousness', 0.6)
                    self.traits.extraversion = data.get('extraversion', 0.7)
                    self.traits.agreeableness = data.get('agreeableness', 0.8)
                    self.traits.neuroticism = data.get('neuroticism', 0.3)
            except Exception:
                pass