import json
import os
import logging
import time
from datetime import datetime, timezone
from typing import Optional
logger = logging.getLogger('spaceax.auto_trainer')

class AutoTrainer:

    def __init__(self, data_dir: str, min_new_pairs: int=50):
        self.data_dir = data_dir
        self.seed_dir = os.path.join(data_dir, 'seed')
        self.checkpoint_dir = os.path.join(data_dir, 'checkpoints')
        self.state_file = os.path.join(data_dir, 'auto_trainer_state.json')
        self.min_new_pairs = min_new_pairs
        os.makedirs(self.seed_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'last_training_time': None, 'last_training_pairs': 0, 'total_trainings': 0, 'pending_new_pairs': 0}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def collect_all_training_data(self) -> list[dict]:
        all_pairs = []
        seed_file = os.path.join(self.seed_dir, 'conversations.json')
        if os.path.exists(seed_file):
            try:
                with open(seed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    pairs = data.get('conversations', [])
                    all_pairs.extend(pairs)
                    logger.info(f'📚 Seed data: {len(pairs)} pairs')
            except Exception as e:
                logger.error(f'❌ Gagal load seed data: {e}')
        web_file = os.path.join(self.seed_dir, 'web_learned.json')
        if os.path.exists(web_file):
            try:
                with open(web_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    pairs = data.get('conversations', [])
                    all_pairs.extend(pairs)
                    logger.info(f'🌐 Web learned data: {len(pairs)} pairs')
            except Exception as e:
                logger.error(f'❌ Gagal load web data: {e}')
        conv_file = os.path.join(self.seed_dir, 'conversation_learned.json')
        if os.path.exists(conv_file):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    pairs = data.get('conversations', [])
                    all_pairs.extend(pairs)
                    logger.info(f'💬 Conversation data: {len(pairs)} pairs')
            except Exception as e:
                logger.error(f'❌ Gagal load conversation data: {e}')
        for filename in os.listdir(self.seed_dir):
            if filename.endswith('.json') and filename not in ['conversations.json', 'web_learned.json', 'conversation_learned.json']:
                filepath = os.path.join(self.seed_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'conversations' in data:
                            pairs = data['conversations']
                            all_pairs.extend(pairs)
                            logger.info(f'📄 Extra data ({filename}): {len(pairs)} pairs')
                except Exception:
                    pass
        logger.info(f'📊 Total training data: {len(all_pairs)} pairs')
        return all_pairs

    def add_conversation_data(self, user_input: str, ai_response: str, emotion: str='neutral', topic: str='percakapan'):
        conv_file = os.path.join(self.seed_dir, 'conversation_learned.json')
        existing = []
        if os.path.exists(conv_file):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing = data.get('conversations', [])
            except Exception:
                existing = []
        new_pair = {'input': user_input, 'response': ai_response, 'emotion': emotion, 'topic': topic, 'preference_update': {}, 'learned_from': 'conversation', 'learned_at': datetime.now(timezone.utc).isoformat()}
        existing.append(new_pair)
        with open(conv_file, 'w', encoding='utf-8') as f:
            json.dump({'conversations': existing, 'total': len(existing), 'last_updated': datetime.now(timezone.utc).isoformat()}, f, ensure_ascii=False, indent=2)
        self.state['pending_new_pairs'] = self.state.get('pending_new_pairs', 0) + 1
        self._save_state()

    def should_retrain(self) -> bool:
        return self.state.get('pending_new_pairs', 0) >= self.min_new_pairs

    def prepare_training_file(self) -> str:
        all_pairs = self.collect_all_training_data()
        seen_inputs = set()
        unique_pairs = []
        for pair in all_pairs:
            input_key = pair.get('input', '').strip().lower()
            if input_key and input_key not in seen_inputs:
                seen_inputs.add(input_key)
                unique_pairs.append(pair)
        quality_pairs = []
        for pair in unique_pairs:
            inp = pair.get('input', '').strip()
            resp = pair.get('response', '').strip()
            if len(inp) >= 3 and len(resp) >= 5:
                quality_pairs.append(pair)
        output_file = os.path.join(self.seed_dir, 'training_combined.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'conversations': quality_pairs, 'total': len(quality_pairs), 'sources': {'total_raw': len(all_pairs), 'after_dedup': len(unique_pairs), 'after_quality_filter': len(quality_pairs)}, 'prepared_at': datetime.now(timezone.utc).isoformat()}, f, ensure_ascii=False, indent=2)
        logger.info(f'✅ Training file siap: {len(quality_pairs)} pairs (dari {len(all_pairs)} raw)')
        return output_file

    def trigger_retrain(self, model=None, tokenizer=None, epochs: int=5):
        if model is None or tokenizer is None:
            logger.warning('⚠️ Model dan tokenizer harus disediakan untuk retrain')
            return False
        logger.info('🔄 Memulai retraining dengan data baru...')
        training_file = self.prepare_training_file()
        try:
            from training.dataset import create_dataloaders
            from training.trainer import Trainer
            from core.config import TrainingConfig
            train_loader, val_loader = create_dataloaders(training_file, tokenizer, batch_size=TrainingConfig.batch_size, max_seq_len=512)
            config = TrainingConfig()
            config.num_epochs = epochs
            config.learning_rate = 0.0001
            trainer = Trainer(model, train_loader, val_loader, config)
            trainer.train()
            self.state['last_training_time'] = datetime.now(timezone.utc).isoformat()
            self.state['total_trainings'] = self.state.get('total_trainings', 0) + 1
            self.state['pending_new_pairs'] = 0
            self._save_state()
            logger.info('✅ Retraining selesai!')
            return True
        except Exception as e:
            logger.error(f'❌ Retraining gagal: {e}')
            return False

    def get_stats(self) -> dict:
        all_pairs = self.collect_all_training_data()
        return {'total_training_pairs': len(all_pairs), 'pending_new_pairs': self.state.get('pending_new_pairs', 0), 'total_trainings': self.state.get('total_trainings', 0), 'last_training': self.state.get('last_training_time', 'Belum pernah'), 'retrain_threshold': self.min_new_pairs, 'should_retrain': self.should_retrain()}