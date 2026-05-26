import os
import re
from typing import List, Dict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:

    def __init__(self, vocab_size: int=16000):
        self.vocab_size = vocab_size
        self.special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<SEP>': 4, '<EMO_JOY>': 5, '<EMO_SAD>': 6, '<EMO_ANGER>': 7, '<EMO_FEAR>': 8, '<EMO_SURPRISE>': 9, '<EMO_DISGUST>': 10, '<EMO_TRUST>': 11, '<EMO_ANTICIPATION>': 12, '<EMO_NEUTRAL>': 13}
        self.special_tokens_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<SEP>', '<EMO_JOY>', '<EMO_SAD>', '<EMO_ANGER>', '<EMO_FEAR>', '<EMO_SURPRISE>', '<EMO_DISGUST>', '<EMO_TRUST>', '<EMO_ANTICIPATION>', '<EMO_NEUTRAL>']
        self._build_fresh_tokenizer()

    def _build_fresh_tokenizer(self):
        self.tokenizer = Tokenizer(BPE(unk_token='<UNK>'))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.add_special_tokens(self.special_tokens_list)

    def train(self, text: str):
        print(f'⚡ Memulai BPE training (Rust backend)… target vocab_size: {self.vocab_size}')
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens_list, min_frequency=2, show_progress=True)
        iterator = [line for line in text.split('\n') if line.strip()]
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        vocab = self.tokenizer.get_vocab()
        for tok, expected_id in self.special_tokens.items():
            actual_id = vocab.get(tok, None)
            if actual_id is not None and actual_id != expected_id:
                pass
        final_size = self.tokenizer.get_vocab_size()
        print(f'✅ Training selesai. Final vocab size: {final_size:,}')

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, vocab_dir: str):
        os.makedirs(vocab_dir, exist_ok=True)
        path = os.path.join(vocab_dir, 'tokenizer.json')
        self.tokenizer.save(path)

    def load(self, vocab_dir: str) -> bool:
        path = os.path.join(vocab_dir, 'tokenizer.json')
        if not os.path.exists(path):
            return False
        try:
            self.tokenizer = Tokenizer.from_file(path)
            existing = set(self.tokenizer.get_vocab().keys())
            missing = [t for t in self.special_tokens_list if t not in existing]
            if missing:
                self.tokenizer.add_special_tokens(missing)
            return True
        except Exception as e:
            print(f'⚠️ Gagal memuat tokenizer: {e}')
            return False