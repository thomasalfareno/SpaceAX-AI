"""
SpaceaxAI - HuggingFace BPE Tokenizer v3.0
Tokenizer dirombak menggunakan implementasi Rust dari HuggingFace agar sangat cepat.
Kompatibel 100% dengan pipeline SpaceAx (special_tokens dict, encode, decode, save, load, vision tokens).
"""

import os
import re
from typing import List, Dict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    def __init__(self, vocab_size: int = 72000):
        self.target_vocab_size = vocab_size

        # Special tokens — ID tetap, digunakan oleh EmotionEngine, Dataset, Trainer, Vision
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<SEP>": 4,
            "<EMO_JOY>": 5,
            "<EMO_SAD>": 6,
            "<EMO_ANGER>": 7,
            "<EMO_FEAR>": 8,
            "<EMO_SURPRISE>": 9,
            "<EMO_DISGUST>": 10,
            "<EMO_TRUST>": 11,
            "<EMO_ANTICIPATION>": 12,
            "<EMO_NEUTRAL>": 13,
            "<pikir>": 14,
            "</pikir>": 15,
            "<IMG>": 16,
            "</IMG>": 17,
            "<IMG_PAD>": 18,
        }

        # Susun list terurut agar BpeTrainer menetapkan ID yang benar
        self.special_tokens_list = [
            "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>",
            "<EMO_JOY>", "<EMO_SAD>", "<EMO_ANGER>", "<EMO_FEAR>",
            "<EMO_SURPRISE>", "<EMO_DISGUST>", "<EMO_TRUST>",
            "<EMO_ANTICIPATION>", "<EMO_NEUTRAL>",
            "<pikir>", "</pikir>",
            "<IMG>", "</IMG>", "<IMG_PAD>",
        ]

        self._build_fresh_tokenizer()

    def _build_fresh_tokenizer(self):
        """Buat instance tokenizer HF baru (belum dilatih)."""
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.add_special_tokens(self.special_tokens_list)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str):
        """Latih tokenizer BPE dengan teks korpus."""
        print(f"⚡ Memulai BPE training (Rust backend)… target vocab_size: {self.target_vocab_size}")

        min_freq = 1 if self.target_vocab_size >= 72000 else 2
        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=self.special_tokens_list,
            min_frequency=min_freq,
            show_progress=True,
        )

        chunks = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) > 4000:
                for i in range(0, len(line), 2000):
                    part = line[i : i + 2000].strip()
                    if part:
                        chunks.append(part)
            else:
                chunks.append(line)
        iterator = chunks

        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        final_size = self.tokenizer.get_vocab_size()
        print(f"✅ Training selesai. Final vocab size: {final_size:,}")

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Ubah teks menjadi urutan ID token."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Ubah urutan ID token kembali menjadi teks."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, vocab_dir: str):
        """Simpan tokenizer ke disk (satu file JSON)."""
        os.makedirs(vocab_dir, exist_ok=True)
        path = os.path.join(vocab_dir, "tokenizer.json")
        self.tokenizer.save(path)

    def load(self, vocab_dir: str) -> bool:
        """Muat tokenizer dari disk. Return True jika berhasil."""
        path = os.path.join(vocab_dir, "tokenizer.json")
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
            print(f"⚠️ Gagal memuat tokenizer: {e}")
            return False

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
