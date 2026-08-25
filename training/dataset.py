"""
SpaceaxAI - Data Loader & Dataset
Memproses format data percakapan menjadi input/target tensor untuk model causal LM.

Response-only loss masking:
  Format sequence: <BOS> user_tokens [EMO_*] ai_tokens <EOS>
  Loss hanya dihitung pada ai_tokens dan <EOS>.

Augmentasi on-the-fly (train): paraphrase + kadang tanpa blok <pikir>
agar model belajar merangkai kata, bukan menghafal satu kalimat tetap.
"""

import json
import random
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

from training.text_augment import augment_conversation_pair, strip_thought_blocks


class ConversationDataset(Dataset):
    """Dataset causal LM — tokenisasi per batch dengan augmentasi opsional."""

    EMO_TOKEN_IDS = set(range(5, 14))

    def __init__(
        self,
        conversations: List[dict],
        tokenizer,
        max_seq_len: int = 512,
        augment: bool = False,
        oversample_emotion: bool = False,
        dedupe_inputs: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.oversample_emotion = oversample_emotion

        self.pad_id = tokenizer.special_tokens["<PAD>"]
        self.bos_id = tokenizer.special_tokens["<BOS>"]
        self.eos_id = tokenizer.special_tokens["<EOS>"]

        self.conversations = self._filter_and_dedupe(conversations, dedupe_inputs)
        self.index_map: List[int] = []
        self._build_index_map()

    @classmethod
    def from_file(
        cls,
        data_file: str,
        tokenizer,
        max_seq_len: int = 512,
        augment: bool = False,
        oversample_emotion: bool = False,
    ) -> "ConversationDataset":
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        convs = data.get("conversations", [])
        return cls(
            convs,
            tokenizer,
            max_seq_len=max_seq_len,
            augment=augment,
            oversample_emotion=oversample_emotion,
        )

    def _filter_and_dedupe(
        self, conversations: List[dict], dedupe: bool
    ) -> List[dict]:
        seen = set()
        out = []
        for conv in conversations:
            inp = (conv.get("input") or "").strip()
            resp = (conv.get("response") or "").strip()
            if not inp or not resp:
                continue
            key = hashlib.md5(inp.lower().encode()).hexdigest()
            if dedupe and key in seen:
                continue
            seen.add(key)
            out.append(conv)
        return out

    def _build_index_map(self):
        """Indeks ke conversations; duplikat indeks = oversample dengan augment berbeda."""
        rng = random.Random(42)
        for i, conv in enumerate(self.conversations):
            self.index_map.append(i)
            if self.augment and rng.random() < 0.35:
                self.index_map.append(i)

            # Oversample vision pairs agar model vision terlatih secara intensif di background
            if conv.get("topic") == "vision" or conv.get("image"):
                self.index_map.append(i)
                self.index_map.append(i)

            if self.oversample_emotion:
                emo_topics = {
                    "emosi", "empati", "sedih", "senang", "marah", "perasaan",
                }
                topic = conv.get("topic", "")
                emotion = conv.get("emotion", "neutral")
                if emotion != "neutral" or topic in emo_topics or "emosi" in topic:
                    self.index_map.append(i)

    def _resolve_emotion_id(self, emotion_str: str) -> int:
        emo_token_str = f"<EMO_{emotion_str.upper()}>"
        if emo_token_str in self.tokenizer.special_tokens:
            return self.tokenizer.special_tokens[emo_token_str]
        return self.tokenizer.special_tokens["<EMO_NEUTRAL>"]

    def _build_and_truncate(
        self, user_tokens: List[int], emo_id: int, ai_tokens: List[int]
    ) -> Tuple[List[int], int]:
        overhead = 3
        max_content = self.max_seq_len - overhead

        if max_content <= 0:
            full_seq = [self.bos_id, emo_id, self.eos_id]
            return full_seq, 2

        total_content = len(user_tokens) + len(ai_tokens)

        if total_content <= max_content:
            pass
        elif len(ai_tokens) <= max_content:
            avail_user = max_content - len(ai_tokens)
            user_tokens = user_tokens[-avail_user:] if avail_user > 0 else []
        else:
            min_user = min(10, len(user_tokens))
            avail_ai = max_content - min_user
            user_tokens = user_tokens[-min_user:]
            ai_tokens = ai_tokens[: max(avail_ai, 1)]

        full_seq = [self.bos_id] + user_tokens + [emo_id] + ai_tokens + [self.eos_id]
        response_start = len(user_tokens) + 2
        return full_seq, response_start

    def _conv_to_tensors(self, conv: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_text = conv.get("input", "").strip()
        ai_text = conv.get("response", "").strip()
        emotion = conv.get("emotion", "neutral")
        context = conv.get("context", "").strip()

        if self.augment:
            user_text, ai_text = augment_conversation_pair(user_text, ai_text)
            if random.random() < 0.3:
                stripped = strip_thought_blocks(ai_text)
                if stripped and len(stripped) > 8:
                    ai_text = stripped

        emo_id = self._resolve_emotion_id(emotion)

        if context:
            user_text = f"{context}\nUser: {user_text}"

        user_tokens = self.tokenizer.encode(user_text)
        ai_tokens = self.tokenizer.encode(ai_text)

        full_seq, response_start = self._build_and_truncate(
            user_tokens, emo_id, ai_tokens
        )

        seq = list(full_seq)
        input_ids = seq[:-1]
        target_ids = seq[1:]

        mask_end = response_start - 1
        labels = [-100] * mask_end + target_ids[mask_end:]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv = self.conversations[self.index_map[idx]]
        input_ids, labels = self._conv_to_tensors(conv)

        img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        image_path = conv.get("image")
        if image_path:
            try:
                from core.vision import load_and_preprocess_image
                loaded = load_and_preprocess_image(image_path)
                if loaded is not None:
                    img_tensor = loaded.squeeze(0)
            except Exception:
                pass

        return input_ids, labels, img_tensor


def create_dataloaders(
    data_file: str,
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 512,
    split_ratio: float = 0.9,
    augment: bool = True,
    num_workers: Optional[int] = None,
    oversample_emotion: bool = False,
):
    """Buat Train dan Validation DataLoader."""
    import multiprocessing

    with open(data_file, "r", encoding="utf-8") as f:
        all_convs = json.load(f).get("conversations", [])

    n = len(all_convs)
    indices = list(range(n))
    rng = random.Random(42)
    rng.shuffle(indices)

    train_n = max(1, int(split_ratio * n)) if n > 1 else n
    train_idx_set = set(indices[:train_n])
    val_idx_set = set(indices[train_n:])

    train_convs = [all_convs[i] for i in range(n) if i in train_idx_set]
    val_convs = [all_convs[i] for i in range(n) if i in val_idx_set]
    if not val_convs:
        val_convs = train_convs[: max(1, len(train_convs) // 10)]

    train_dataset = ConversationDataset(
        train_convs,
        tokenizer,
        max_seq_len,
        augment=augment,
        oversample_emotion=oversample_emotion,
        dedupe_inputs=True,
    )
    val_dataset = ConversationDataset(
        val_convs,
        tokenizer,
        max_seq_len,
        augment=False,
        oversample_emotion=False,
        dedupe_inputs=True,
    )

    from core.config import get_system_ram_gb

    total_ram = get_system_ram_gb()

    if num_workers is None:
        if torch.cuda.is_available() and total_ram >= 16.0:
            num_workers = min(2, multiprocessing.cpu_count() or 0)
        else:
            num_workers = 0

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    aug_note = (
        " (augmentasi on-the-fly: paraphrase + variasi jawaban)"
        if augment
        else ""
    )
    print(
        f"   📚 Train: {len(train_dataset):,} sampel | Val: {len(val_dataset):,}{aug_note}"
    )

    return train_loader, val_loader
