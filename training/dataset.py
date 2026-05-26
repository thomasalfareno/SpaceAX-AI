import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

class ConversationDataset(Dataset):
    EMO_TOKEN_IDS = set(range(5, 14))

    def __init__(self, data_file: str, tokenizer, max_seq_len: int=512, augment: bool=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.pad_id = tokenizer.special_tokens['<PAD>']
        self.bos_id = tokenizer.special_tokens['<BOS>']
        self.eos_id = tokenizer.special_tokens['<EOS>']
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.conversations = data.get('conversations', [])
        self.samples: List[Tuple[List[int], int]] = []
        self._prepare_data()

    def _resolve_emotion_id(self, emotion_str: str) -> int:
        emo_token_str = f'<EMO_{emotion_str.upper()}>'
        if emo_token_str in self.tokenizer.special_tokens:
            return self.tokenizer.special_tokens[emo_token_str]
        return self.tokenizer.special_tokens['<EMO_NEUTRAL>']

    def _augment_text(self, text: str) -> str:
        if not text:
            return text
        if random.random() < 0.3:
            return text[0].swapcase() + text[1:]
        return text

    def _build_and_truncate(self, user_tokens: List[int], emo_id: int, ai_tokens: List[int]) -> Tuple[List[int], int]:
        overhead = 3
        max_content = self.max_seq_len - overhead
        if max_content <= 0:
            full_seq = [self.bos_id, emo_id, self.eos_id]
            return (full_seq, 2)
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
            ai_tokens = ai_tokens[:max(avail_ai, 1)]
        full_seq = [self.bos_id] + user_tokens + [emo_id] + ai_tokens + [self.eos_id]
        response_start = len(user_tokens) + 2
        return (full_seq, response_start)

    def _prepare_data(self):
        for conv in self.conversations:
            user_text = conv.get('input', '').strip()
            ai_text = conv.get('response', '').strip()
            emotion = conv.get('emotion', 'neutral')
            if not user_text or not ai_text:
                continue
            emo_id = self._resolve_emotion_id(emotion)
            user_tokens = self.tokenizer.encode(user_text)
            ai_tokens = self.tokenizer.encode(ai_text)
            full_seq, response_start = self._build_and_truncate(user_tokens, emo_id, ai_tokens)
            self.samples.append((full_seq, response_start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        full_seq, response_start = self.samples[idx]
        seq = list(full_seq)
        input_ids = seq[:-1]
        target_ids = seq[1:]
        mask_end = response_start - 1
        labels = [-100] * mask_end + target_ids[mask_end:]
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len
        return (torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long))

def create_dataloaders(data_file: str, tokenizer, batch_size: int=32, max_seq_len: int=512, split_ratio: float=0.9, augment: bool=True, num_workers: Optional[int]=None):
    import multiprocessing
    dataset = ConversationDataset(data_file, tokenizer, max_seq_len, augment=augment)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    if num_workers is None:
        if torch.cuda.is_available():
            num_workers = min(4, multiprocessing.cpu_count() or 0)
        else:
            num_workers = 0
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    return (train_loader, val_loader)