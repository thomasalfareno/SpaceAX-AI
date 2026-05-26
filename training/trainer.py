import os
import time
import math
import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Optional

class _WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int=-1):
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]

class Trainer:
    SAMPLE_PROMPTS = ['Halo', 'Apa itu Python?', 'Turunan sin x?']

    def __init__(self, model, train_loader, val_loader, config, tokenizer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_accum = getattr(config, 'gradient_accumulation_steps', 1)
        steps_per_epoch = max(len(train_loader) // grad_accum, 1)
        self.total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = getattr(config, 'warmup_steps', 300)
        self.scheduler = _WarmupCosineScheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=self.total_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.use_amp = config.fp16 and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp) if self.use_amp else None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 3
        self.step = 0

    def save_checkpoint(self, path: str, is_best: bool=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'step': self.step, 'best_val_loss': self.best_val_loss, 'patience_counter': self.patience_counter}
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'model_best.pt')
            torch.save(checkpoint, best_path)
            print(f'⭐ Best model disimpan ke {best_path}')

    def load_checkpoint(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f'Checkpoint tidak ditemukan di {path}')
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        print(f'✅ Checkpoint dimuat dari {path}')
        return True

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()
        grad_accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        use_bfloat16 = getattr(self.config, 'use_bfloat16_cpu', False) and self.device.type == 'cpu'
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_tokens = (targets != -100).sum().item()
            total_tokens += batch_tokens
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(inputs)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / grad_accum_steps
                self.scaler.scale(loss).backward()
            elif use_bfloat16:
                with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                    logits, _ = self.model(inputs)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / grad_accum_steps
                loss.backward()
            else:
                logits, _ = self.model(inputs)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / grad_accum_steps
                loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx + 1 == len(self.train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
            loss_item = loss.item() * grad_accum_steps
            total_loss += loss_item
            num_batches += 1
            if batch_idx % 10 == 0:
                curr_lr = self.scheduler.get_last_lr()[0]
                ppl = math.exp(loss_item) if loss_item < 20 else float('inf')
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / max(elapsed, 0.001)
                batches_per_sec = (batch_idx + 1) / max(elapsed, 0.001)
                remaining = len(self.train_loader) - (batch_idx + 1)
                eta_sec = remaining / batches_per_sec
                eta_str = f'{int(eta_sec // 60)}m {int(eta_sec % 60)}s'
                pct = (batch_idx + 1) / len(self.train_loader) * 100
                print(f'  Epoch {epoch} | [{pct:5.1f}%] Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss_item:.4f} | PPL: {ppl:.1f} | LR: {curr_lr:.2e} | {tokens_per_sec:.0f} tok/s | ETA: {eta_str}')
            if batch_idx % 200 == 0:
                gc.collect()
        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        tok_s = total_tokens / max(elapsed, 0.001)
        print(f'  → Epoch {epoch} selesai: avg_loss={avg_loss:.4f}, {tok_s:.0f} tok/s, {elapsed:.0f}s')
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logits, _ = self.model(inputs)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def _generate_samples(self, epoch: int):
        if self.tokenizer is None:
            return
        self.model.eval()
        bos_id = self.tokenizer.special_tokens['<BOS>']
        eos_id = self.tokenizer.special_tokens['<EOS>']
        emo_neutral_id = self.tokenizer.special_tokens['<EMO_NEUTRAL>']
        print(f'\n  📝 Sampel generasi (Epoch {epoch}):')
        print(f"  {'-' * 46}")
        for prompt_text in self.SAMPLE_PROMPTS:
            prompt_tokens = [bos_id] + self.tokenizer.encode(prompt_text) + [emo_neutral_id]
            try:
                generated_ids = self.model.generate(prompt_tokens=prompt_tokens, max_gen_len=80, temperature=0.7, top_p=0.9, top_k=50, eos_id=eos_id)
                response_text = self.tokenizer.decode(generated_ids)
                for st in self.tokenizer.special_tokens:
                    response_text = response_text.replace(st, '')
                response_text = response_text.strip()
                if len(response_text) > 150:
                    response_text = response_text[:150] + '...'
            except Exception as e:
                response_text = f'[Error: {e}]'
            print(f'  Prompt: "{prompt_text}"')
            print(f'  → {response_text}')
            print()

    def train(self):
        train_start = time.time()
        print(f'🚀 Memulai training di device: {self.device}')
        print(f'   Parameter: {self.model.count_parameters():,}')
        print(f'   Epochs: {self.config.num_epochs}')
        print(f'   Batch size: {self.config.batch_size}')
        print(f'   Batches per epoch: {len(self.train_loader)}')
        print(f'   Total optimizer steps: {self.total_steps}')
        print(f"   Warmup steps: {getattr(self.config, 'warmup_steps', 300)}")
        print(f'   Early stopping patience: {self.patience}')
        if self.tokenizer:
            print(f'   Sample generation: aktif ({len(self.SAMPLE_PROMPTS)} prompts)')
        print()
        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            epoch_time = time.time() - epoch_start
            train_ppl = math.exp(train_loss) if train_loss < 20 else float('inf')
            val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
            elapsed_total = time.time() - train_start
            avg_epoch_time = elapsed_total / epoch
            remaining_epochs = self.config.num_epochs - epoch
            eta_total = avg_epoch_time * remaining_epochs
            eta_str = f'{int(eta_total // 3600)}h {int(eta_total % 3600 // 60)}m' if eta_total > 3600 else f'{int(eta_total // 60)}m {int(eta_total % 60)}s'
            print(f"\n{'=' * 55}")
            print(f'Epoch {epoch}/{self.config.num_epochs} selesai ({epoch_time:.0f}s)')
            print(f'  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.1f}')
            print(f'  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.1f}')
            print(f'  Total elapsed: {elapsed_total:.0f}s | ETA sisa: {eta_str}')
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f'  🎉 Rekor baru! Val Loss terbaik: {val_loss:.4f}')
            else:
                self.patience_counter += 1
                remaining_patience = self.patience - self.patience_counter
                if remaining_patience > 0:
                    print(f'  ⚠️  Val loss tidak membaik. Patience: {remaining_patience}/{self.patience} tersisa.')
                else:
                    print(f'  🛑 Val loss tidak membaik selama {self.patience} epoch berturut-turut.')
            checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'checkpoints', f'model_epoch_{epoch}.pt')
            self.save_checkpoint(checkpoint_path, is_best)
            self._generate_samples(epoch)
            print(f"{'=' * 55}\n")
            if self.patience_counter >= self.patience:
                print(f'⏹️  Early stopping! Training dihentikan pada epoch {epoch}.')
                break
            gc.collect()
        total_time = time.time() - train_start
        total_str = f'{int(total_time // 3600)}h {int(total_time % 3600 // 60)}m {int(total_time % 60)}s'
        print(f'✅ Training selesai! Total waktu: {total_str}')
        print(f'   Best val loss: {self.best_val_loss:.4f}')