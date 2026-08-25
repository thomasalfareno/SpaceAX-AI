"""
SpaceaxAI - Trainer
Siklus pelatihan (Training Loop) untuk melatih model Transformer dari nol.
Dioptimalkan untuk CPU dan RAM terbatas.

Fitur:
  - Linear warmup + cosine decay LR scheduler
  - Early stopping (patience=3)
  - Sample text generation di akhir setiap epoch
  - Tokens/sec dan estimasi waktu total
"""

import os
import time
import math
import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Optional


class _WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup lalu cosine decay ke min_lr_ratio * base_lr.

    Args:
        optimizer: Optimizer PyTorch.
        warmup_steps: Jumlah step warmup linear.
        total_steps: Total step training (warmup + decay).
        min_lr_ratio: Rasio minimum LR terhadap base_lr (default 0.1 = 10%).
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1, last_epoch: int = -1):
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, 1)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        if step < self.warmup_steps:
            # Linear warmup: 0 → base_lr
            scale = step / self.warmup_steps
        else:
            # Cosine decay: base_lr → min_lr_ratio * base_lr
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


class Adafactor(torch.optim.Optimizer):
    """Custom pure-PyTorch implementation of the Adafactor optimizer.
    Dioptimalkan untuk melatih model Transformer berukuran besar dengan memory overhead seminimal mungkin.
    """
    def __init__(self, params, lr=1e-3, eps1=1e-30, eps2=1e-3, clip_threshold=1.0,
                 beta1=None, beta2_decay=0.8, weight_decay=0.0):
        if lr is not None and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps1=eps1, eps2=eps2, clip_threshold=clip_threshold,
                        beta1=beta1, beta2_decay=beta2_decay, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps1 = group['eps1']
            eps2 = group['eps2']
            clip_threshold = group['clip_threshold']
            beta1 = group['beta1']
            beta2_decay = group['beta2_decay']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")

                state = self.state[p]

                # Inisialisasi state
                if len(state) == 0:
                    state['step'] = 0
                    if beta1 is not None:
                        state['exp_avg'] = torch.zeros_like(p)
                    
                    # Faktorkan second moment HANYA untuk parameter 2D matrix (Linear/Embedding)
                    shape = p.shape
                    if len(shape) == 2:
                        state['exp_avg_sq_row'] = torch.zeros((shape[0], 1), dtype=p.dtype, device=p.device)
                        state['exp_avg_sq_col'] = torch.zeros((1, shape[1]), dtype=p.dtype, device=p.device)
                    else:
                        # Gunakan second moment penuh untuk parameter non-2D (0D, 1D, 3D, 4D seperti Conv2d)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                beta2 = 1.0 - math.pow(step, -beta2_decay)

                # Hitung RMS gradien kuadrat
                grad_sq = grad.square().add_(eps1)

                if len(p.shape) == 2:
                    # Parameter 2D matrix: Factored update
                    r_avg = state['exp_avg_sq_row']
                    c_avg = state['exp_avg_sq_col']

                    # Update row dan col averages
                    row_mean = grad_sq.mean(dim=1, keepdim=True)
                    col_mean = grad_sq.mean(dim=0, keepdim=True)

                    r_avg.mul_(beta2).add_(row_mean, alpha=1.0 - beta2)
                    c_avg.mul_(beta2).add_(col_mean, alpha=1.0 - beta2)

                    # Estimasi second moment penuh dari faktor
                    r_avg_mean = r_avg.mean(dim=0, keepdim=True).add_(eps2)
                    v = torch.matmul(r_avg, c_avg).div_(r_avg_mean)
                else:
                    # Parameter non-2D: Standard unfactored update
                    v = state['exp_avg_sq']
                    v.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)

                # Scaling update step
                # U = G / (sqrt(v) + eps2)
                if len(p.shape) >= 2:
                    v.sqrt_().add_(eps2)
                    u = grad.div(v)
                else:
                    u = grad.div(v.sqrt().add_(eps2))

                # Clip step jika RMS melebihi threshold
                rms_u = u.square().mean().sqrt()
                if rms_u > clip_threshold:
                    u.mul_(clip_threshold / max(rms_u, 1e-12))

                # Terapkan learning rate
                u.mul_(lr)

                # Terapkan momentum (jika diaktifkan)
                if beta1 is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(u, alpha=1.0 - beta1)
                    p.add_(exp_avg, alpha=-1.0)
                else:
                    p.add_(u, alpha=-1.0)

                # Weight decay (jika ada)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

        return loss


class Trainer:
    """Trainer untuk SpaceaxModel.

    Args:
        model: Instance SpaceaxModel.
        train_loader: DataLoader training.
        val_loader: DataLoader validasi.
        config: TrainingConfig dataclass.
        tokenizer: BPETokenizer (opsional, untuk sample generation).
    """

    # 5 Prompt sampel teks untuk monitoring kualitas generasi
    SAMPLE_PROMPTS = [
        "Halo, siapa kamu?",
        "Apa itu Python dan SpaceAX AI?",
        "Berapa turunan dari sin x?",
        "Jelaskan tentang konsep Grouped Query Attention (GQA)",
        "Bagaimana cara belajar programming yang efektif?",
    ]

    # 5 Prompt sampel vision untuk monitoring kemampuan multimodal
    SAMPLE_VISION_PROMPTS = [
        ("assets/vision_train/cat_01.jpg", "Deskripsikan apa yang terlihat pada gambar ini"),
        ("assets/vision_train/fruit_01.jpg", "Gambar buah apakah ini?"),
        ("assets/vision_train/computer_01.jpg", "Jelaskan perangkat pada gambar ini"),
        ("assets/vision_train/city_night_01.jpg", "Pemandangan apa yang terlihat pada gambar?"),
        ("assets/vision_train/pizza_01.jpg", "Makanan apakah yang terlihat pada gambar?"),
    ]

    def __init__(self, model, train_loader, val_loader, config,
                 tokenizer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        optimizer_type = getattr(config, "optimizer_type", "adamw")
        if optimizer_type == "adafactor":
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            print("   ⚙️  Trainer: Menggunakan optimizer low-memory 'Adafactor' kustom.")
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Scheduler: linear warmup + cosine decay ke min_lr_ratio
        grad_accum = getattr(config, "gradient_accumulation_steps", 1)
        steps_per_epoch = max(len(train_loader) // grad_accum, 1)
        self.total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = getattr(config, "warmup_steps", 200)
        warmup_steps = min(warmup_steps, max(1, self.total_steps // 20))
        min_lr_ratio = getattr(config, "min_lr_ratio", 0.1)

        self.scheduler = _WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=min_lr_ratio,
        )

        # Loss (label smoothing rendah = konvergensi awal lebih cepat)
        label_smoothing = getattr(config, "label_smoothing", 0.05)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

        # Mixed precision (GPU only)
        self.use_amp = config.fp16 and self.device.type == "cuda"
        self.amp_dtype = torch.float16
        if self.use_amp:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                try:
                    major = torch.cuda.get_device_capability(0)[0]
                    if major >= 8:
                        self.amp_dtype = torch.bfloat16
                        print("   ⚡ Trainer: GPU mendukung bfloat16 secara native di hardware. Mengaktifkan bfloat16 mixed precision.")
                except Exception:
                    pass
        
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
            if self.use_amp
            else None
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.force_train = getattr(config, "force_train", False)
        patience_cfg = getattr(config, "early_stopping_patience", 5)

        if self.force_train or patience_cfg is None or patience_cfg <= 0:
            self.patience = config.num_epochs
            self.early_stopping_disabled = True
        else:
            self.patience = patience_cfg
            self.early_stopping_disabled = False

        self.step = 0
        self.checkpoint_meta = {}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Simpan checkpoint model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "meta": getattr(self, "checkpoint_meta", {}),
        }
        torch.save(checkpoint, path)
        if is_best:
            meta = getattr(self, "checkpoint_meta", {})
            tier_name = meta.get("promax_tier") or meta.get("profile_name") or "medium"
            best_path = os.path.join(os.path.dirname(path), f"model_best_{tier_name}.pt")
            torch.save(checkpoint, best_path)
            # Also save copy to generic model_best.pt for default convenience
            generic_best = os.path.join(os.path.dirname(path), "model_best.pt")
            torch.save(checkpoint, generic_best)
            print(f"⭐ Best model saved to {best_path}")

    def load_checkpoint(self, path: str) -> bool:
        """Muat checkpoint model."""
        if not os.path.exists(path):
            print(f"Checkpoint tidak ditemukan di {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.patience_counter = checkpoint.get("patience_counter", 0)
        print(f"✅ Checkpoint dimuat dari {path}")
        return True

    # ------------------------------------------------------------------
    # Training loop — satu epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()

        grad_accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        use_bfloat16 = (
            getattr(self.config, "use_bfloat16_cpu", False)
            and self.device.type == "cpu"
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                inputs, targets, images = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                images = images.to(self.device)
                if (images == 0).all():
                    images = None
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                images = None

            # Hitung jumlah token yang berkontribusi ke loss (non -100)
            batch_tokens = (targets != -100).sum().item()
            total_tokens += batch_tokens

            # Forward pass
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    model_out = self.model(inputs, images=images)
                    logits = model_out[0]
                    aux_loss = model_out[2] if len(model_out) > 2 else 0.0
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                    if isinstance(aux_loss, torch.Tensor) and aux_loss > 0:
                        loss = loss + 0.01 * aux_loss
                    loss = loss / grad_accum_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            elif use_bfloat16:
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    model_out = self.model(inputs, images=images)
                    logits = model_out[0]
                    aux_loss = model_out[2] if len(model_out) > 2 else 0.0
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                    if isinstance(aux_loss, torch.Tensor) and aux_loss > 0:
                        loss = loss + 0.01 * aux_loss
                    loss = loss / grad_accum_steps
                loss.backward()
            else:
                model_out = self.model(inputs, images=images)
                logits = model_out[0]
                aux_loss = model_out[2] if len(model_out) > 2 else 0.0
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                if isinstance(aux_loss, torch.Tensor) and aux_loss > 0:
                    loss = loss + 0.01 * aux_loss
                loss = loss / grad_accum_steps
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0 or (
                batch_idx + 1
            ) == len(self.train_loader):
                step_skipped = False
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    step_skipped = scale_after < scale_before
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()

                if not step_skipped:
                    self.scheduler.step()
                    self.step += 1
                self.optimizer.zero_grad(set_to_none=True)

            # Logging
            loss_item = loss.item() * grad_accum_steps
            total_loss += loss_item
            num_batches += 1

            if batch_idx % 10 == 0:
                curr_lr = self.scheduler.get_last_lr()[0]
                ppl = math.exp(loss_item) if loss_item < 20 else float("inf")
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / max(elapsed, 0.001)
                batches_per_sec = (batch_idx + 1) / max(elapsed, 0.001)
                remaining = len(self.train_loader) - (batch_idx + 1)
                eta_sec = remaining / batches_per_sec
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                pct = (batch_idx + 1) / len(self.train_loader) * 100

                print(
                    f"  Epoch {epoch} | [{pct:5.1f}%] Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss_item:.4f} | PPL: {ppl:.1f} | "
                    f"LR: {curr_lr:.2e} | {tokens_per_sec:.0f} tok/s | ETA: {eta_str}"
                )

            # Bersihkan memori secara berkala (sangat penting untuk RAM terbatas)
            is_low_mem = getattr(self.config, "optimizer_type", "adamw") == "adafactor"
            gc_interval = 250 if is_low_mem else 500
            if batch_idx % gc_interval == 0:
                gc.collect()

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        tok_s = total_tokens / max(elapsed, 0.001)
        print(
            f"  → Epoch {epoch} selesai: avg_loss={avg_loss:.4f}, "
            f"{tok_s:.0f} tok/s, {elapsed:.0f}s"
        )
        
        # Bersihkan cache CUDA di akhir epoch untuk optimalitas tokens/sec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return avg_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if len(batch) == 3:
                inputs, targets, images = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                images = images.to(self.device)
                if (images == 0).all():
                    images = None
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                images = None

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    model_out = self.model(inputs, images=images)
                    logits = model_out[0]
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
            else:
                model_out = self.model(inputs, images=images)
                logits = model_out[0]
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def _generate_samples(self, epoch: int):
        """Generate 5 sampel teks + 5 sampel vision untuk monitoring kualitas."""
        if self.tokenizer is None:
            return

        self.model.eval()
        bos_id = self.tokenizer.special_tokens["<BOS>"]
        eos_id = self.tokenizer.special_tokens["<EOS>"]
        emo_neutral_id = self.tokenizer.special_tokens["<EMO_NEUTRAL>"]

        print(f"\n  📝 Sampel Generasi Teks (5 Prompts | Epoch {epoch}):")
        print(f"  {'-' * 46}")

        for prompt_text in self.SAMPLE_PROMPTS:
            prompt_tokens = (
                [bos_id]
                + self.tokenizer.encode(prompt_text)
                + [emo_neutral_id]
            )

            try:
                from chat import is_valid_output, is_valid_output_relaxed

                generated_ids = self.model.generate(
                    prompt_tokens=prompt_tokens,
                    max_gen_len=80,
                    temperature=0.65,
                    top_p=0.88,
                    top_k=40,
                    eos_id=eos_id,
                )
                response_text = self.tokenizer.decode(generated_ids)
                for st in self.tokenizer.special_tokens:
                    if st not in ["<pikir>", "</pikir>"]:
                        response_text = response_text.replace(st, "")
                response_text = response_text.strip()

                ok = is_valid_output(response_text) or (
                    epoch <= 3 and is_valid_output_relaxed(response_text)
                )
                if response_text and not ok:
                    response_text = (
                        "[belum koheren — lanjutkan training; augmentasi + lebih banyak epoch]"
                    )
                elif len(response_text) > 150:
                    response_text = response_text[:150] + "..."
                elif not response_text:
                    response_text = "[kosong — epoch masih awal]"
            except Exception as e:
                response_text = f"[Error: {e}]"

            print(f"  Prompt: \"{prompt_text}\"")
            print(f"  → {response_text}")
            print()

        # Sampel Vision Multimodal (5 Prompts)
        if getattr(self.model, "vision_encoder", None) is not None:
            print(f"  👁️ Sampel Generasi Vision (5 Gambar | Epoch {epoch}):")
            print(f"  {'-' * 46}")
            from core.vision import load_and_preprocess_image
            for img_path, prompt_text in self.SAMPLE_VISION_PROMPTS:
                if not os.path.exists(img_path):
                    continue
                img_tensor = load_and_preprocess_image(img_path)
                if img_tensor is None:
                    continue
                img_tensor = img_tensor.to(self.device)
                prompt_tokens = (
                    [bos_id]
                    + self.tokenizer.encode(prompt_text)
                    + [emo_neutral_id]
                )
                prompt_tensor = torch.tensor([prompt_tokens], device=self.device)

                try:
                    with torch.no_grad():
                        out = self.model(prompt_tensor, images=img_tensor)
                        logits = out[0] if isinstance(out, (tuple, list)) else out
                        gen_ids = torch.argmax(logits[:, -1, :], dim=-1).tolist()
                        gen_text = self.tokenizer.decode(gen_ids).strip()
                        for st in self.tokenizer.special_tokens:
                            if st not in ["<pikir>", "</pikir>"]:
                                gen_text = gen_text.replace(st, "")
                        gen_text = gen_text.strip()
                        if not gen_text:
                            gen_text = "[Visual embedding diproyeksikan ke ruang d_model LLM]"
                except Exception as e:
                    gen_text = f"[Vision Error: {e}]"

                fname = os.path.basename(img_path)
                print(f"  Gambar: {fname} | Prompt: \"{prompt_text}\"")
                print(f"  → {gen_text}")
                print()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Jalankan training loop lengkap dengan early stopping."""
        train_start = time.time()

        print(f"🚀 Memulai training di device: {self.device}")
        print(f"   Parameter: {self.model.count_parameters():,}")
        print(
            "   💡 Epoch 1: model belajar pola bahasa (bukan hafalan 1:1); "
            "augmentasi on-the-fly memperbanyak variasi frasa. "
            "Chat bagus biasanya setelah val_loss < 4 (ProMax: ≥15–30 epoch)."
        )
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Batches per epoch: {len(self.train_loader)}")
        print(f"   Total optimizer steps: {self.total_steps}")
        warmup_actual = self.scheduler.warmup_steps
        peak_lr = self.config.learning_rate
        min_lr = peak_lr * getattr(self.config, "min_lr_ratio", 0.1)
        label_sm = getattr(self.config, "label_smoothing", 0.05)
        print(f"   Warmup steps: {warmup_actual} (peak LR: {peak_lr:.1e})")
        print(f"   LR Schedule: cosine decay {peak_lr:.1e} → {min_lr:.1e}")
        print(f"   Label smoothing: {label_sm}")
        if getattr(self, "early_stopping_disabled", False) or self.force_train:
            print("   Early stopping: NONAKTIF (semua epoch akan dijalankan)")
        else:
            print(f"   Early stopping patience: {self.patience}")
        if self.tokenizer:
            print(f"   Sample generation: aktif (5 text prompts + 5 vision samples)")
        print()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            # ---- Training ----
            train_loss = self.train_epoch(epoch)

            # ---- Validation ----
            val_loss = self.evaluate()

            epoch_time = time.time() - epoch_start
            train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

            # Estimasi waktu total
            elapsed_total = time.time() - train_start
            avg_epoch_time = elapsed_total / epoch
            remaining_epochs = self.config.num_epochs - epoch
            eta_total = avg_epoch_time * remaining_epochs
            eta_str = (
                f"{int(eta_total // 3600)}h {int((eta_total % 3600) // 60)}m"
                if eta_total > 3600
                else f"{int(eta_total // 60)}m {int(eta_total % 60)}s"
            )

            print(f"\n{'=' * 55}")
            print(f"Epoch {epoch}/{self.config.num_epochs} selesai ({epoch_time:.0f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.1f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.1f}")
            print(f"  Total elapsed: {elapsed_total:.0f}s | ETA sisa: {eta_str}")

            # ---- Early stopping check ----
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  🎉 Rekor baru! Val Loss terbaik: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                remaining_patience = self.patience - self.patience_counter
                if remaining_patience > 0:
                    print(
                        f"  ⚠️  Val loss tidak membaik. "
                        f"Patience: {remaining_patience}/{self.patience} tersisa."
                    )
                else:
                    print(
                        f"  🛑 Val loss tidak membaik selama {self.patience} epoch berturut-turut."
                    )

            # ---- Checkpoint ----
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "checkpoints", f"model_epoch_{epoch}.pt",
            )
            self.save_checkpoint(checkpoint_path, is_best)

            # ---- Sample generation ----
            self._generate_samples(epoch)

            print(f"{'=' * 55}\n")

            if not self.force_train and self.patience_counter >= self.patience:
                print(
                    f"⏹️  Early stopping! Training dihentikan pada epoch {epoch}."
                )
                break

            # Bersihkan memori
            gc.collect()

        total_time = time.time() - train_start
        total_str = (
            f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s"
        )
        print(f"✅ Training selesai! Total waktu: {total_str}")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
