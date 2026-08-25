"""
Penyesuaian VRAM untuk ProMax 8B — tier tetap 8B, hyperparameter mengikuti GPU/RAM terpasang.
"""
from __future__ import annotations

import gc
import os
import time

import torch

from core.config import get_gpu_vram_gb, get_system_ram_gb
from core.model import SpaceaxModel
from core.promax import PROMAX_TIERS, estimate_transformer_params


def _estimate_training_vram_gb(
    param_count: int,
    batch_size: int,
    max_seq_len: int,
    d_model: int,
    *,
    weights_gb_per_b_param: float = 2.0,
) -> float:
    """Perkiraan kasar puncak VRAM saat training (bobot + Adafactor + aktivasi + checkpoint)."""
    params_b = param_count / 1e9
    weights_gb = params_b * weights_gb_per_b_param
    optim_gb = params_b * 0.45
    # Aktivasi dengan gradient checkpointing (konservatif agar jarang OOM)
    act_gb = (
        batch_size
        * max_seq_len
        * d_model
        * 56
        * 4
        / (1024**3)
        * 0.32
    )
    return weights_gb + optim_gb + act_gb + 2.0


def _pick_8b_profile(vram_gb: float) -> dict:
    """Pilih max_seq_len / accum / batch agar muat di VRAM (sisakan ~12% headroom)."""
    tier = PROMAX_TIERS["promax_8b"]
    params = estimate_transformer_params(
        tier["d_model"], tier["n_layers"], tier["vocab_size"], tier["d_ff"]
    )
    d_model = tier["d_model"]
    # Di GPU <40 GB bobot disimpan bfloat16 (~2 byte/param)
    w_factor = 1.05 if 0 < vram_gb < 40.0 else 2.0

    candidates = [
        {"max_seq_len": 1024, "batch_size": 2, "gradient_accumulation_steps": 16},
        {"max_seq_len": 1024, "batch_size": 1, "gradient_accumulation_steps": 16},
        {"max_seq_len": 768, "batch_size": 1, "gradient_accumulation_steps": 24},
        {"max_seq_len": 512, "batch_size": 1, "gradient_accumulation_steps": 32},
        {"max_seq_len": 384, "batch_size": 1, "gradient_accumulation_steps": 48},
        {"max_seq_len": 256, "batch_size": 1, "gradient_accumulation_steps": 64},
        {"max_seq_len": 128, "batch_size": 1, "gradient_accumulation_steps": 64},
    ]

    if vram_gb <= 0:
        budget = 0.0
    elif vram_gb < 16:
        budget = vram_gb * 0.78
    elif vram_gb < 32:
        budget = vram_gb * 0.80
    else:
        budget = vram_gb * 0.88
    chosen = candidates[-1]
    for cand in candidates:
        est = _estimate_training_vram_gb(
            params,
            cand["batch_size"],
            cand["max_seq_len"],
            d_model,
            weights_gb_per_b_param=w_factor,
        )
        if budget <= 0 or est <= budget:
            chosen = cand
            break

    return {
        **chosen,
        "est_vram_gb": _estimate_training_vram_gb(
            params,
            chosen["batch_size"],
            chosen["max_seq_len"],
            d_model,
            weights_gb_per_b_param=w_factor,
        ),
        "param_count": params,
        "weights_bf16": w_factor < 2.0,
    }


def apply_promax_8b_vram_fit(model_cfg, training_cfg, vram_gb: float | None = None) -> dict:
    """
    Sesuaikan training agar ProMax 8B memakai VRAM semaksimal mungkin tanpa menurunkan tier.
    Mengembalikan ringkasan profil yang dipilih.
    """
    vram = vram_gb if vram_gb is not None else get_gpu_vram_gb()
    ram = get_system_ram_gb()

    model_cfg.use_gradient_checkpointing = True
    training_cfg.fp16 = True
    training_cfg.optimizer_type = "adafactor"

    if not torch.cuda.is_available():
        training_cfg.batch_size = 1
        training_cfg.gradient_accumulation_steps = max(
            training_cfg.gradient_accumulation_steps, 32
        )
        model_cfg.max_seq_len = min(model_cfg.max_seq_len, 256)
        training_cfg.use_bfloat16_cpu = ram >= 32.0
        profile = {
            "mode": "cpu",
            "max_seq_len": model_cfg.max_seq_len,
            "batch_size": training_cfg.batch_size,
            "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
        }
        print(
            f"\n   🛡️  VRAM-fit 8B (CPU): seq={profile['max_seq_len']}, "
            f"batch={profile['batch_size']}, accum={profile['gradient_accumulation_steps']}"
        )
        return profile

    picked = _pick_8b_profile(vram)

    model_cfg.max_seq_len = min(model_cfg.max_seq_len, picked["max_seq_len"])
    training_cfg.batch_size = picked["batch_size"]
    training_cfg.gradient_accumulation_steps = max(
        training_cfg.gradient_accumulation_steps,
        picked["gradient_accumulation_steps"],
    )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    eff = training_cfg.batch_size * training_cfg.gradient_accumulation_steps
    profile = {
        "mode": "cuda",
        "vram_gb": vram,
        "max_seq_len": model_cfg.max_seq_len,
        "batch_size": training_cfg.batch_size,
        "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
        "effective_batch": eff,
        "est_peak_vram_gb": picked["est_vram_gb"],
    }
    print(f"\n   🛡️  VRAM-fit ProMax 8B (tier tetap, tanpa downgrade)")
    print(f"      GPU VRAM: {vram:.1f} GB | perkiraan puncak: ~{picked['est_vram_gb']:.1f} GB")
    print(
        f"      seq_len={profile['max_seq_len']} | batch={profile['batch_size']} | "
        f"accum={profile['gradient_accumulation_steps']} | effective batch={eff}"
    )
    if picked.get("weights_bf16"):
        print("      Bobot GPU: bfloat16 (otomatis jika VRAM < 40 GB)")
    if vram > 0 and picked["est_vram_gb"] > vram * 0.95:
        print(
            "      ⚠️  Perkiraan mendekati batas VRAM — jika OOM, kurangi --batch-size "
            "atau naikkan --grad-accum."
        )
    return profile


def clamp_8b_after_user_overrides(model_cfg, training_cfg) -> None:
    """Setelah CLI --batch-size/--grad-accum: pastikan kombinasi masih muat di VRAM."""
    if not torch.cuda.is_available():
        return
    vram = get_gpu_vram_gb()
    if vram <= 0:
        return

    tier = PROMAX_TIERS["promax_8b"]
    params = estimate_transformer_params(
        tier["d_model"], tier["n_layers"], tier["vocab_size"], tier["d_ff"]
    )

    w_factor = 1.05 if vram < 40.0 else 2.0
    while training_cfg.batch_size > 1:
        est = _estimate_training_vram_gb(
            params,
            training_cfg.batch_size,
            model_cfg.max_seq_len,
            tier["d_model"],
            weights_gb_per_b_param=w_factor,
        )
        if est <= vram * 0.88:
            break
        training_cfg.batch_size -= 1
        print(
            f"   🛡️  VRAM-fit: batch diturunkan ke {training_cfg.batch_size} "
            f"(estimasi {est:.1f} GB > budget)"
        )

    est = _estimate_training_vram_gb(
        params,
        training_cfg.batch_size,
        model_cfg.max_seq_len,
        tier["d_model"],
        weights_gb_per_b_param=w_factor,
    )
    seq = model_cfg.max_seq_len
    while est > vram * 0.88 and seq > 128:
        seq = max(128, seq // 2)
        model_cfg.max_seq_len = seq
        est = _estimate_training_vram_gb(
            params,
            training_cfg.batch_size,
            seq,
            tier["d_model"],
            weights_gb_per_b_param=w_factor,
        )
        print(f"   🛡️  VRAM-fit: max_seq_len → {seq} (estimasi puncak ~{est:.1f} GB)")

    if est > vram:
        training_cfg.gradient_accumulation_steps = max(
            training_cfg.gradient_accumulation_steps, 64
        )


def build_spaceax_model_vram_safe(mc, promax_tier: str | None = None) -> SpaceaxModel:
    """Bangun model; untuk 8B + CUDA muat layer ke GPU bertahap."""
    t0 = time.time()
    model = SpaceaxModel(mc)

    if not torch.cuda.is_available():
        print(f"   ✅ Model siap (CPU) dalam {time.time() - t0:.0f}s")
        return model

    device = torch.device("cuda")
    layerwise = promax_tier == "promax_8b"

    if layerwise:
        print("   🛡️  Memuat bobot ke GPU bertahap (kurangi puncak VRAM saat init)...")
        model.tok_embeddings.to(device)
        for i, layer in enumerate(model.layers):
            layer.to(device)
            if (i + 1) % 8 == 0:
                torch.cuda.empty_cache()
        model.norm.to(device)
        model.output.to(device)
        if model.freqs_cis is not None:
            model.freqs_cis = model.freqs_cis.to(device)
    else:
        print("   ⚡ Memindahkan model ke GPU...")
        model = model.to(device)

    vram = get_gpu_vram_gb()
    if (
        layerwise
        and vram > 0
        and vram < 40.0
        and torch.cuda.is_bf16_supported()
    ):
        print("   🛡️  Bobot model → bfloat16 (hemat ~50% VRAM parameter)")
        model = model.to(dtype=torch.bfloat16)

    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        print(f"   ✅ Model di GPU dalam {time.time() - t0:.0f}s (VRAM terpakai ~{alloc:.1f} GB)")
    return model
