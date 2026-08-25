"""
Skala tier ProMax — memilih arsitektur ~1.2B / ~4B / ~8B berdasarkan RAM & VRAM.
"""
from __future__ import annotations

from core.config import get_gpu_vram_gb, get_system_ram_gb


def estimate_transformer_params(d_model: int, n_layers: int, vocab_size: int, d_ff: int) -> int:
    """Estimasi kasar parameter decoder-only Transformer."""
    embed = vocab_size * d_model * 2
    per_layer = 12 * d_model * d_model + 2 * d_model * d_ff
    return int(embed + n_layers * per_layer)


PROMAX_TIERS = {
    "promax_1b": {
        "d_model": 1536,
        "n_heads": 24,
        "n_layers": 28,
        "d_ff": 6144,
        "max_seq_len": 1024,
        "vocab_size": 96000,
        "batch_size": 1,
        "min_ram_gb": 48.0,
        "label": "PROMAX 1.2B — Tier Dasar",
    },
    "promax_4b": {
        "d_model": 2560,
        "n_heads": 32,
        "n_layers": 36,
        "d_ff": 10240,
        "max_seq_len": 1024,
        "vocab_size": 128000,
        "batch_size": 1,
        "min_ram_gb": 64.0,
        "label": "PROMAX ~4B — Tier Canggih",
    },
    "promax_8b": {
        "d_model": 3584,
        "n_heads": 28,
        "n_layers": 40,
        "d_ff": 14336,
        "max_seq_len": 1024,
        "vocab_size": 160000,
        "batch_size": 1,
        "min_ram_gb": 96.0,
        "label": "PROMAX ~8B — Tier Tertinggi",
    },
}


def resolve_promax_tier(
    total_ram: float | None = None,
    vram_gb: float | None = None,
    force_tier: str | None = None,
    hardware_force: bool = False,
) -> str:
    """
    Pilih sub-tier ProMax.
    force_tier: promax_1b | promax_4b | promax_8b (env SPACEAX_PROMAX_TIER)
    """
    if force_tier and force_tier in PROMAX_TIERS:
        ram = total_ram if total_ram is not None else get_system_ram_gb()
        need = PROMAX_TIERS[force_tier]["min_ram_gb"]
        vram = vram_gb if vram_gb is not None else get_gpu_vram_gb()
        if ram < need:
            msg = (
                f"   ⚠️  Tier {force_tier}: RAM {ram:.1f} GB "
                f"< rekomendasi {need:.0f} GB — risiko OOM / swap lambat."
            )
            if hardware_force:
                print(msg + " (--force: training tetap dilanjutkan.)")
            else:
                print(msg)
        if force_tier == "promax_4b" and 0 < vram < 24:
            print(
                f"   ⚠️  VRAM {vram:.1f} GB: 4B biasanya butuh ≥24 GB (L4/A10) "
                f"atau batch=1 + checkpointing di Colab Pro+."
            )
        if force_tier == "promax_8b" and 0 < vram < 40:
            print(
                f"   ⚠️  VRAM {vram:.1f} GB: 8B biasanya butuh ≥40 GB (A100 40GB+)."
            )
        return force_tier

    ram = total_ram if total_ram is not None else get_system_ram_gb()
    vram = vram_gb if vram_gb is not None else get_gpu_vram_gb()

    if ram >= 96.0 and (vram >= 20.0 or ram >= 128.0):
        return "promax_8b"
    if ram >= 64.0:
        return "promax_4b"
    return "promax_1b"


def get_promax_profile(tier: str) -> dict:
    profile = dict(PROMAX_TIERS[tier])
    profile["tier"] = tier
    profile["est_params"] = estimate_transformer_params(
        profile["d_model"],
        profile["n_layers"],
        profile["vocab_size"],
        profile["d_ff"],
    )
    profile["label"] = (
        f"{profile['label']} (~{profile['est_params'] / 1e9:.1f}B params, "
        f"vocab={profile['vocab_size']:,})"
    )
    return profile


def apply_promax_training_overrides(training_cfg) -> None:
    """Hyperparameter khusus tier ProMax."""
    training_cfg.num_epochs = max(training_cfg.num_epochs, 30)
    training_cfg.warmup_steps = max(training_cfg.warmup_steps, 2000)
    if not getattr(training_cfg, "force_train", False):
        training_cfg.early_stopping_patience = max(
            getattr(training_cfg, "early_stopping_patience", 5), 7
        )
    training_cfg.learning_rate = min(training_cfg.learning_rate, 2e-4)
    training_cfg.gradient_accumulation_steps = max(
        training_cfg.gradient_accumulation_steps, 16
    )
