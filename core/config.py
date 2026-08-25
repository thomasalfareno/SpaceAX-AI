"""
SpaceaxAI - Configuration
Konfigurasi utama dengan auto-detect spesifikasi hardware.
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import platform
import multiprocessing
import torch

# Optimasi CUDA Tensor Cores & CUDNN
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
from dataclasses import dataclass, field

# Optimasi CPU Threads
try:
    cores = multiprocessing.cpu_count()
    # Sisakan 1 core untuk OS agar tidak hang
    torch.set_num_threads(max(1, cores - 1))
except Exception:
    pass

def get_system_ram_gb() -> float:
    """Deteksi total RAM sistem dalam GB."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulonglong = ctypes.c_ulonglong
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', c_ulonglong),
                ('ullAvailPhys', c_ulonglong),
                ('ullTotalPageFile', c_ulonglong),
                ('ullAvailPageFile', c_ulonglong),
                ('ullTotalVirtual', c_ulonglong),
                ('ullAvailVirtual', c_ulonglong),
                ('ullAvailExtendedVirtual', c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 4.0  # Default konservatif

def get_available_ram_gb() -> float:
    """Deteksi RAM tersedia (belum dipakai) dalam GB."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulonglong = ctypes.c_ulonglong
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', c_ulonglong),
                ('ullAvailPhys', c_ulonglong),
                ('ullTotalPageFile', c_ulonglong),
                ('ullAvailPageFile', c_ulonglong),
                ('ullTotalVirtual', c_ulonglong),
                ('ullAvailVirtual', c_ulonglong),
                ('ullAvailExtendedVirtual', c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        pass
    try:
        with open('/proc/meminfo', 'r') as f:
            mem = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mem[parts[0].rstrip(':')] = int(parts[1])
            available = mem.get('MemAvailable', mem.get('MemFree', 0))
            return available / (1024 * 1024)
    except Exception:
        pass
    return 2.0

def is_force_mode() -> bool:
    """Paksa tier/ukuran model dan nonaktifkan early stopping (--force / SPACEAX_FORCE)."""
    return os.environ.get("SPACEAX_FORCE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_gpu_vram_gb() -> float:
    """Deteksi total memori VRAM GPU dalam GB (backward compat wrapper)."""
    info = detect_compute_device()
    return info["vram_gb"]


def detect_compute_device() -> dict:
    """Deteksi perangkat komputasi secara cerdas dan komprehensif.

    Mendukung:
      - NVIDIA CUDA (GeForce, Quadro, Tesla, RTX, dll.)
      - AMD ROCm (Radeon RX, Instinct, dll.)
      - Apple Metal Performance Shaders (MPS) — M1/M2/M3/M4
      - Intel Extension for PyTorch (XPU) — Arc / iGPU
      - CPU-only fallback

    Returns:
        dict dengan keys:
          backend   : str   — 'cuda' | 'rocm' | 'mps' | 'xpu' | 'cpu'
          device    : str   — torch device string ('cuda', 'mps', 'xpu', 'cpu')
          gpu_name  : str   — nama GPU atau 'CPU Only'
          vram_gb   : float — VRAM dalam GB (0.0 jika CPU-only)
          vendor    : str   — 'nvidia' | 'amd' | 'apple' | 'intel' | 'cpu'
          cuda_ver  : str   — versi CUDA runtime (jika ada)
          driver    : str   — versi driver GPU (jika ada)
          compute   : str   — compute capability (jika ada)
          gpu_count : int   — jumlah GPU terdeteksi
    """
    result = {
        "backend": "cpu",
        "device": "cpu",
        "gpu_name": "CPU Only",
        "vram_gb": 0.0,
        "vendor": "cpu",
        "cuda_ver": "",
        "driver": "",
        "compute": "",
        "gpu_count": 0,
    }

    # -------------------------------------------------------------------
    # 1. NVIDIA CUDA (termasuk Tesla T4, RTX 3090/4090, A100, H100, dll.)
    # -------------------------------------------------------------------
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_name(0)

            cuda_ver = ""
            try:
                cuda_ver = torch.version.cuda or ""
            except Exception:
                pass

            driver_ver = ""
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    timeout=5, stderr=subprocess.DEVNULL,
                ).decode().strip().split("\n")[0]
                driver_ver = out.strip()
            except Exception:
                pass

            compute_cap = f"{props.major}.{props.minor}"
            gpu_count = torch.cuda.device_count()

            # Tentukan apakah ini CUDA native atau AMD ROCm yang menggunakan HIP
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            if is_rocm:
                result.update({
                    "backend": "rocm",
                    "device": "cuda",
                    "gpu_name": gpu_name,
                    "vram_gb": round(vram, 1),
                    "vendor": "amd",
                    "cuda_ver": str(torch.version.hip or ""),
                    "driver": driver_ver,
                    "compute": compute_cap,
                    "gpu_count": gpu_count,
                })
            else:
                result.update({
                    "backend": "cuda",
                    "device": "cuda",
                    "gpu_name": gpu_name,
                    "vram_gb": round(vram, 1),
                    "vendor": "nvidia",
                    "cuda_ver": cuda_ver,
                    "driver": driver_ver,
                    "compute": compute_cap,
                    "gpu_count": gpu_count,
                })
            return result
        except Exception:
            pass

    # -------------------------------------------------------------------
    # 2. Apple MPS (Metal Performance Shaders — M1/M2/M3/M4 Silicon)
    # -------------------------------------------------------------------
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            gpu_name = "Apple Silicon (MPS)"
            # Apple MPS menggunakan Unified Memory, estimasi dari System RAM
            unified_mem = get_system_ram_gb()
            # MPS biasanya dapat mengakses ~75% dari unified memory untuk GPU
            estimated_vram = round(unified_mem * 0.75, 1)

            try:
                import subprocess
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    timeout=3, stderr=subprocess.DEVNULL,
                ).decode().strip()
                if out:
                    gpu_name = f"Apple Silicon MPS ({out})"
            except Exception:
                pass

            result.update({
                "backend": "mps",
                "device": "mps",
                "gpu_name": gpu_name,
                "vram_gb": estimated_vram,
                "vendor": "apple",
                "cuda_ver": "",
                "driver": "Metal",
                "compute": "Apple Neural Engine",
                "gpu_count": 1,
            })
            return result
        except Exception:
            pass

    # -------------------------------------------------------------------
    # 3. Intel XPU (Arc / iGPU via Intel Extension for PyTorch)
    # -------------------------------------------------------------------
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            gpu_name = torch.xpu.get_device_name(0)
            vram = 0.0
            try:
                props = torch.xpu.get_device_properties(0)
                vram = getattr(props, "total_memory", 0) / (1024 ** 3)
            except Exception:
                # Intel iGPU biasanya menggunakan shared memory dari System RAM
                vram = round(get_system_ram_gb() * 0.5, 1)

            result.update({
                "backend": "xpu",
                "device": "xpu",
                "gpu_name": gpu_name,
                "vram_gb": round(vram, 1) if vram > 0 else round(get_system_ram_gb() * 0.5, 1),
                "vendor": "intel",
                "cuda_ver": "",
                "driver": "Intel IPEX",
                "compute": "",
                "gpu_count": torch.xpu.device_count(),
            })
            return result
    except ImportError:
        pass
    except Exception:
        pass

    # -------------------------------------------------------------------
    # 4. Fallback: deteksi iGPU atau discrete GPU via sistem (platform)
    # -------------------------------------------------------------------
    try:
        import subprocess
        if platform.system() == "Windows":
            # Query GPU name
            try:
                name_out = subprocess.check_output(
                    ["wmic", "path", "win32_videocontroller", "get", "name"],
                    timeout=5, stderr=subprocess.DEVNULL,
                ).decode()
                name_lines = [l.strip() for l in name_out.strip().split("\n")
                              if l.strip() and "Name" not in l]
                gpu_name = name_lines[0] if name_lines else "Unknown GPU"
            except Exception:
                gpu_name = "Unknown GPU"

            # Query adapter RAM
            igpu_vram = 0.0
            try:
                ram_out = subprocess.check_output(
                    ["wmic", "path", "win32_videocontroller", "get", "adapterram"],
                    timeout=5, stderr=subprocess.DEVNULL,
                ).decode()
                ram_lines = [l.strip() for l in ram_out.strip().split("\n")
                             if l.strip() and "AdapterRAM" not in l]
                if ram_lines:
                    try:
                        igpu_vram = round(int(ram_lines[0]) / (1024 ** 3), 1)
                    except ValueError:
                        pass
            except Exception:
                pass

            result["gpu_name"] = gpu_name
            if igpu_vram > 0:
                result["vram_gb"] = igpu_vram

        elif platform.system() == "Linux":
            out = subprocess.check_output(
                ["lspci"], timeout=5, stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.split("\n"):
                low = line.lower()
                if "vga" in low or "3d" in low or "display" in low:
                    result["gpu_name"] = line.split(": ", 1)[-1].strip()
                    break
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.split("\n"):
                if "Chipset Model" in line or "Chip" in line:
                    result["gpu_name"] = line.split(": ", 1)[-1].strip()
                    break
    except Exception:
        pass

    return result

def get_checkpoint_path(checkpoints_dir: str, profile_name: str = None, promax_tier: str = None) -> str:
    """Mendapatkan path checkpoint spesifik untuk tier model agar tidak tumpang tindih."""
    tier = promax_tier or profile_name or "medium"
    specific_path = os.path.join(checkpoints_dir, f"model_best_{tier}.pt")
    if os.path.exists(specific_path):
        return specific_path
    
    generic_path = os.path.join(checkpoints_dir, "model_best.pt")
    if os.path.exists(generic_path):
        return generic_path

    return specific_path

# ============================================================================
# Profil Model — dari SMALL hingga ULTRA (upgraded: GQA + MoE + Vision)
# ============================================================================

MODEL_PROFILES = {
    "small": {
        "d_model": 512, "n_heads": 8, "n_kv_heads": 2, "n_layers": 8,
        "d_ff": 1408, "max_seq_len": 512, "vocab_size": 72000,
        "n_experts": 2, "n_active_experts": 1,
        "vision_enabled": True,
        "batch_size": 16, "label": "SMALL (~35M params, GQA+MoE+Vision)",
        "min_ram_gb": 4.0,
    },
    "medium": {
        "d_model": 768, "n_heads": 12, "n_kv_heads": 4, "n_layers": 12,
        "d_ff": 2048, "max_seq_len": 1024, "vocab_size": 96000,
        "n_experts": 2, "n_active_experts": 1,
        "vision_enabled": True,
        "batch_size": 8, "label": "MEDIUM (~120M params, GQA+MoE+Vision)",
        "min_ram_gb": 8.0,
    },
    "large": {
        "d_model": 1024, "n_heads": 16, "n_kv_heads": 4, "n_layers": 18,
        "d_ff": 2816, "max_seq_len": 1024, "vocab_size": 96000,
        "n_experts": 4, "n_active_experts": 1,
        "vision_enabled": True,
        "batch_size": 4, "label": "LARGE (~350M params, GQA+MoE+Vision)",
        "min_ram_gb": 16.0,
    },
    "ultra": {
        "d_model": 1280, "n_heads": 20, "n_kv_heads": 4, "n_layers": 24,
        "d_ff": 3584, "max_seq_len": 1024, "vocab_size": 128000,
        "n_experts": 4, "n_active_experts": 2,
        "vision_enabled": True,
        "batch_size": 2, "label": "ULTRA (~750M params, GQA+MoE+Vision)",
        "min_ram_gb": 32.0,
    },
    # Alias — sub-tier dipilih di auto_model_config via core.promax
    "promax": {
        "d_model": 1536, "n_heads": 24, "n_kv_heads": 8, "n_layers": 28,
        "d_ff": 4096, "max_seq_len": 1024, "vocab_size": 96000,
        "n_experts": 4, "n_active_experts": 2,
        "vision_enabled": True,
        "batch_size": 1, "label": "PROMAX (auto-tier 1B/4B/8B)",
        "min_ram_gb": 48.0,
    },
}

def auto_model_config(size_override: str = None, force: bool = False):
    """Pilih konfigurasi model otomatis berdasarkan RAM dan GPU/VRAM."""
    total_ram = get_system_ram_gb()
    avail_ram = get_available_ram_gb()
    hw = detect_compute_device()
    vram_gb = hw["vram_gb"]

    print("=" * 60)
    print("Hardware System Detected:")
    print("-" * 60)
    print(f"   System RAM  : {total_ram:.1f} GB (available: {avail_ram:.1f} GB)")
    print(f"   OS          : {platform.system()} {platform.machine()}")
    print(f"   CPU Cores   : {multiprocessing.cpu_count()}")

    # GPU detail
    if hw["backend"] == "cuda":
        print(f"   GPU Vendor  : NVIDIA (CUDA)")
        print(f"   GPU Device  : {hw['gpu_name']}")
        print(f"   GPU VRAM    : {hw['vram_gb']:.1f} GB")
        print(f"   CUDA Version: {hw['cuda_ver']}")
        if hw["driver"]:
            print(f"   Driver      : {hw['driver']}")
        print(f"   Compute Cap.: {hw['compute']}")
        if hw["gpu_count"] > 1:
            print(f"   GPU Count   : {hw['gpu_count']}")
    elif hw["backend"] == "rocm":
        print(f"   GPU Vendor  : AMD (ROCm / HIP)")
        print(f"   GPU Device  : {hw['gpu_name']}")
        print(f"   GPU VRAM    : {hw['vram_gb']:.1f} GB")
        print(f"   HIP Version : {hw['cuda_ver']}")
        if hw["gpu_count"] > 1:
            print(f"   GPU Count   : {hw['gpu_count']}")
    elif hw["backend"] == "mps":
        print(f"   GPU Vendor  : Apple (Metal Performance Shaders)")
        print(f"   GPU Device  : {hw['gpu_name']}")
        print(f"   Unified Mem : ~{hw['vram_gb']:.1f} GB (estimated GPU share)")
    elif hw["backend"] == "xpu":
        print(f"   GPU Vendor  : Intel (XPU / IPEX)")
        print(f"   GPU Device  : {hw['gpu_name']}")
        print(f"   GPU VRAM    : {hw['vram_gb']:.1f} GB")
        if hw["gpu_count"] > 1:
            print(f"   GPU Count   : {hw['gpu_count']}")
    else:
        gpu_label = hw["gpu_name"] if hw["gpu_name"] != "CPU Only" else "None detected"
        print(f"   GPU         : {gpu_label}")
        print(f"   Compute     : CPU Only (no accelerator)")
    print("-" * 60)

    # Pilih profil
    if size_override and size_override in MODEL_PROFILES:
        profile_name = size_override
        print(f"   Manual Profile Selected: {size_override.upper()}")
    else:
        # Auto-detect berdasarkan RAM
        if total_ram >= 48.0:
            profile_name = "promax"
        elif total_ram >= 32.0:
            profile_name = "ultra"
        elif total_ram >= 16.0:
            profile_name = "large"
        elif total_ram >= 8.0:
            profile_name = "medium"
        else:
            profile_name = "small"

    promax_tier = None
    if profile_name == "promax":
        from core.promax import resolve_promax_tier, get_promax_profile
        hw_force = force or is_force_mode()
        tier_override = os.environ.get("SPACEAX_PROMAX_TIER")
        promax_tier = resolve_promax_tier(
            total_ram,
            vram_gb,
            force_tier=tier_override,
            hardware_force=hw_force,
        )
        profile = get_promax_profile(promax_tier)
        if hw_force and total_ram < profile["min_ram_gb"]:
            print(
                f"   [FORCE MODE]: Using {promax_tier} "
                f"(RAM {total_ram:.1f} GB < recommended {profile['min_ram_gb']:.0f} GB)."
            )
    else:
        profile = MODEL_PROFILES[profile_name]

    hw_force = force or is_force_mode()
    if (
        not hw_force
        and not size_override
        and total_ram < profile["min_ram_gb"]
    ):
        print(
            f"   Notice: System RAM may be insufficient for profile {profile_name.upper()} "
            f"(needs {profile['min_ram_gb']}GB, found {total_ram:.1f}GB)"
        )
        print("   Adjusting to safe model profile...")
        for pname in ["small", "medium", "large", "ultra", "promax"]:
            if total_ram >= MODEL_PROFILES[pname]["min_ram_gb"]:
                profile_name = pname
                profile = MODEL_PROFILES[pname]
                if profile_name == "promax":
                    from core.promax import resolve_promax_tier, get_promax_profile
                    promax_tier = resolve_promax_tier(total_ram, vram_gb, force_tier=None)
                    profile = get_promax_profile(promax_tier)

    cfg = ModelConfig(
        d_model=profile["d_model"],
        n_heads=profile["n_heads"],
        n_kv_heads=profile.get("n_kv_heads", profile["n_heads"]),
        n_layers=profile["n_layers"],
        d_ff=profile["d_ff"],
        max_seq_len=profile["max_seq_len"],
        vocab_size=profile["vocab_size"],
        n_experts=profile.get("n_experts", 1),
        n_active_experts=profile.get("n_active_experts", 1),
        vision_enabled=profile.get("vision_enabled", False),
        use_gradient_checkpointing=profile_name in ["ultra", "promax"],
    )
    batch = profile["batch_size"]
    label = profile["label"]

    print(f"   Model Profile: {label}")
    if promax_tier:
        print(f"      ProMax Tier: {promax_tier}")
    print(f"      d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"n_kv_heads={cfg.n_kv_heads}, n_layers={cfg.n_layers}, d_ff={cfg.d_ff}")
    print(f"      vocab_size={cfg.vocab_size}, max_seq_len={cfg.max_seq_len}")
    print(f"      MoE: {cfg.n_experts} experts (top-{cfg.n_active_experts})")
    if cfg.vision_enabled:
        print(f"      Vision: ENABLED")

    return cfg, batch, label, promax_tier, profile_name


@dataclass
class ModelConfig:
    """Konfigurasi arsitektur model."""
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4          # Grouped Query Attention (GQA)
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 512
    vocab_size: int = 96000
    dropout: float = 0.1
    rope_theta: float = 10000.0
    use_gradient_checkpointing: bool = False
    # Mixture of Experts
    n_experts: int = 2           # Total expert count per layer
    n_active_experts: int = 1    # Top-K experts to activate
    moe_aux_loss_weight: float = 0.01  # Load balancing loss coefficient
    # Vision
    vision_enabled: bool = True
    vision_patch_size: int = 16
    vision_image_size: int = 224
    vision_n_layers: int = 2     # Lightweight ViT layers

@dataclass
class TrainingConfig:
    """Konfigurasi untuk proses training."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Efektif batch size = batch_size * ini
    learning_rate: float = 8e-4           # Agresif untuk Adafactor (internal adaptive scaling)
    num_epochs: int = 20                  # Lebih banyak epoch untuk dataset besar
    warmup_steps: int = 200               # Warmup cepat agar model mulai belajar sejak batch awal
    grad_clip: float = 1.0
    weight_decay: float = 0.05            # Regularisasi lebih kuat mencegah overfitting
    label_smoothing: float = 0.05         # Rendah: konvergensi awal lebih cepat
    checkpoint_interval: int = 500
    fp16: bool = True                     # Aktifkan mixed precision di GPU
    use_bfloat16_cpu: bool = False
    num_workers: int = 0
    optimizer_type: str = "adamw"          # "adamw" atau "adafactor"
    early_stopping_patience: int = 5
    force_train: bool = False
    min_lr_ratio: float = 0.1             # Cosine decay ke 10% base_lr (bukan 0)
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_start_seq_len: int = 256   # Mulai dari seq pendek
    curriculum_warmup_epochs: int = 3     # Naik ke max_seq_len setelah N epoch
    # Packing
    use_packing: bool = True              # Gabung sequence pendek jadi 1 row

@dataclass
class PathConfig:
    """Konfigurasi path direktori."""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    seed_dir: str = os.path.join(data_dir, "seed")
    checkpoints_dir: str = os.path.join(data_dir, "checkpoints")
    knowledge_dir: str = os.path.join(data_dir, "knowledge")
    memories_dir: str = os.path.join(data_dir, "memories")
    vocab_dir: str = os.path.join(data_dir, "vocab")
    personality_dir: str = os.path.join(data_dir, "personality")
    kbbi_dir: str = os.path.join(base_dir, "kbbi")
    assets_dir: str = os.path.join(base_dir, "assets")
    vision_train_dir: str = os.path.join(assets_dir, "vision_train")

    def ensure_dirs(self):
        dirs = [self.data_dir, self.seed_dir, self.checkpoints_dir,
                self.knowledge_dir, self.memories_dir, self.vocab_dir,
                self.personality_dir, self.assets_dir, self.vision_train_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

# Identitas AI
AI_IDENTITY = {
    "name": "SpaceAx AI",
    "team": "Space Ax Corp",
    "developer": "Thomas Alfareno Ananta Nugraha",
    "university": "Institut Teknologi Sepuluh Nopember Surabaya",
    "faculty": "Fakultas Teknologi Elektro dan Informatika Cerdas (FTEIC)",
    "department": "Departemen Teknik Informatika",
    "program": "Prodi Teknik Informatika",
    "version": "3.0.0",
}

@dataclass
class EmotionConfig:
    emotions: list[str] = field(default_factory=lambda: [
        "joy", "sadness", "anger", "fear",
        "surprise", "disgust", "trust", "anticipation", "neutral"
    ])
    default_emotion: str = "neutral"
    decay_rate: float = 0.05

def get_config(auto_detect: bool = True, size_override: str = None, force: bool = False):
    """Dapatkan semua config sekaligus.
    
    Args:
        auto_detect: Jika True, auto-detect hardware untuk pilih profil model
        size_override: Override manual profil model ('small', 'medium', 'large', 'ultra')
    """
    paths = PathConfig()
    paths.ensure_dirs()

    if auto_detect:
        force = force or is_force_mode()
        model_cfg, batch_size, label, promax_tier, profile_name = auto_model_config(
            size_override, force=force
        )
        training_cfg = TrainingConfig(batch_size=batch_size, force_train=force)

        if promax_tier or size_override == "promax":
            from core.promax import apply_promax_training_overrides
            apply_promax_training_overrides(training_cfg)
            print(
                f"   ProMax training: epochs>={training_cfg.num_epochs}, "
                f"warmup={training_cfg.warmup_steps}, "
                f"accum={training_cfg.gradient_accumulation_steps}"
            )
        
        # CPU Mode Optimization
        if not torch.cuda.is_available():
            print("   CPU Mode detected: Scaling down batch size.")
            old_batch = training_cfg.batch_size
            training_cfg.batch_size = min(4, max(1, old_batch // 4))
            factor = max(1, old_batch // training_cfg.batch_size)
            training_cfg.gradient_accumulation_steps = training_cfg.gradient_accumulation_steps * factor
            training_cfg.use_bfloat16_cpu = False
            print(f"      Batch Size adjusted: {old_batch} -> {training_cfg.batch_size}")
            print(f"      Gradient Accumulation Steps: {training_cfg.gradient_accumulation_steps}")
        else:
            vram = get_gpu_vram_gb()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            if promax_tier == "promax_8b":
                from core.vram_fit import apply_promax_8b_vram_fit
                apply_promax_8b_vram_fit(model_cfg, training_cfg, vram)
            elif vram > 0:
                # Aktifkan gradient checkpointing jika VRAM <= 20GB untuk menghemat memori GPU
                if vram <= 20.0:
                    model_cfg.use_gradient_checkpointing = True
                    
                    # Target safe micro batch per pass untuk mencegah VRAM overflow saat backward pass
                    max_safe_micro = 4 if model_cfg.d_model <= 512 else (2 if model_cfg.d_model <= 1024 else 1)
                    if training_cfg.batch_size > max_safe_micro:
                        old_batch = training_cfg.batch_size
                        old_accum = training_cfg.gradient_accumulation_steps
                        training_cfg.batch_size = max_safe_micro
                        training_cfg.gradient_accumulation_steps = max(1, (old_batch * old_accum) // training_cfg.batch_size)
                        print(
                            f"   VRAM Auto-Tune (VRAM {vram:.1f} GB): "
                            f"Micro batch adjusted ({old_batch} -> {training_cfg.batch_size}) "
                            f"with accum ({old_accum} -> {training_cfg.gradient_accumulation_steps}) "
                            f"to prevent CUDA OOM."
                        )
                elif vram >= 75.0:
                    multiplier = 8
                    old_batch = training_cfg.batch_size
                    training_cfg.batch_size = old_batch * multiplier
                    old_accum = training_cfg.gradient_accumulation_steps
                    training_cfg.gradient_accumulation_steps = max(1, old_accum // multiplier)
                elif vram >= 38.0:
                    multiplier = 4
                    old_batch = training_cfg.batch_size
                    training_cfg.batch_size = old_batch * multiplier
                    old_accum = training_cfg.gradient_accumulation_steps
                    training_cfg.gradient_accumulation_steps = max(1, old_accum // multiplier)
                elif vram >= 20.0:
                    multiplier = 2
                    old_batch = training_cfg.batch_size
                    training_cfg.batch_size = old_batch * multiplier
                    old_accum = training_cfg.gradient_accumulation_steps
                    training_cfg.gradient_accumulation_steps = max(1, old_accum // multiplier)
        
        # Tentukan optimizer_type otomatis
        total_ram = get_system_ram_gb()
        is_large_model = size_override in ["ultra", "promax"] or promax_tier or (
            size_override is None and total_ram >= 32.0
        )
        if is_large_model or total_ram < 16.0:
            training_cfg.optimizer_type = "adafactor"
            print(f"   Memory Optimization: Enabled Adafactor low-memory optimizer.")
        else:
            training_cfg.optimizer_type = "adamw"
    else:
        model_cfg = ModelConfig()
        training_cfg = TrainingConfig()
        profile_name = size_override or "medium"

    return {
        "model": model_cfg,
        "training": training_cfg,
        "paths": paths,
        "emotion": EmotionConfig(),
        "identity": AI_IDENTITY,
        "profile_name": profile_name if auto_detect else "medium",
        "promax_tier": promax_tier if auto_detect else None,
        "is_promax": bool(promax_tier or size_override == "promax"),
    }
