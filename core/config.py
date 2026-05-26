import os
import platform
import multiprocessing
import torch
from dataclasses import dataclass, field
try:
    cores = multiprocessing.cpu_count()
    torch.set_num_threads(max(1, cores - 1))
except Exception:
    pass

def get_system_ram_gb() -> float:
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 4.0

def get_available_ram_gb() -> float:
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
MODEL_PROFILES = {'small': {'d_model': 384, 'n_heads': 6, 'n_layers': 6, 'd_ff': 1536, 'max_seq_len': 512, 'vocab_size': 64000, 'batch_size': 16, 'label': 'SMALL (~25M params)', 'min_ram_gb': 4.0}, 'medium': {'d_model': 768, 'n_heads': 12, 'n_layers': 12, 'd_ff': 3072, 'max_seq_len': 1024, 'vocab_size': 64000, 'batch_size': 8, 'label': 'MEDIUM (~100M params)', 'min_ram_gb': 8.0}, 'large': {'d_model': 1024, 'n_heads': 16, 'n_layers': 16, 'd_ff': 4096, 'max_seq_len': 1024, 'vocab_size': 64000, 'batch_size': 4, 'label': 'LARGE (~250M params)', 'min_ram_gb': 16.0}, 'ultra': {'d_model': 1280, 'n_heads': 20, 'n_layers': 24, 'd_ff': 5120, 'max_seq_len': 1024, 'vocab_size': 64000, 'batch_size': 2, 'label': 'ULTRA (~650M params)', 'min_ram_gb': 32.0}}

def auto_model_config(size_override: str=None):
    total_ram = get_system_ram_gb()
    avail_ram = get_available_ram_gb()
    print(f'🖥️  Hardware terdeteksi:')
    print(f'   Total RAM: {total_ram:.1f} GB')
    print(f'   RAM Tersedia: {avail_ram:.1f} GB')
    print(f'   OS: {platform.system()} {platform.machine()}')
    print(f'   CPU Cores: {multiprocessing.cpu_count()}')
    print(f"   CUDA: {('✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ Tidak tersedia (CPU mode)')}")
    if size_override and size_override in MODEL_PROFILES:
        profile_name = size_override
        print(f'   📌 Profil manual: {size_override.upper()}')
    elif total_ram >= 32.0:
        profile_name = 'ultra'
    elif total_ram >= 16.0:
        profile_name = 'large'
    elif total_ram >= 8.0:
        profile_name = 'medium'
    else:
        profile_name = 'small'
    profile = MODEL_PROFILES[profile_name]
    if total_ram < profile['min_ram_gb']:
        print(f"   ⚠️  RAM mungkin tidak cukup untuk profil {profile_name.upper()} (butuh {profile['min_ram_gb']}GB, punya {total_ram:.1f}GB)")
        print(f'   ⚠️  Menurunkan ke profil yang lebih kecil...')
        for pname in ['small', 'medium', 'large', 'ultra']:
            if total_ram >= MODEL_PROFILES[pname]['min_ram_gb']:
                profile_name = pname
                profile = MODEL_PROFILES[pname]
    cfg = ModelConfig(d_model=profile['d_model'], n_heads=profile['n_heads'], n_layers=profile['n_layers'], d_ff=profile['d_ff'], max_seq_len=profile['max_seq_len'], vocab_size=profile['vocab_size'])
    batch = profile['batch_size']
    label = profile['label']
    print(f'   🧠 Profil Model: {label}')
    print(f'      d_model={cfg.d_model}, n_heads={cfg.n_heads}, n_layers={cfg.n_layers}, d_ff={cfg.d_ff}')
    print(f'      vocab_size={cfg.vocab_size}, max_seq_len={cfg.max_seq_len}')
    return (cfg, batch, label)

@dataclass
class ModelConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    vocab_size: int = 64000
    dropout: float = 0.1
    rope_theta: float = 10000.0

@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 0.0003
    num_epochs: int = 20
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    checkpoint_interval: int = 500
    fp16: bool = True
    use_bfloat16_cpu: bool = False
    num_workers: int = 0
    early_stopping_patience: int = 5

@dataclass
class PathConfig:
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, 'data')
    seed_dir: str = os.path.join(data_dir, 'seed')
    checkpoints_dir: str = os.path.join(data_dir, 'checkpoints')
    knowledge_dir: str = os.path.join(data_dir, 'knowledge')
    memories_dir: str = os.path.join(data_dir, 'memories')
    vocab_dir: str = os.path.join(data_dir, 'vocab')
    personality_dir: str = os.path.join(data_dir, 'personality')
    kbbi_dir: str = os.path.join(base_dir, 'kbbi')

    def ensure_dirs(self):
        dirs = [self.data_dir, self.seed_dir, self.checkpoints_dir, self.knowledge_dir, self.memories_dir, self.vocab_dir, self.personality_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
AI_IDENTITY = {'name': 'SpaceAx AI', 'team': 'Space Ax Corp', 'developer': 'Thomas Alfareno Ananta Nugraha', 'university': 'Institut Teknologi Sepuluh Nopember Surabaya', 'faculty': 'Fakultas Teknologi Elektro dan Informatika Cerdas (FTEIC)', 'department': 'Departemen Teknik Informatika', 'program': 'Prodi Teknik Informatika', 'version': '2.0.0'}

@dataclass
class EmotionConfig:
    emotions: list[str] = field(default_factory=lambda: ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation', 'neutral'])
    default_emotion: str = 'neutral'
    decay_rate: float = 0.05

def get_config(auto_detect: bool=True, size_override: str=None):
    paths = PathConfig()
    paths.ensure_dirs()
    if auto_detect:
        model_cfg, batch_size, label = auto_model_config(size_override)
        training_cfg = TrainingConfig(batch_size=batch_size)
        if not torch.cuda.is_available():
            print('   ⚠️  CPU Mode terdeteksi: Menurunkan batch size.')
            old_batch = training_cfg.batch_size
            training_cfg.batch_size = min(4, max(1, old_batch // 4))
            factor = max(1, old_batch // training_cfg.batch_size)
            training_cfg.gradient_accumulation_steps = training_cfg.gradient_accumulation_steps * factor
            training_cfg.use_bfloat16_cpu = False
            print(f'      Batch Size disesuaikan: {old_batch} → {training_cfg.batch_size}')
            print(f'      Gradient Accumulation Steps: {training_cfg.gradient_accumulation_steps}')
    else:
        model_cfg = ModelConfig()
        training_cfg = TrainingConfig()
    return {'model': model_cfg, 'training': training_cfg, 'paths': paths, 'emotion': EmotionConfig(), 'identity': AI_IDENTITY}