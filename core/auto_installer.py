"""
SpaceaxAI - Smart Auto-Installer & Hardware Optimizer v3.0
Mendeteksi hardware (CPU/RAM/GPU/CUDA) dan menginstall/memverifikasi dependensi
Python yang paling optimal secara otomatis, bahkan jika modul sudah terpasang.
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import sys
import os
import subprocess
import shutil
import platform


def detect_system_gpu():
    """Deteksi hardware GPU pada tingkat sistem (OS/Driver).

    Returns:
        tuple: (vendor, cuda_version_str)
               vendor: 'nvidia', 'amd', 'apple', 'intel', atau None
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output([nvidia_smi], text=True, timeout=5)
            if "NVIDIA" in out:
                if "CUDA Version: 12" in out:
                    return "nvidia", "cu121"
                elif "CUDA Version: 11" in out:
                    return "nvidia", "cu118"
                return "nvidia", "cu121"
        except Exception:
            pass

    # Check AMD ROCm
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            out = subprocess.check_output([rocm_smi], text=True, timeout=5)
            if "AMD" in out or "GPU" in out:
                return "amd", "rocm6.0"
        except Exception:
            pass

    # Check Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple", "mps"

    return None, None


def verify_pytorch_optimality():
    """Memeriksa apakah PyTorch yang terinstall sudah optimal untuk hardware GPU yang ada.

    Returns:
        bool: True jika PyTorch sudah optimal, False jika perlu dipasang ulang dengan GPU wheel.
    """
    try:
        import torch
    except ImportError:
        return False

    vendor, cuda_ver = detect_system_gpu()

    # Jika hardware punya NVIDIA GPU tapi PyTorch yang terinstall adalah CPU-only
    if vendor == "nvidia" and not torch.cuda.is_available():
        print(f"\n[OPTIMIZER] NVIDIA GPU detected in system, but current PyTorch build is CPU-only.")
        print(f"[OPTIMIZER] Auto-reinstalling optimal PyTorch with CUDA acceleration ({cuda_ver})...")
        return False

    # Jika hardware punya AMD GPU tapi PyTorch CUDA/HIP tidak aktif
    if vendor == "amd" and not torch.cuda.is_available():
        print(f"\n[OPTIMIZER] AMD GPU detected in system, but current PyTorch build lacks ROCm support.")
        return False

    return True


def check_and_install_dependencies():
    """Memeriksa dan menginstall/mengoptimalkan modul Python secara otomatis."""
    required_modules = {
        "torch": "torch",
        "tokenizers": "tokenizers",
        "rich": "rich",
        "PIL": "Pillow",
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "easyocr": "easyocr",
    }

    missing = []
    for mod, pkg in required_modules.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, pkg))

    torch_optimal = verify_pytorch_optimality()
    if not torch_optimal and ("torch", "torch") not in missing:
        missing.append(("torch", "torch"))

    if not missing and torch_optimal:
        return True

    print("=" * 65)
    print("SpaceAX AI — Smart Environment & Hardware Optimizer v3.0")
    print("=" * 65)
    if missing:
        print(f"Target packages to process: {[pkg for _, pkg in missing]}")
    print("Auditing compute acceleration and environment build...")

    vendor, cuda_ver = detect_system_gpu()

    # Build PyTorch command based on detected accelerator
    if vendor == "nvidia" and cuda_ver:
        print(f"-> Selected target: PyTorch CUDA build ({cuda_ver}) for NVIDIA GPU")
        torch_install_cmd = [
            sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision",
            "--extra-index-url", f"https://download.pytorch.org/whl/{cuda_ver}"
        ]
    elif vendor == "amd":
        print(f"-> Selected target: PyTorch ROCm build for AMD GPU")
        torch_install_cmd = [
            sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/rocm6.0"
        ]
    else:
        print("-> Selected target: Standard PyTorch build")
        torch_install_cmd = [
            sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision"
        ]

    # Install/upgrade PyTorch if needed
    if any(mod == "torch" for mod, _ in missing):
        try:
            subprocess.check_call(torch_install_cmd)
        except Exception as e:
            print(f"Accelerated PyTorch installation notice: {e}. Falling back to standard PyTorch...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torch"])

    # Install remaining packages
    other_pkgs = [pkg for mod, pkg in missing if mod != "torch"]
    if other_pkgs:
        print(f"Installing missing dependencies: {', '.join(other_pkgs)}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + other_pkgs)

    print("Environment setup and optimization completed successfully.")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    check_and_install_dependencies()
