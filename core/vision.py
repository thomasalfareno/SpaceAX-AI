"""
SpaceaxAI - Vision Module v3.0
Ringan & cepat: Vision Transformer (ViT) Encoder + Projector ke d_model SpaceAX.
Memungkinkan model SMALL maupun model besar melihat dan memahami gambar!
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


class PatchEmbedding(nn.Module):
    """Membagi gambar (B, C, H, W) menjadi patch (B, num_patches, d_model)."""
    def __init__(self, in_channels: int = 3, patch_size: int = 16, image_size: int = 224, embed_dim: int = 384):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bicubic', align_corners=False)
            
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed[:, :x.size(1)]
        return x


class VisionBlock(nn.Module):
    """Self-Attention block untuk Vision Transformer."""
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    Lightweight Vision Encoder & Projector.
    Menerima gambar tensor (B, 3, 224, 224) -> mengembalikan (B, num_patches, d_model) LLM.
    """
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 16,
        image_size: int = 224,
        n_layers: int = 2,
        vit_dim: int = 384
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, image_size=image_size, embed_dim=vit_dim)
        self.blocks = nn.ModuleList([VisionBlock(dim=vit_dim, n_heads=6) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(vit_dim)
        
        # Projector dari vit_dim ke LLM d_model
        self.projector = nn.Sequential(
            nn.Linear(vit_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) float32 normalized image tensor [0..1]
        Returns:
            visual_embeds: (B, num_patches+1, d_model)
        """
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        visual_tokens = self.projector(x)
        return visual_tokens


def load_and_preprocess_image(image_path_or_url: str) -> Optional[torch.Tensor]:
    """Membaca gambar lokal atau URL dan mengonversi menjadi Tensor RGB (1, 3, 224, 224)."""
    try:
        from PIL import Image
        import numpy as np
        import requests
        from io import BytesIO

        path_clean = image_path_or_url.strip().strip("'\"")

        if path_clean.startswith("http://") or path_clean.startswith("https://"):
            resp = requests.get(path_clean, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            if not os.path.exists(path_clean):
                return None
            img = Image.open(path_clean).convert("RGB")

        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
        return tensor
    except Exception as e:
        print(f"Error loading image '{image_path_or_url}': {e}")
        return None


# Cache global untuk EasyOCR Reader agar tidak re-initialize berulang kali
_EASYOCR_READER = None


def extract_ocr_text(image_path_or_url: str) -> Optional[str]:
    """Ekstraksi teks (OCR) dari gambar lokal atau URL secara presisi dan akurat.
    
    Mendukung EasyOCR (Bahasa Indonesia + Inggris), PyTesseract, dan fallback.
    """
    global _EASYOCR_READER
    try:
        from PIL import Image
        import requests
        from io import BytesIO

        path_clean = image_path_or_url.strip().strip("'\"")
        if path_clean.startswith("http://") or path_clean.startswith("https://"):
            resp = requests.get(path_clean, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            if not os.path.exists(path_clean):
                return None
            img = Image.open(path_clean).convert("RGB")

        # 1. Coba EasyOCR (Sangat akurat untuk Bahasa Indonesia & Inggris)
        try:
            import easyocr
            import numpy as np
            if _EASYOCR_READER is None:
                use_gpu = torch.cuda.is_available()
                _EASYOCR_READER = easyocr.Reader(['id', 'en'], gpu=use_gpu)
            
            img_np = np.array(img)
            results = _EASYOCR_READER.readtext(img_np, detail=0)
            if results:
                ocr_output = "\n".join(results).strip()
                if ocr_output:
                    return ocr_output
        except ImportError:
            pass
        except Exception as e:
            print(f"EasyOCR warning: {e}")

        # 2. Coba PyTesseract (Fallback jika Tesseract terpasang)
        try:
            import pytesseract
            ocr_output = pytesseract.image_to_string(img, lang='ind+eng').strip()
            if ocr_output:
                return ocr_output
        except ImportError:
            pass
        except Exception:
            pass

        return None
    except Exception as e:
        print(f"Error OCR image '{image_path_or_url}': {e}")
        return None

