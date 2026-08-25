"""
SpaceaxAI - Transformer Model v3.0
Implementasi arsitektur LLM modern dari nol menggunakan PyTorch.
Dilengkapi: GQA (Grouped Query Attention), SwiGLU, RMSNorm, QK-Norm, MoE (Mixture of Experts), KV Cache, Vision Integration.
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List, Dict, Any

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frekuensi untuk Rotary Positional Embedding (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Terapkan RoPE pada query dan key."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Tangani match dimensi
    if freqs_cis.shape[1] < xq_.shape[1]:
        # Expand jika seandainya sequence melampaui precomputed
        freqs_cis = freqs_cis[:, :xq_.shape[1]]

    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :xq_.shape[1]]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :xk_.shape[1]]).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key & Value heads untuk Grouped Query Attention (GQA)."""
    if n_rep == 1:
        return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) dengan RoPE, QK-Norm, dan KV Cache."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if getattr(config, 'n_kv_heads', None) else config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_heads

        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # QK-Norm untuk kestabilan training
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # QK Normalization sebelum RoPE
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)

        new_kv_cache = (xk, xv)

        # Repeat KV heads jika GQA (n_kv_heads < n_heads)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        dropout_p = self.dropout.p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), new_kv_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU FeedForward Network."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_ff = config.d_ff
        self.w1 = nn.Linear(config.d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEFeedForward(nn.Module):
    """Mixture of Experts FeedForward Network dengan Top-K routing & load balancing loss."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = getattr(config, 'n_experts', 1)
        self.top_k = getattr(config, 'n_active_experts', 1)
        self.d_model = config.d_model

        if self.num_experts <= 1:
            self.experts = nn.ModuleList([SwiGLUFFN(config)])
            self.gate = None
        else:
            self.experts = nn.ModuleList([SwiGLUFFN(config) for _ in range(self.num_experts)])
            self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gate is None or self.num_experts <= 1:
            return self.experts[0](x), torch.tensor(0.0, device=x.device)

        bsz, seqlen, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (N, d_model) where N = bsz * seqlen

        # Router logits & softmax
        gate_logits = self.gate(x_flat)  # (N, num_experts)
        routing_weights = F.softmax(gate_logits, dim=-1)

        # Top-K selection
        weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)  # Renormalize

        # Compute load balancing loss (Auxiliary loss)
        # density = fraction of tokens routed to expert e
        # fraction = average gate probability for expert e
        density = torch.zeros(self.num_experts, device=x.device)
        density.scatter_add_(0, selected_experts.view(-1), torch.ones_like(selected_experts.view(-1), dtype=torch.float))
        density = density / (x_flat.shape[0] * self.top_k)

        fraction = routing_weights.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * fraction)

        # Expert execution (Vectorized & Checkpoint-Safe)
        # Selalu jalankan semua expert (tanpa skip) agar deterministic saat gradient checkpointing recompute.
        # Weight = 0 untuk expert yang tidak di-route secara otomatis meng-nolkan kontribusinya.
        final_output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            expert_mask = (selected_experts == e).float()
            expert_weight = (weights * expert_mask).sum(dim=-1, keepdim=True)  # (N, 1)
            expert_out = self.experts[e](x_flat)
            final_output = final_output + expert_weight * expert_out

        return final_output.view(bsz, seqlen, d_model), aux_loss


class TransformerBlock(nn.Module):
    """Transformer Layer dengan GQA, MoE FFN, dan RMSNorm."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = MoEFeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        h, new_kv_cache = self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache)
        x = x + h
        ffn_out, aux_loss = self.feed_forward(self.ffn_norm(x))
        x = x + ffn_out
        return x, new_kv_cache, aux_loss


class SpaceaxModel(nn.Module):
    """Model LLM Modern SpaceaxAI v3.0 (Cerdas, Compact, MoE + Vision Ready)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)

        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (Embedding & Output Head)
        self.tok_embeddings.weight = self.output.weight

        # Precompute RoPE
        head_dim = config.d_model // config.n_heads
        self.freqs_cis = precompute_freqs_cis(head_dim, config.max_seq_len * 2, config.rope_theta)

        # Vision Encoder (optional, lazy or initialized if config.vision_enabled)
        self.vision_encoder = None
        if getattr(config, 'vision_enabled', False):
            try:
                from .vision import VisionEncoder
                self.vision_encoder = VisionEncoder(
                    d_model=config.d_model,
                    patch_size=getattr(config, 'vision_patch_size', 16),
                    image_size=getattr(config, 'vision_image_size', 224),
                    n_layers=getattr(config, 'vision_n_layers', 2),
                )
            except ImportError:
                pass

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        images: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass SpaceAX Model.
        Args:
            tokens: (bsz, seqlen) token IDs
            start_pos: posisi awal untuk KV cache / inference
            kv_caches: cache per-layer
            images: (bsz, 3, H, W) tensor gambar opsional untuk vision mode
        """
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Scale embeddings dengan sqrt(d_model) untuk stabilitas
        h = h * math.sqrt(self.config.d_model)

        # Terapkan vision embeddings jika ada gambar dan vision_encoder aktif
        if images is not None and self.vision_encoder is not None:
            v_embeds = self.vision_encoder(images)  # (bsz, num_patches, d_model)
            # Prepend/Blend vision embeddings ke awal sequence
            if seqlen >= v_embeds.shape[1]:
                h[:, :v_embeds.shape[1]] = h[:, :v_embeds.shape[1]] + v_embeds

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        total_aux_loss = torch.tensor(0.0, device=tokens.device)
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            if self.training and h.requires_grad and getattr(self.config, "use_gradient_checkpointing", False):
                # Factory function agar setiap `layer` ter-bind ke closure yang benar
                # (menghindari Python late-binding closure bug dalam loop)
                def make_custom_forward(block):
                    def custom_forward(*inputs):
                        return block(*inputs)
                    return custom_forward
                h, new_cache, aux = checkpoint(
                    make_custom_forward(layer), h, freqs_cis, mask, cache,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                h, new_cache, aux = layer(h, freqs_cis, mask, cache)

            new_kv_caches.append(new_cache)
            total_aux_loss = total_aux_loss + aux

        h = self.norm(h)
        logits = self.output(h)

        if self.training:
            return logits, new_kv_caches, total_aux_loss
        return logits, new_kv_caches

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        max_gen_len: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_id: int = 2,
        images: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.1,
    ) -> List[int]:
        """Fungsi generasi autoregresif dengan KV Cache dan repetition penalty."""
        device = next(self.parameters()).device

        max_prompt = self.config.max_seq_len - max_gen_len - 5
        if len(prompt_tokens) > max_prompt:
            prompt_tokens = prompt_tokens[-max_prompt:]

        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        kv_caches = None
        start_pos = 0

        generated_tokens = []
        all_tokens = list(prompt_tokens)

        for _ in range(max_gen_len):
            if start_pos == 0 and images is not None:
                logits, kv_caches = self(tokens, start_pos=start_pos, kv_caches=kv_caches, images=images)
            else:
                logits, kv_caches = self(tokens, start_pos=start_pos, kv_caches=kv_caches)

            next_token_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(all_tokens):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1).item()
            else:
                next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated_tokens.append(next_token)
            all_tokens.append(next_token)

            if next_token == eos_id:
                break

            tokens = torch.tensor([[next_token]], dtype=torch.long, device=device)
            start_pos += logits.shape[1]

        return generated_tokens

    def count_parameters(self) -> int:
        """Hitung jumlah parameter dalam model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
