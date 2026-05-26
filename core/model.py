import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List
from .config import ModelConfig

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return (xq_out.type_as(xq), xk_out.type_as(xk))

class Attention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_heads
        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = (self.wq(x), self.wk(x), self.wv(x))
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)
        new_kv_cache = (xk, xv)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return (self.wo(output), new_kv_cache)

class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        h, new_kv_cache = self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return (x, new_kv_cache)

class SpaceaxModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.freqs_cis = precompute_freqs_cis(config.d_model // config.n_heads, config.max_seq_len * 2, config.rope_theta)

    def forward(self, tokens: torch.Tensor, start_pos: int=0, kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            if self.training and h.requires_grad:

                def custom_forward(*inputs):
                    return layer(*inputs)
                h, new_cache = checkpoint(custom_forward, h, freqs_cis, mask, cache, use_reentrant=False)
            else:
                h, new_cache = layer(h, freqs_cis, mask, cache)
            new_kv_caches.append(new_cache)
        h = self.norm(h)
        output = self.output(h)
        return (output, new_kv_caches)

    @torch.inference_mode()
    def generate(self, prompt_tokens: List[int], max_gen_len: int, temperature: float=0.7, top_p: float=0.9, top_k: int=50, eos_id: int=2) -> List[int]:
        device = next(self.parameters()).device
        max_prompt = self.config.max_seq_len - max_gen_len - 5
        if len(prompt_tokens) > max_prompt:
            prompt_tokens = prompt_tokens[-max_prompt:]
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        kv_caches = None
        start_pos = 0
        generated_tokens = []
        for _ in range(max_gen_len):
            logits, kv_caches = self(tokens, start_pos, kv_caches)
            next_token_logits = logits[0, -1, :]
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
            if next_token == eos_id:
                break
            tokens = torch.tensor([[next_token]], dtype=torch.long, device=device)
            start_pos += logits.shape[1]
        return generated_tokens

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))