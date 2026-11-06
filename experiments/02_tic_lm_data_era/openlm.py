"""Minimal pure-PyTorch OpenLM decoder (inference only) for the TiC-LM Oracle checkpoints.

Follows mlfoundations/open_lm `Transformer` with the settings in the Oracles' params.txt:
gain-only LayerNorm (eps 1e-5), qk_norm over the full q/k projection before the head split, rotary on
[B, T, H, D] with rotate_half (NeoX style, base 10000), SwiGLU with fused w12, untied output layer.
open_lm itself imports xformers at module load, which is not available on this Windows setup, so the
forward pass is re-implemented here and checked against open_lm's own code (check_openlm.py).

Rotary tables: rope="fp32" exact; rope="inv_bf16" uses the bf16-rounded inverse frequencies that the
checkpoints store (pure-bf16 FSDP cast the buffer) with fp32 positions; rope="bf16" also rounds positions and
their product to bf16, as eager bf16 autocast would. check_openlm.py picks by loss at long positions.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

CONFIGS = {"open_lm_1b": dict(dim=2048, n_layers=24, n_heads=16, vocab_size=50432),
           "open_lm_3b": dict(dim=2560, n_layers=32, n_heads=32, vocab_size=50432)}
TOKENIZER = "EleutherAI/gpt-neox-20b"


def rope_tables(T, head_dim, mode, device):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(T, dtype=torch.float32)
    if mode == "inv_bf16":  # inverse frequencies as stored in the checkpoint (bf16-rounded), rest in fp32
        inv = inv.to(torch.bfloat16).float()
    if mode == "bf16":
        bf = torch.bfloat16
        freqs = (t.to(bf).float()[:, None] * inv.to(bf).float()[None, :]).to(bf).float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(bf).float(), emb.sin().to(bf).float()
    else:
        freqs = t[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
    return cos[None, :, None, :].to(device), sin[None, :, None, :].to(device)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, dim // n_heads
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.k_norm = nn.LayerNorm(dim, eps=1e-5, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (z.view(B, T, self.h, self.hd) for z in (q, k, v))
        c, s = cos[:, :T].to(q.dtype), sin[:, :T].to(q.dtype)
        q, k = q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, x = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * x)


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        hidden = 256 * ((int(2 * 4 * dim / 3) + 256 - 1) // 256)
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden)
        self.attention_norm = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-5, bias=False)

    def forward(self, x, cos, sin):
        h = x + self.attention(self.attention_norm(x), cos, sin)
        return h + self.feed_forward(self.ffn_norm(h))


class OpenLM(nn.Module):
    def __init__(self, dim, n_layers, n_heads, vocab_size, seq_len=2048, rope="fp32"):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.head_dim, self.seq_len, self.rope = dim // n_heads, seq_len, rope
        self._tables = None

    def forward(self, input_ids, attention_mask=None):
        if self._tables is None or self._tables[0].device != input_ids.device:
            self._tables = rope_tables(self.seq_len, self.head_dim, self.rope, input_ids.device)
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, *self._tables)
        return self.output(self.norm(x))


def load_state_dict(ckpt_path):
    """Read an OpenLM training checkpoint (pickle) with weights_only=True and strip wrapper prefixes."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    sd = ck.get("state_dict", ck)
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
            k = k.replace(p, "")
        out[k] = v
    return out, {k: v for k, v in ck.items() if k != "state_dict"}


def load_oracle(ckpt_dir, dtype=torch.float16, device="cuda", rope="fp32"):
    """ckpt_dir: .../oracles_1b_220b/<YYYYMM> (holds params.txt and checkpoints/epoch_*.pt)."""
    ckpt_dir = Path(ckpt_dir)
    params = dict(l.split(": ", 1) for l in (ckpt_dir / "params.txt").read_text().splitlines() if ": " in l)
    cfg = CONFIGS[params["model"]]
    assert params["model_norm"] == "gain_only_lp_layer_norm" and params["qk_norm"] == "True"
    assert params["positional_embedding_type"] == "rotary" and params["ffn_type"] == "swiglu_torch"
    (ckpt,) = sorted((ckpt_dir / "checkpoints").glob("epoch_*.pt"))
    sd, meta = load_state_dict(ckpt)
    sd = {k: v for k, v in sd.items() if not k.endswith("pos_embed.inv_freq")}
    with torch.device("meta"):
        model = OpenLM(**cfg, rope=rope)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not missing and not unexpected, (missing, unexpected)
    model = model.to(device=device, dtype=dtype).eval()
    return model, {"params": params, "ckpt": str(ckpt), "epoch": meta.get("epoch")}
