"""Exact fp16 inference for models larger than the GPU, one decoder layer at a time.

All texts of a dataset are embedded first and kept on CPU. Each decoder layer is then loaded to the GPU
from the safetensors shards, applied to every text, and freed. The final norm and lm_head run last.
The arithmetic is the same as a normal fp16 forward pass (fp16 residual stream between layers, causal
SDPA attention, no padding), so scores match a full-GPU fp16 run up to kernel nondeterminism.
"""
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


class LayerwiseLM:
    def __init__(self, name, dtype=torch.float16, device="cuda"):
        self.dtype, self.device = dtype, device
        self.dir = Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors"]))
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.where = json.load(open(index, encoding="utf-8"))["weight_map"]
        else:
            with safe_open(self.dir / "model.safetensors", "pt") as f:
                self.where = {k: "model.safetensors" for k in f.keys()}
        cfg = AutoConfig.from_pretrained(name)
        with init_empty_weights(include_buffers=False):
            self.model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="sdpa")
        self.model.eval()
        self.cfg = cfg
        self.model.model.rotary_emb.to(device)

    def _tensor(self, key):
        with safe_open(self.dir / self.where[key], "pt", device="cpu") as f:
            return f.get_tensor(key).to(self.dtype)

    def _load(self, module_prefix, device):
        for key in [k for k in self.where if k.startswith(module_prefix + ".")]:
            set_module_tensor_to_device(self.model, key, device, value=self._tensor(key))

    def _free(self, module):
        for name, _ in list(module.named_parameters()):
            set_module_tensor_to_device(module, name, "meta")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def logits(self, input_ids_list, progress=None):
        """input_ids_list: list of 1-D LongTensors. Yields (index, logits[T, V] fp16 on GPU) in order."""
        emb_w = self._tensor("model.embed_tokens.weight")  # CPU
        hidden = [emb_w[ids].unsqueeze(0) for ids in input_ids_list]  # each [1, T, H] fp16 on CPU
        rotary = self.model.model.rotary_emb
        for i, layer in enumerate(self.model.model.layers):
            self._load(f"model.layers.{i}", self.device)
            for j, h in enumerate(hidden):
                x = h.to(self.device)
                pos = torch.arange(x.size(1), device=self.device).unsqueeze(0)
                out = layer(x, attention_mask=None, position_ids=pos, position_embeddings=rotary(x, pos))
                hidden[j] = (out[0] if isinstance(out, tuple) else out).cpu()
            self._free(layer)
            if progress:
                progress(i + 1, len(self.model.model.layers))
        self._load("model.norm", self.device)
        head_key = "lm_head.weight" if "lm_head.weight" in self.where else "model.embed_tokens.weight"  # tied
        head_w = (emb_w if head_key == "model.embed_tokens.weight" else self._tensor(head_key)).to(self.device)
        del emb_w
        for j, h in enumerate(hidden):
            x = self.model.model.norm(h.to(self.device))
            yield j, torch.nn.functional.linear(x, head_w)[0]
        self._free(self.model.model.norm)
        del head_w
        torch.cuda.empty_cache()
