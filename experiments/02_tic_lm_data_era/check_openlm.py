"""Checks for openlm.py before any scoring run.

1) Architecture: our OpenLM(rope="fp32") vs open_lm's own Transformer with the same weights, both fp32 on CPU
   (open_lm builds exact fp32 rope tables on CPU) -> max |logit diff|.
2) Rope table precision: mean token NLL of human text with rope="fp32" vs rope="bf16" (tables as built under
   bf16 autocast in training), by position bucket. The variant that matches training has the lower loss
   at long positions, where bf16 rounding of the positions is coarse.
3) fp16 on GPU vs fp32 on CPU: Fast-DetectGPT / likelihood per text.

    git clone https://github.com/mlfoundations/open_lm <src>     # checked at 9bb92ef
    python experiments/02_tic_lm_data_era/check_openlm.py --ckpt_dir data/tic-lm/oracles_1b_220b/202407 --open_lm_src <src>
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "01_fastdetectgpt_1b")]  # also works under python -I
from run import DATA, scores_from_logits  # noqa: E402
from openlm import CONFIGS, TOKENIZER, OpenLM, load_oracle, load_state_dict  # noqa: E402


def reference_model(src, sd, cfg):
    xf = types.ModuleType("xformers"); xf.ops = types.ModuleType("xformers.ops")
    sys.modules["xformers"], sys.modules["xformers.ops"] = xf, xf.ops  # only used by code paths we do not take
    sys.path.insert(0, str(src))
    from open_lm.attention import torch_attn
    from open_lm.model import Params, Transformer
    from open_lm.norms import get_norm_class
    from open_lm.positional_embedding.rotary import RotaryEmbedding
    p = Params(dim=cfg["dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], vocab_size=cfg["vocab_size"],
               seq_len=2048, norm_type=get_norm_class("gain_only_lp_layer_norm"), attn_func=torch_attn,
               apply_qk_norm=True, positional_embedding_type="rotary", ffn_type="swiglu_torch")
    with torch.device("meta"):
        m = Transformer(p)
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    for mod in m.modules():
        if isinstance(mod, RotaryEmbedding):
            mod.reset_parameters()  # rebuild inv_freq / tables on CPU exactly as open_lm does
    print("reference load: missing", [k for k in missing if "inv_freq" not in k], "unexpected", unexpected)
    return m.eval()


@torch.no_grad()
def nll(model, ids):
    lg = model(ids.unsqueeze(0))
    lg = lg[0] if isinstance(lg, tuple) else lg
    return F.cross_entropy(lg[0, :-1].float(), ids[1:], reduction="none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--open_lm_src", default=None, help="clone of mlfoundations/open_lm; check 1 is skipped without it")
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ckpt_dir = Path(args.ckpt_dir)
    params = dict(l.split(": ", 1) for l in (ckpt_dir / "params.txt").read_text().splitlines() if ": " in l)
    cfg = CONFIGS[params["model"]]
    (ckpt,) = sorted((ckpt_dir / "checkpoints").glob("epoch_*.pt"))
    sd, meta = load_state_dict(ckpt)
    print("checkpoint keys besides state_dict:", {k: (v if isinstance(v, (int, str)) else type(v).__name__)
                                                  for k, v in meta.items()})
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    inv = [k for k in sd if k.endswith("inv_freq")]
    print("tensor dtypes:", dtypes, "| n tensors", len(sd), "| inv_freq keys", len(inv),
          (sd[inv[0]].dtype, sd[inv[0]][:4].tolist()) if inv else "")

    xs = json.load(open(DATA / "raw" / "xsum_gpt-4.raw_data.json", encoding="utf-8"))
    wr = json.load(open(DATA / "raw" / "writing_gpt-4.raw_data.json", encoding="utf-8"))
    short = [tok(t, return_tensors="pt").input_ids[0] for t in xs["original"][:4]]
    long_text = "\n\n".join(wr["original"][:12])
    long_ids = tok(long_text, return_tensors="pt").input_ids[0][:2048]
    print("long sequence tokens:", len(long_ids))

    sd_ours = {k: v for k, v in sd.items() if not k.endswith("inv_freq")}
    with torch.device("meta"):
        ours = OpenLM(**cfg, rope="fp32")
    ours.load_state_dict(sd_ours, strict=True, assign=True)
    ours.eval()

    # 1) architecture vs open_lm (fp32, CPU)
    if args.open_lm_src:
        ref = reference_model(args.open_lm_src, sd, cfg)
        with torch.no_grad():
            for ids in short[:2] + [long_ids[:600]]:
                a = ours(ids.unsqueeze(0))[0]
                b = ref(ids.unsqueeze(0))[0][0]
                print(f"[1] T={len(ids)} max|logit diff| ours vs open_lm = {(a - b).abs().max().item():.2e}, "
                      f"argmax agree {(a.argmax(-1) == b.argmax(-1)).float().mean().item():.4f}")
        del ref
    else:
        print("[1] skipped (no --open_lm_src)")
    with torch.no_grad():  # sanity without the reference: per-token NLL on human news text
        print("[1b] mean NLL of 4 human XSum texts (fp32 rope):",
              round(float(np.mean([nll(ours, ids).mean().item() for ids in short])), 3))

    # 2) rope table precision
    modes = ("fp32", "inv_bf16", "bf16")
    res = {}
    for mode in modes:
        ours.rope, ours._tables = mode, None
        res[mode] = nll(ours, long_ids)
    edges = [0, 128, 256, 512, 1024, 2048]
    print("[2] mean NLL by position, rope " + " / ".join(modes) + ":")
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo >= len(long_ids) - 1:
            break
        print(f"    pos {lo:4d}-{hi:4d}: " + "  ".join(f"{m} {res[m][lo:hi].mean().item():.4f}" for m in modes))
    best = min(modes, key=lambda m: res[m][256:].mean().item())
    print("    -> lowest loss at pos >= 256:", best)

    # 3) fp16 GPU vs fp32 CPU, with the chosen rope
    ours.rope, ours._tables = best, None
    gpu, _ = load_oracle(ckpt_dir, dtype=torch.float16, device="cuda", rope=best)
    d_fdg, d_lik = [], []
    for t in xs["original"][:5] + xs["sampled"][:5]:
        ids = tok(t, return_tensors="pt").input_ids
        with torch.no_grad():
            s32 = scores_from_logits(ours(ids), ids)
            s16 = scores_from_logits(gpu(ids.cuda()), ids.cuda())
        d_fdg.append(abs(s32["fast_detect_gpt"] - s16["fast_detect_gpt"]))
        d_lik.append(abs(s32["likelihood"] - s16["likelihood"]))
    print(f"[3] fp16-GPU vs fp32-CPU over 10 texts: max |dFDG| {max(d_fdg):.4f} (mean {np.mean(d_fdg):.4f}), "
          f"max |dLik| {max(d_lik):.5f}")


if __name__ == "__main__":
    main()
