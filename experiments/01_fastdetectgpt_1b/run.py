"""Fast-DetectGPT (analytic) + Likelihood / LogRank / Entropy baselines.

Criterion code follows the upstream implementation
(baoguangsheng/fast-detect-gpt @ 971b052, scripts/fast_detect_gpt.py and scripts/baselines.py)
with sampling model == scoring model. All four criteria are computed from one forward pass.

Usage:
    python experiments/01_fastdetectgpt_1b/run.py --model meta-llama/Llama-3.2-1B
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "fastdetectgpt"
DATASETS = ["xsum", "writing", "pubmed"]
SOURCES = ["davinci", "gpt-3.5-turbo", "gpt-4"]
CRITERIA = ["fast_detect_gpt", "likelihood", "logrank", "entropy"]
# upstream result file suffix for each criterion
PAPER_FILES = {
    "fast_detect_gpt": "gpt-j-6B_gpt-neo-2.7B.sampling_discrepancy_analytic",
    "likelihood": "gpt-neo-2.7B.likelihood",
    "logrank": "gpt-neo-2.7B.logrank",
    "entropy": "gpt-neo-2.7B.entropy",
}


@torch.no_grad()
def criteria_for_text(model, tokenizer, text, device, save_tokens=False):
    tok = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
    labels = tok.input_ids[:, 1:]
    logits = model(**tok).logits[:, :-1].float()  # fp32 for softmax stability

    lprobs = F.log_softmax(logits, dim=-1)
    probs = lprobs.exp()
    ll_tok = lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T]

    # Fast-DetectGPT analytic: sampling model == scoring model
    mean_ref = (probs * lprobs).sum(-1)
    var_ref = (probs * lprobs.square()).sum(-1) - mean_ref.square()
    fdg = ((ll_tok.sum(-1) - mean_ref.sum(-1)) / var_ref.sum(-1).sqrt()).mean().item()

    # baselines (upstream baselines.py)
    likelihood = ll_tok.mean().item()
    ranks = (logits > logits.gather(-1, labels.unsqueeze(-1))).sum(-1).float() + 1  # 1-indexed rank
    logrank = -ranks.log().mean().item()
    entropy = (-(probs * lprobs).sum(-1)).mean().item()

    out = {"fast_detect_gpt": fdg, "likelihood": likelihood, "logrank": logrank,
           "entropy": entropy, "n_tokens": int(labels.size(1))}
    if save_tokens:  # per-token self-information L, entropy E, and variance (for L-E coupling analysis)
        r4 = lambda t: [round(v, 4) for v in t[0].tolist()]
        out.update(tok_L=r4(-ll_tok), tok_E=r4(-mean_ref), tok_var=r4(var_ref))
    return out


class SplitLM(torch.nn.Module):
    """4-bit decoder on GPU, input embedding on CPU, fp16 lm_head on GPU (fits an 8B model in ~5.3 GB).

    Only the decoder blocks are quantized; the output layer that produces the probabilities stays fp16.
    """

    def __init__(self, name):
        super().__init__()
        from accelerate.hooks import remove_hook_from_module
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               llm_int8_enable_fp32_cpu_offload=True)
        dm = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": "cpu"}
        lm = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map=dm, dtype=torch.float16)
        def real_weight(mod):
            w = mod.weight
            if w.device.type == "meta":  # offloaded by accelerate: real tensor lives in the hook's weights_map
                w = mod._hf_hook.weights_map["weight"]
            return w.detach()

        w_emb = real_weight(lm.model.embed_tokens)
        w_head = w_emb if lm.config.tie_word_embeddings else real_weight(lm.lm_head)
        for mod, w, dev, dt in ((lm.model.embed_tokens, w_emb, "cpu", torch.float32),
                                (lm.lm_head, w_head, "cuda", torch.float16)):
            remove_hook_from_module(mod)
            mod.weight = torch.nn.Parameter(w.to(dev, dt), requires_grad=False)
        self.lm = lm

    def forward(self, input_ids, attention_mask=None):
        emb = self.lm.model.embed_tokens(input_ids.cpu()).to("cuda", torch.float16)
        h = self.lm.model(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm.lm_head(h.to(torch.float16)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--max_gpu_mem", default=None, help="e.g. 4GiB; offloads the rest to CPU")
    ap.add_argument("--limit", type=int, default=None, help="only first N pairs per dataset")
    ap.add_argument("--datasets", nargs="*", default=None, help="e.g. xsum_gpt-4 writing_davinci")
    ap.add_argument("--save_tokens", action="store_true", help="store per-token L/E/var; writes to <model>/tokens/")
    ap.add_argument("--load_4bit", action="store_true", help="bitsandbytes NF4 weights (fp16 compute); tag gets -nf4")
    ap.add_argument("--split_4bit", action="store_true",
                    help="NF4 decoder on GPU + CPU embedding + fp16 lm_head on GPU (for 8B on 6 GB); tag gets -nf4split")
    args = ap.parse_args()

    torch.manual_seed(0)
    tag = args.model.split("/")[-1] + ("-nf4" if args.load_4bit else "") + ("-nf4split" if args.split_4bit else "")
    out_dir = ROOT / "outputs" / "01_fastdetectgpt_1b" / tag
    if args.save_tokens:
        out_dir = out_dir / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kw = {"dtype": getattr(torch, args.dtype)}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=bool(args.max_gpu_mem))
        kw["device_map"] = {"": 0}
    if args.max_gpu_mem:
        kw.update(device_map="auto", max_memory={0: args.max_gpu_mem, "cpu": "12GiB"})
    t0 = time.time()
    if args.split_4bit:
        model = SplitLM(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.max_gpu_mem and not args.load_4bit and not args.split_4bit:
        model.to("cuda")
    model.eval()
    device = "cuda"
    print(f"loaded {args.model} ({args.dtype}) in {time.time() - t0:.1f}s, "
          f"GPU mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    names = args.datasets or [f"{d}_{s}" for d in DATASETS for s in SOURCES]
    summary = {}
    for name in names:
        raw = json.load(open(DATA / "raw" / f"{name}.raw_data.json", encoding="utf-8"))
        n = len(raw["original"]) if args.limit is None else min(args.limit, len(raw["original"]))
        rows = []
        for i in tqdm(range(n), desc=name, leave=False):
            rows.append({"idx": i,
                         "original": criteria_for_text(model, tokenizer, raw["original"][i], device, args.save_tokens),
                         "sampled": criteria_for_text(model, tokenizer, raw["sampled"][i], device, args.save_tokens)})
        res = {"model": args.model, "dtype": args.dtype, "dataset": name, "n": n, "metrics": {}}
        for c in CRITERIA:
            real = [r["original"][c] for r in rows]
            fake = [r["sampled"][c] for r in rows]
            res["metrics"][c] = {"roc_auc": float(roc_auc_score([0] * n + [1] * n, real + fake)),
                                 "real_mean": float(np.mean(real)), "real_std": float(np.std(real)),
                                 "fake_mean": float(np.mean(fake)), "fake_std": float(np.std(fake))}
        res["rows"] = rows
        suffix = "" if args.limit is None else f".limit{n}"
        json.dump(res, open(out_dir / f"{name}{suffix}.json", "w", encoding="utf-8"))
        summary[name] = {c: res["metrics"][c]["roc_auc"] for c in CRITERIA}
        print(name, " ".join(f"{c}={summary[name][c]:.4f}" for c in CRITERIA), flush=True)

    # comparison with upstream (paper) result files
    print(f"\nAUROC  ours ({tag}) vs paper (Fast-DetectGPT: GPT-J/Neo-2.7, baselines: Neo-2.7)")
    print(f"{'dataset':22s}" + "".join(f"{c:>26s}" for c in CRITERIA))
    for name in names:
        cells = []
        for c in CRITERIA:
            p = json.load(open(DATA / "paper_results" / f"{name}.{PAPER_FILES[c]}.json",
                               encoding="utf-8"))["metrics"]["roc_auc"]
            cells.append(f"{summary[name][c]:.4f} / {p:.4f}")
        print(f"{name:22s}" + "".join(f"{x:>26s}" for x in cells))
    json.dump(summary, open(out_dir / f"summary{'' if args.limit is None else '.limit'}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
