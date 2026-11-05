"""Quantization control: Llama-3.2-1B fp16 vs the same model with a 4-bit (NF4) decoder (run.py --split_4bit).

Compares, on identical texts and token ids:
  1. per-token self-information L = -log p and entropy E (mean |diff|, correlation)   [needs --save_tokens runs]
  2. L-E coupling beta (from analyze_tokens.py, same definition)                       [needs --save_tokens runs]
  3. per-text Fast-DetectGPT score (correlation) and AUROC / entropy-cancellation per dataset   [full runs]

    python experiments/01_fastdetectgpt_1b/analyze_quant.py
"""
import json
from pathlib import Path

import numpy as np

from analyze_scorers import per_dataset
from analyze_tokens import beta

OUT = Path(__file__).resolve().parents[2] / "outputs" / "01_fastdetectgpt_1b"
REF, Q = "Llama-3.2-1B", "Llama-3.2-1B-nf4split"
NAMES = [f"{a}_{b}" for a in ("xsum", "writing", "pubmed") for b in ("davinci", "gpt-3.5-turbo", "gpt-4")]


def load(model, name, tokens=False):
    f = OUT / model / ("tokens" if tokens else "") / f"{name}.json"
    return json.load(open(f, encoding="utf-8"))["rows"] if f.exists() else None


def main():
    print("== per dataset (full runs): AUROC FDG / Likelihood, keep ratio, per-text FDG correlation")
    print("dataset | FDG fp16 | FDG nf4 | Lik fp16 | Lik nf4 | keep fp16 | keep nf4 | corr(FDG per text)")
    for n in NAMES:
        a, b = load(REF, n), load(Q, n)
        if a is None or b is None:
            print(n, "missing"); continue
        pa, pb = per_dataset(a), per_dataset(b)
        fa = [r[k]["fast_detect_gpt"] for r in a for k in ("original", "sampled")]
        fb = [r[k]["fast_detect_gpt"] for r in b for k in ("original", "sampled")]
        print(f"{n} | {pa['auc_fdg']:.3f} | {pb['auc_fdg']:.3f} | {pa['auc_lik']:.3f} | {pb['auc_lik']:.3f} | "
              f"{pa['keep']:.0%} | {pb['keep']:.0%} | {np.corrcoef(fa, fb)[0, 1]:.4f}")

    print("\n== per token (save_tokens runs): mean |dL|, mean |dE|, corr L, corr E, beta human/AI fp16 vs nf4")
    for n in ("xsum_davinci", "xsum_gpt-4"):
        a, b = load(REF, n, True), load(Q, n, True)
        if a is None or b is None:
            print(n, "missing"); continue
        L1, L2, E1, E2 = [], [], [], []
        texts = {(m, c): [] for m in ("fp16", "nf4") for c in ("human", "ai")}
        for ra, rb in zip(a, b):
            for k, c in (("original", "human"), ("sampled", "ai")):
                x, y = ra[k], rb[k]
                assert len(x["tok_L"]) == len(y["tok_L"]), "token ids differ between runs"
                L1 += x["tok_L"]; L2 += y["tok_L"]; E1 += x["tok_E"]; E2 += y["tok_E"]
                texts[("fp16", c)].append((np.array(x["tok_L"]), np.array(x["tok_E"])))
                texts[("nf4", c)].append((np.array(y["tok_L"]), np.array(y["tok_E"])))
        L1, L2, E1, E2 = map(np.array, (L1, L2, E1, E2))
        print(f"{n}: tokens={len(L1)} | mean|dL|={np.abs(L1 - L2).mean():.4f} (mean L {L1.mean():.3f}) | "
              f"mean|dE|={np.abs(E1 - E2).mean():.4f} (mean E {E1.mean():.3f}) | corr L={np.corrcoef(L1, L2)[0, 1]:.4f} "
              f"| corr E={np.corrcoef(E1, E2)[0, 1]:.4f}")
        print(f"   beta human fp16 {beta(texts[('fp16', 'human')]):.3f} vs nf4 {beta(texts[('nf4', 'human')]):.3f} | "
              f"beta AI fp16 {beta(texts[('fp16', 'ai')]):.3f} vs nf4 {beta(texts[('nf4', 'ai')]):.3f}")


if __name__ == "__main__":
    main()
