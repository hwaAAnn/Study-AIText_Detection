"""Which tokens became more predictable between two Oracles? Per-token change of entropy E and surprisal L.

For every continuation token (shared human prefix removed) of the six GPT-3.5/4 sets, compare the two
cutoffs on the same token, separately for AI and human text, and list the token types with the largest
total entropy drop.

    python experiments/02_tic_lm_data_era/analyze_token_shift.py --a 202301 --b 202407
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "02_tic_lm_data_era"
RAW = ROOT / "data" / "fastdetectgpt" / "raw"
CHAT = [f"{d}_{s}" for d in ("xsum", "writing", "pubmed") for s in ("gpt-3.5-turbo", "gpt-4")]


def collect(tok, a, b, kind):
    per_tok, dE_all, dL_all = defaultdict(list), [], []
    for name in CHAT:
        raw = json.load(open(RAW / f"{name}.raw_data.json", encoding="utf-8"))
        ra = json.load(open(OUT / a / f"{name}.json", encoding="utf-8"))["rows"]
        rb = json.load(open(OUT / b / f"{name}.json", encoding="utf-8"))["rows"]
        for i, (x, y) in enumerate(zip(ra, rb)):
            ids = tok(raw[kind][i]).input_ids[1:2048]
            Ea, Eb = np.array(x[kind]["tok_E"]), np.array(y[kind]["tok_E"])
            La, Lb = np.array(x[kind]["tok_L"]), np.array(y[kind]["tok_L"])
            assert len(ids) == len(Ea) == len(Eb)
            h, s = x["original"]["tok_L"], x["sampled"]["tok_L"]
            k = 0
            while k < min(len(h), len(s)) and h[k] == s[k]:
                k += 1
            for j in range(k, len(ids)):
                per_tok[tok.decode([ids[j]])].append((Eb[j] - Ea[j], Lb[j] - La[j], Ea[j]))
            dE_all.append(Eb[k:] - Ea[k:]); dL_all.append(Lb[k:] - La[k:])
    return per_tok, np.concatenate(dE_all), np.concatenate(dL_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="oracles_1b_220b")
    ap.add_argument("--a", default="202301")
    ap.add_argument("--b", default="202407")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    a, b = f"{args.series}-{args.a}", f"{args.series}-{args.b}"
    report = {}
    for kind, label in (("sampled", "AI"), ("original", "human")):
        per_tok, dE, dL = collect(tok, a, b, kind)
        rows = sorted(((t, len(v), float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])),
                        float(np.sum([x[0] for x in v])), float(np.mean([x[2] for x in v])))
                       for t, v in per_tok.items() if len(v) >= 15), key=lambda r: r[4])
        top_share = sum(r[4] for r in rows[:args.top]) / float(dE.sum())
        report[label] = {"n_tokens": int(len(dE)), "mean_dE": float(dE.mean()), "mean_dL": float(dL.mean()),
                         "share_dE_lt_-0.5": float((dE < -0.5).mean()), "top_share_of_total_dE": top_share,
                         "top": [dict(zip(("token", "count", "mean_dE", "mean_dL", "sum_dE", "E_a"), r))
                                 for r in rows[:args.top]]}
        print(f"\n== {label}: {len(dE)} continuation tokens | mean dE {dE.mean():+.3f} | mean dL {dL.mean():+.3f} | "
              f"dE < -0.5: {(dE < -0.5).mean():.1%} | top {args.top} types carry {top_share:.0%} of the total dE")
        print("token | count | mean dE | mean dL | sum dE | E at " + args.a)
        for t, n, me, ml, se, e0 in rows[:args.top]:
            print(f"  {t!r:16s} {n:5d} {me:+.3f} {ml:+.3f} {se:+8.1f} {e0:.2f}")
    json.dump(report, open(OUT / f"token_shift_{args.a}_{args.b}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
