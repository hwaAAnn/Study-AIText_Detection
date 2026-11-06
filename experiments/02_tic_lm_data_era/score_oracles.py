"""Score the exp_gpt3to4 data with TiC-LM Oracle checkpoints (same criteria as experiment 01).

Each Oracle is an OpenLM model trained from scratch on Common Crawl up to one cutoff month, with the same
architecture, tokenizer, filtering and token budget (the 201305 model has half the budget). Per-token L/E are
always stored so the token analysis can run on every dataset.

    python experiments/02_tic_lm_data_era/score_oracles.py --months 202301 202407
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "01_fastdetectgpt_1b")]  # also works under python -I
from run import CRITERIA, DATA, DATASETS, SOURCES, scores_from_logits  # noqa: E402
from openlm import TOKENIZER, load_oracle  # noqa: E402

ROOT = HERE.parents[1]
CKPT = ROOT / "data" / "tic-lm"
OUT = ROOT / "outputs" / "02_tic_lm_data_era"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="oracles_1b_220b")
    ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--rope", default="fp32", choices=["fp32", "inv_bf16", "bf16"], help="see check_openlm.py")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    names = args.datasets or [f"{d}_{s}" for d in DATASETS for s in SOURCES]

    for month in args.months:
        t0 = time.time()
        model, info = load_oracle(CKPT / args.series / month, dtype=torch.float16, device="cuda", rope=args.rope)
        tag = f"{args.series}-{month}" + ("" if args.rope == "fp32" else f"-rope_{args.rope}")
        out_dir = OUT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"loaded {tag} ({info['ckpt']}) in {time.time() - t0:.0f}s, "
              f"GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
        summary = {}
        for name in names:
            raw = json.load(open(DATA / "raw" / f"{name}.raw_data.json", encoding="utf-8"))
            n = len(raw["original"]) if args.limit is None else min(args.limit, len(raw["original"]))
            rows = []
            for i in tqdm(range(n), desc=f"{month} {name}", leave=False):
                row = {"idx": i}
                for kind in ("original", "sampled"):
                    ids = tok(raw[kind][i], return_tensors="pt").input_ids[:, :2048].cuda()
                    with torch.no_grad():
                        row[kind] = scores_from_logits(model(ids), ids, save_tokens=True)
                rows.append(row)
            res = {"model": f"tic-lm/{args.series}/{month}", "dtype": "float16", "rope": args.rope,
                   "dataset": name, "n": n, "params": {k: info["params"][k] for k in
                   ("name", "epochs", "train_num_samples", "seq_len", "global_batch_size")}, "metrics": {}}
            for c in CRITERIA:
                real, fake = [r["original"][c] for r in rows], [r["sampled"][c] for r in rows]
                res["metrics"][c] = {"roc_auc": float(roc_auc_score([0] * n + [1] * n, real + fake)),
                                     "real_mean": float(np.mean(real)), "fake_mean": float(np.mean(fake))}
            res["rows"] = rows
            suffix = "" if args.limit is None else f".limit{n}"
            json.dump(res, open(out_dir / f"{name}{suffix}.json", "w", encoding="utf-8"))
            summary[name] = {c: res["metrics"][c]["roc_auc"] for c in CRITERIA}
            print(month, name, " ".join(f"{c}={summary[name][c]:.4f}" for c in CRITERIA), flush=True)
        if args.limit is None:
            json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
