"""Compare scoring models on the exp_gpt3to4 data: AUROCs, entropy cancellation, normalization effect.

Reads outputs/01_fastdetectgpt_1b/<model>/<dataset>.json written by run.py.

    python experiments/01_fastdetectgpt_1b/analyze_scorers.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "01_fastdetectgpt_1b"
DATASETS = ["xsum", "writing", "pubmed"]
SOURCES = ["davinci", "gpt-3.5-turbo", "gpt-4"]
# (output dir, training-data era, distillation reported by the model's own report)
MODELS = [
    ("gpt-neo-1.3B", "2020 (Pile)", "no"),
    ("gpt-neo-2.7B", "2020 (Pile)", "no"),
    ("OLMo-2-0425-1B", "mostly pre-2023 web (DCLM)", "not reported"),
    ("SmolLM2-1.7B", "web to 2024 + synthetic (Cosmopedia)", "not reported"),
    ("gemma-3-1b-pt", "2024-25", "yes"),
    ("Llama-3.2-1B", "2023-24", "yes (Llama-3.1 logits)"),
    ("Llama-3.2-3B", "2023-24", "yes (Llama-3.1 logits)"),
    ("Llama-3.2-1B-nf4split", "2023-24", "yes; 4-bit decoder (quantization control)"),
    ("Llama-3.1-8B-nf4split", "2023-24", "no (teacher of 3.2); 4-bit decoder"),
]


def per_dataset(rows):
    g = lambda kind, c: np.array([r[kind][c] for r in rows], dtype=float)
    Lh, La = g("original", "likelihood"), g("sampled", "likelihood")
    Hh, Ha = g("original", "entropy"), g("sampled", "entropy")
    Fh, Fa = g("original", "fast_detect_gpt"), g("sampled", "fast_detect_gpt")
    y = [0] * len(rows) + [1] * len(rows)
    auc = lambda h, a: float(roc_auc_score(y, np.concatenate([h, a])))
    dL, dH = La.mean() - Lh.mean(), Ha.mean() - Hh.mean()
    vals = np.concatenate([Lh, La, Hh, Ha, Fh, Fa])
    return {
        "auc_fdg": auc(Fh, Fa),
        "auc_lik": auc(Lh, La),
        "auc_egs": auc(Lh + Hh, La + Ha),  # unnormalized per-token numerator (= -EGS)
        "dL": float(dL), "dH": float(dH), "keep": float((dL + dH) / dL) if dL > 0 else float("nan"),
        "fdg_h": float(Fh.mean()), "fdg_a": float(Fa.mean()),
        "nan": int((~np.isfinite(vals)).sum()),
    }


def main():
    report = {}
    for tag, era, distill in MODELS:
        d = OUT / tag
        names = [f"{a}_{b}" for a in DATASETS for b in SOURCES]
        if not all((d / f"{n}.json").exists() for n in names):
            print(f"skip {tag}: incomplete")
            continue
        per = {n: per_dataset(json.load(open(d / f"{n}.json", encoding="utf-8"))["rows"]) for n in names}
        chat = [n for n in names if "davinci" not in n]
        dav = [n for n in names if "davinci" in n]
        m = lambda key, ns: float(np.mean([per[n][key] for n in ns]))
        report[tag] = {
            "era": era, "distill": distill, "per": per,
            "fdg_all": m("auc_fdg", names), "lik_all": m("auc_lik", names),
            "fdg_chat": m("auc_fdg", chat), "lik_chat": m("auc_lik", chat), "egs_chat": m("auc_egs", chat),
            "fdg_dav": m("auc_fdg", dav), "lik_dav": m("auc_lik", dav),
            "keep_chat": m("keep", chat), "dH_chat": m("dH", chat), "dL_chat": m("dL", chat),
            "fdg_lt_lik_chat": sum(per[n]["auc_fdg"] < per[n]["auc_lik"] for n in chat),
            "xsum_gpt4_fdg": per["xsum_gpt-4"]["auc_fdg"], "xsum_gpt4_keep": per["xsum_gpt-4"]["keep"],
            "human_fdg_xsum": m("fdg_h", [n for n in names if n.startswith("xsum")]),
            "nan": sum(per[n]["nan"] for n in names),
        }
    json.dump(report, open(OUT / "scorer_comparison.json", "w", encoding="utf-8"), indent=1)

    hdr = ("model", "era", "distill", "FDG all", "Lik all", "FDG chat", "Lik chat", "EGS chat",
           "FDG<Lik (chat/6)", "keep chat", "dH chat", "xsum4 FDG", "xsum4 keep", "human FDG xsum", "NaN")
    print(" | ".join(hdr))
    for tag, r in report.items():
        print(" | ".join([tag, r["era"], r["distill"]] + [f"{r[k]:.3f}" for k in
              ("fdg_all", "lik_all", "fdg_chat", "lik_chat", "egs_chat")] +
              [str(r["fdg_lt_lik_chat"]), f"{r['keep_chat']:.0%}", f"{r['dH_chat']:+.3f}",
               f"{r['xsum_gpt4_fdg']:.3f}", f"{r['xsum_gpt4_keep']:.0%}", f"{r['human_fdg_xsum']:+.2f}", str(r["nan"])]))


if __name__ == "__main__":
    main()
