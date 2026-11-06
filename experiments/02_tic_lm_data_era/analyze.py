"""Compare TiC-LM Oracles across training-data cutoff months, with the experiment 01 scorers as reference.

Per month: AUROC of Fast-DetectGPT and Likelihood, keep ratio (dL+dH)/dL, token-level L-E statistics, and a
paired bootstrap (resampling text pairs) for the change between consecutive cutoffs.

    python experiments/02_tic_lm_data_era/analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "01_fastdetectgpt_1b")]  # also works under python -I
from analyze_scorers import per_dataset  # noqa: E402
from analyze_tokens import shared_prefix, stats  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "02_tic_lm_data_era"
EXP01 = ROOT / "outputs" / "01_fastdetectgpt_1b" / "scorer_comparison.json"
SERIES = "oracles_1b_220b"
MONTHS = ["201305", "201501", "201701", "201901", "202101", "202301", "202407"]
NAMES = [f"{d}_{s}" for d in ("xsum", "writing", "pubmed") for s in ("davinci", "gpt-3.5-turbo", "gpt-4")]
CHAT = [n for n in NAMES if "davinci" not in n]
B = 2000


def load(month):
    d = OUT / f"{SERIES}-{month}"
    if not all((d / f"{n}.json").exists() for n in NAMES):
        return None
    return {n: json.load(open(d / f"{n}.json", encoding="utf-8"))["rows"] for n in NAMES}


def lh_arrays(rows_by_name):
    """per chat set: [4, n] array of human L, AI L, human H, AI H (likelihood = mean log p, entropy)."""
    return {n: np.array([[r[k][c] for r in rows_by_name[n]] for c in ("likelihood", "entropy")
                         for k in ("original", "sampled")]) for n in CHAT}


def keep_chat(arr, idx=None):
    """keep ratio averaged over the 6 chat sets; idx: per-set resampled pair indices."""
    ks = []
    for n in CHAT:
        a = arr[n] if idx is None else arr[n][:, idx[n]]
        dL, dH = a[1].mean() - a[0].mean(), a[3].mean() - a[2].mean()
        ks.append((dL + dH) / dL)
    return float(np.mean(ks))


def token_stats(rows):
    cls = {"human": [], "ai": []}
    for r in rows:
        h, a = r["original"], r["sampled"]
        k = shared_prefix(h["tok_L"], a["tok_L"])
        for name, x in (("human", h), ("ai", a)):
            L, E = np.array(x["tok_L"][k:]), np.array(x["tok_E"][k:])
            if len(L):
                cls[name].append((L, E))
    return {c: stats(t) for c, t in cls.items()}


def main():
    data = {m: load(m) for m in MONTHS}
    data = {m: v for m, v in data.items() if v is not None}
    arrs = {m: lh_arrays(v) for m, v in data.items()}
    report = {}
    for m, rows in data.items():
        per = {n: per_dataset(rows[n]) for n in NAMES}
        tk = {n: token_stats(rows[n]) for n in ("xsum_davinci", "xsum_gpt-4")}
        mean = lambda key, ns: float(np.mean([per[n][key] for n in ns]))
        report[m] = {"per": per, "tokens": tk,
                     "fdg_chat": mean("auc_fdg", CHAT), "lik_chat": mean("auc_lik", CHAT),
                     "fdg_dav": mean("auc_fdg", [n for n in NAMES if "davinci" in n]),
                     "lik_dav": mean("auc_lik", [n for n in NAMES if "davinci" in n]),
                     "keep_chat": keep_chat(arrs[m]), "dH_chat": mean("dH", CHAT), "dL_chat": mean("dL", CHAT),
                     "fdg_lt_lik_chat": sum(per[n]["auc_fdg"] < per[n]["auc_lik"] for n in CHAT),
                     "xsum4_fdg": per["xsum_gpt-4"]["auc_fdg"], "xsum4_keep": per["xsum_gpt-4"]["keep"],
                     "nan": sum(per[n]["nan"] for n in NAMES)}

    # paired bootstrap of keep_chat differences between consecutive available cutoffs (same text pairs)
    rng = np.random.default_rng(0)
    ms = list(data)
    boots = {}
    for a, b in zip(ms[:-1], ms[1:]):
        diffs = []
        for _ in range(B):
            idx = {n: rng.integers(0, len(data[a][n]), len(data[a][n])) for n in CHAT}
            diffs.append(keep_chat(arrs[b], idx) - keep_chat(arrs[a], idx))
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        boots[f"{a}->{b}"] = {"diff": report[b]["keep_chat"] - report[a]["keep_chat"], "ci95": [float(lo), float(hi)]}
    json.dump({"months": report, "keep_bootstrap": boots}, open(OUT / "oracle_comparison.json", "w"), indent=1)

    print("cutoff | FDG chat | Lik chat | FDG<Lik (/6) | keep chat | dH chat | xsum4 FDG | xsum4 keep | "
          "FDG dav | Lik dav | NaN")
    for m, r in report.items():
        print(f"{m} | {r['fdg_chat']:.3f} | {r['lik_chat']:.3f} | {r['fdg_lt_lik_chat']} | {r['keep_chat']:.0%} | "
              f"{r['dH_chat']:+.3f} | {r['xsum4_fdg']:.3f} | {r['xsum4_keep']:.0%} | {r['fdg_dav']:.3f} | "
              f"{r['lik_dav']:.3f} | {r['nan']}")
    if EXP01.exists():
        ref = json.load(open(EXP01, encoding="utf-8"))
        print("-- experiment 01 reference")
        for tag in ("gpt-neo-1.3B", "OLMo-2-0425-1B", "SmolLM2-1.7B", "Llama-3.2-1B"):
            if tag in ref:
                r = ref[tag]
                print(f"{tag} | {r['fdg_chat']:.3f} | {r['lik_chat']:.3f} | {r['fdg_lt_lik_chat']} | "
                      f"{r['keep_chat']:.0%} | {r['dH_chat']:+.3f} | {r['xsum_gpt4_fdg']:.3f} | "
                      f"{r['xsum_gpt4_keep']:.0%}")
    print("\nkeep chat change between cutoffs (paired bootstrap over text pairs, 95% CI):")
    for k, v in boots.items():
        print(f"  {k}: {v['diff']:+.1%}  [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    print("\ntoken stats, continuation only (beta H/AI | (L-E) in lowest-20% entropy tokens H/AI):")
    for m, r in report.items():
        for ds, t in r["tokens"].items():
            h, a = t["human"], t["ai"]
            print(f"  {m} {ds}: beta {h['beta']:.3f}/{a['beta']:.3f} | low20 {h['gap_low20']:+.3f}/{a['gap_low20']:+.3f}"
                  f" | E {h['mean_E']:.3f}/{a['mean_E']:.3f}")


if __name__ == "__main__":
    main()
