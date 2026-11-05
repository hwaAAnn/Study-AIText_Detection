"""Token-level analysis of per-token self-information L and entropy E (needs run.py --save_tokens).

L-E coupling beta follows "When Less is More" (Han et al., EMNLP 2026, Sec. 6.2): per text, sort tokens by
entropy into 10 buckets, take bucket means of L and E, center by the text mean, pool over texts of one
class and fit the slope of dL on dE. beta ~ 1: L tracks E (well calibrated); beta < 1: decoupled.

Also reports, per class: mean E, share of near-deterministic positions (E < 0.5 nats), and mean (L - E)
in the lowest / highest 20% entropy tokens. Every statistic is computed twice: on all tokens and on the
continuation only (the human-written shared prefix of each pair removed).

    python experiments/01_fastdetectgpt_1b/analyze_tokens.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "01_fastdetectgpt_1b"
MODELS = ["gpt-neo-1.3B", "gpt-neo-2.7B", "OLMo-2-0425-1B", "SmolLM2-1.7B", "gemma-3-1b-pt", "Llama-3.2-1B",
          "Llama-3.2-3B", "Llama-3.2-1B-nf4split", "Llama-3.1-8B-nf4split"]
DATASETS = ["xsum_davinci", "xsum_gpt-4"]


def shared_prefix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def beta(texts):
    dE, dL = [], []
    for L, E in texts:
        if len(L) < 20:
            continue
        order = np.argsort(-E)  # high -> low entropy
        for idx in np.array_split(order, 10):
            dE.append(E[idx].mean() - E.mean())
            dL.append(L[idx].mean() - L.mean())
    return float(np.polyfit(dE, dL, 1)[0])


def stats(texts):
    allL = np.concatenate([L for L, _ in texts]); allE = np.concatenate([E for _, E in texts])
    q20, q80 = np.quantile(allE, [0.2, 0.8])
    gap = allL - allE
    return {"beta": beta(texts), "mean_E": float(allE.mean()), "mean_L": float(allL.mean()),
            "share_E_lt_0.5": float((allE < 0.5).mean()),
            "gap_low20": float(gap[allE <= q20].mean()), "gap_high20": float(gap[allE >= q80].mean())}


def main():
    report = {}
    for m in MODELS:
        for ds in DATASETS:
            f = OUT / m / "tokens" / f"{ds}.json"
            if not f.exists():
                continue
            rows = json.load(open(f, encoding="utf-8"))["rows"]
            res = {}
            for scope in ("all", "continuation"):
                cls = {"human": [], "ai": []}
                for r in rows:
                    h, a = r["original"], r["sampled"]
                    k = shared_prefix(h["tok_L"], a["tok_L"]) if scope == "continuation" else 0
                    for name, x in (("human", h), ("ai", a)):
                        L, E = np.array(x["tok_L"][k:]), np.array(x["tok_E"][k:])
                        if len(L):
                            cls[name].append((L, E))
                res[scope] = {c: stats(t) for c, t in cls.items()}
            report[f"{m}/{ds}"] = res
    json.dump(report, open(OUT / "token_analysis.json", "w", encoding="utf-8"), indent=1)

    for scope in ("all", "continuation"):
        print(f"\n== scope: {scope}")
        print("model/dataset | beta H | beta AI | E H | E AI | share E<0.5 H | AI | (L-E) low20 H | AI | (L-E) high20 H | AI")
        for key, res in report.items():
            h, a = res[scope]["human"], res[scope]["ai"]
            print(f"{key} | {h['beta']:.3f} | {a['beta']:.3f} | {h['mean_E']:.3f} | {a['mean_E']:.3f} | "
                  f"{h['share_E_lt_0.5']:.1%} | {a['share_E_lt_0.5']:.1%} | {h['gap_low20']:+.3f} | {a['gap_low20']:+.3f} | "
                  f"{h['gap_high20']:+.3f} | {a['gap_high20']:+.3f}")


if __name__ == "__main__":
    main()
