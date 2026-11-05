#!/usr/bin/env bash
# Download the upstream exp_gpt3to4 data and paper result files used by run.py.
# Source: baoguangsheng/fast-detect-gpt @ 971b052 (MIT license).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
B=https://raw.githubusercontent.com/baoguangsheng/fast-detect-gpt/971b05202bac2bb504d60c0ac0812fea7a8f7c82/exp_gpt3to4
D="$ROOT/data/fastdetectgpt"
mkdir -p "$D/raw" "$D/paper_results"
for ds in xsum writing pubmed; do
  for m in davinci gpt-3.5-turbo gpt-4; do
    n="${ds}_${m}"
    curl -sf -o "$D/raw/$n.raw_data.json" "$B/data/$n.raw_data.json"
    curl -sf -o "$D/raw/$n.args.json" "$B/data/$n.args.json"
    curl -sf -o "$D/paper_results/$n.gpt-j-6B_gpt-neo-2.7B.sampling_discrepancy_analytic.json" \
      "$B/results/$n.gpt-j-6B_gpt-neo-2.7B.sampling_discrepancy_analytic.json"
    for c in likelihood logrank entropy; do
      curl -sf -o "$D/paper_results/$n.gpt-neo-2.7B.$c.json" "$B/results/$n.gpt-neo-2.7B.$c.json"
    done
  done
done
echo "done: $(ls "$D/raw" | wc -l) raw files, $(ls "$D/paper_results" | wc -l) result files in $D"
