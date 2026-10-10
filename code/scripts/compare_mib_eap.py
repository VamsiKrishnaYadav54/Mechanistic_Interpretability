"""
Cross-check: do MIB's EAP and our AutoCircuit EAP rank the same edges?

Both score the same 32,491 structural edges of GPT-2. They differ in:
  - positions: MIB scores each edge once (all positions together);
    AutoCircuit scores each edge at each of the 15 positions.
  - data: MIB's IOI dataset vs AutoCircuit's vanilla IOI template.
  - TransformerLens version (2.1x vs 2.4).
So expect HIGH but not perfect agreement.

We compare MIB against two ways of collapsing our 15 positions:
  max_abs  : max |score| over positions   (what scores_structural.json uses)
  sum      : signed sum over positions    (closest to how MIB scores an edge)

Runs on the login node, no GPU, no model:
    python code/scripts/compare_mib_eap.py \
        --mib results/mib/circuits/EAP_patching_edge/ioi_gpt2/importances.json \
        --ours results/ioi_eap_gpt2/scores_instance.csv.gz
"""

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict

import numpy as np

EDGE_RE = re.compile(r"^(?P<src>.+?)->(?P<dst>.+?)(?:<(?P<qkv>[qkv])>)?$")


def mib_node_to_ours(name: str) -> str:
    """MIB/EAP-IG node names -> AutoCircuit node names."""
    if name == "input":
        return "Resid Start"
    if name == "logits":
        return "Resid End"
    m = re.fullmatch(r"a(\d+)\.h(\d+)", name)
    if m:
        return f"A{m.group(1)}.{m.group(2)}"
    m = re.fullmatch(r"m(\d+)", name)
    if m:
        return f"MLP {m.group(1)}"
    raise ValueError(f"unknown MIB node name: {name}")


def load_mib(path):
    with open(path) as f:
        d = json.load(f)
    edges = d["edges"]
    out = {}
    for key, val in edges.items():
        m = EDGE_RE.match(key)
        if not m:
            raise ValueError(f"cannot parse MIB edge name: {key}")
        src = mib_node_to_ours(m.group("src"))
        dst = mib_node_to_ours(m.group("dst"))
        if m.group("qkv"):
            dst = f"{dst}.{m.group('qkv').upper()}"
        score = val["score"] if isinstance(val, dict) else float(val)
        out[(src, dst)] = float(score)
    return out


def load_ours(path):
    max_abs = {}
    signed_sum = defaultdict(float)
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            k = (row["src"], row["dest"])
            s = row["score"]
            v = float("inf") if s == "inf" else float("-inf") if s == "-inf" else float(s)
            if not np.isfinite(v):
                raise ValueError("inf scores found -- this check is for EAP, not ACDC")
            signed_sum[k] += v
            if abs(v) > max_abs.get(k, -1.0):
                max_abs[k] = abs(v)
    return max_abs, dict(signed_sum)


def rank(a):
    """Average ranks (ties share a rank)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def topk_overlap(keys, x, y, k):
    ix = set(np.argsort(-x)[:k])
    iy = set(np.argsort(-y)[:k])
    return len(ix & iy) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    mib = load_mib(args.mib)
    max_abs, signed_sum = load_ours(args.ours)

    common = sorted(set(mib) & set(max_abs))
    only_mib = set(mib) - set(max_abs)
    only_ours = set(max_abs) - set(mib)
    print(f"MIB edges      : {len(mib)}")
    print(f"our edges      : {len(max_abs)}")
    print(f"matched        : {len(common)}")
    print(f"only in MIB    : {len(only_mib)}  e.g. {sorted(only_mib)[:3]}")
    print(f"only in ours   : {len(only_ours)}  e.g. {sorted(only_ours)[:3]}")
    if not common:
        raise SystemExit("no edges matched -- naming mismatch, paste this output")

    m_signed = np.array([mib[k] for k in common])
    m_abs = np.abs(m_signed)
    o_max = np.array([max_abs[k] for k in common])
    o_sum = np.array([signed_sum[k] for k in common])

    print("\nSpearman rank correlation (1 = identical ranking, 0 = unrelated)")
    print(f"  |MIB|  vs our max|score| over positions : {spearman(m_abs, o_max):.3f}")
    print(f"  |MIB|  vs |our sum over positions|      : {spearman(m_abs, np.abs(o_sum)):.3f}")
    print(f"  MIB    vs our signed sum (sign matters) : {spearman(m_signed, o_sum):.3f}")

    print("\nTop-k overlap by |score| (fraction of the top k shared)")
    print(f"  {'k':>6} {'% edges':>8} {'vs max|.|':>10} {'vs |sum|':>9}")
    for frac in (0.001, 0.005, 0.01, 0.05, 0.1):
        k = max(1, int(round(frac * len(common))))
        print(f"  {k:6d} {frac*100:7.1f}% {topk_overlap(common, m_abs, o_max, k):10.2f}"
              f" {topk_overlap(common, m_abs, np.abs(o_sum), k):9.2f}")

    agree = np.mean(np.sign(m_signed) == np.sign(o_sum))
    top = np.argsort(-m_abs)[:200]
    agree_top = np.mean(np.sign(m_signed[top]) == np.sign(o_sum[top]))
    print(f"\nsign agreement (MIB vs our sum): all edges {agree:.2f}, MIB top-200 {agree_top:.2f}")

    print(f"\nMIB top {args.show} edges and where ours ranks them (by max|score|):")
    our_rank = {common[i]: r for r, i in enumerate(np.argsort(-o_max), start=1)}
    for i in np.argsort(-m_abs)[:args.show]:
        k = common[i]
        print(f"  {k[0]:>12} -> {k[1]:<14} MIB {m_signed[i]:+9.4f}   our rank {our_rank[k]:>6}")


if __name__ == "__main__":
    main()
