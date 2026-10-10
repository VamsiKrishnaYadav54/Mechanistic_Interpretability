"""
Which edges survive a change of counterfactual?

Reads one importances.json per counterfactual (same model, task, method),
takes each circuit's top-k edges by |score|, and reports:

  1. pairwise overlap (Jaccard) between counterfactuals
  2. per-edge STABILITY = fraction of counterfactuals whose top-k contains it
  3. the direct-vs-upstream split: are edges INTO THE LOGITS more stable than
     edges into other heads / MLPs?  (hypothesis from the Phase 1 cross-check)

CPU only, runs on the login node:
    python code/scripts/compare_cf_circuits.py \
        --root results/mib/circuits_cf --method EAP-IG-inputs --model gpt2 --task ioi \
        --k 32 162 324 --out results/mib/cf_analysis
"""

import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

EDGE_RE = re.compile(r"^(?P<src>.+?)->(?P<dst>.+?)(?:<(?P<qkv>[qkv])>)?$")

SHORT = {
    "s2_io_flip_counterfactual": "s2_flip (MIB default)",
    "abc_counterfactual": "abc",
    "s1_io_flip_counterfactual": "s1_flip",
    "random_names_counterfactual": "random_names",
    "random_names_s1_ioi_flip_counterfactual": "rn+s1_flip",
    "random_names_s2_ioi_flip_counterfactual": "rn+s2_flip",
    "s1_ioi_flip_s2_ioi_flip_counterfactual": "s1+s2_flip",
    "random_names_s1_ioi_flip_s2_ioi_flip_counterfactual": "rn+s1+s2_flip",
}


def load_scores(path):
    with open(path) as f:
        edges = json.load(f)["edges"]
    return {k: abs(v["score"] if isinstance(v, dict) else float(v)) for k, v in edges.items()}


def topk(scores, k):
    return set(sorted(scores, key=scores.get, reverse=True)[:k])


def is_direct(edge):
    m = EDGE_RE.match(edge)
    return m is not None and m.group("dst") == "logits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--method", default="EAP-IG-inputs")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--k", type=int, nargs="+", default=[32, 162, 324])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    sub = Path(f"{args.method}_patching_edge") / f"{args.task.replace('_', '-')}_{args.model}" / "importances.json"
    runs = {}
    for cf_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = cf_dir / sub
        if f.exists():
            runs[cf_dir.name] = load_scores(f)
    if len(runs) < 2:
        raise SystemExit(f"need >= 2 counterfactual runs under {root}, found {list(runs)}")
    names = list(runs)
    print(f"{len(names)} counterfactuals, method={args.method}, model={args.model}\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for k in args.k:
        tops = {n: topk(runs[n], k) for n in names}

        # 1. pairwise Jaccard
        J = np.eye(len(names))
        for i, j in combinations(range(len(names)), 2):
            a, b = tops[names[i]], tops[names[j]]
            J[i, j] = J[j, i] = len(a & b) / len(a | b)
        off = J[~np.eye(len(names), dtype=bool)]

        # 2. stability
        union = set().union(*tops.values())
        stab = {e: sum(e in tops[n] for n in names) / len(names) for e in union}
        core = sorted([e for e, s in stab.items() if s == 1.0])

        # 3. direct vs upstream
        d = [s for e, s in stab.items() if is_direct(e)]
        u = [s for e, s in stab.items() if not is_direct(e)]

        print(f"===== top-{k} edges =====")
        print(f"  mean pairwise Jaccard         : {off.mean():.3f}  (min {off.min():.3f}, max {off.max():.3f})")
        print(f"  edges in ALL {len(names)} circuits     : {len(core)} of {k}")
        print(f"  union across counterfactuals  : {len(union)} edges")
        print(f"  mean stability, into logits   : {np.mean(d) if d else float('nan'):.3f}  ({len(d)} edges)")
        print(f"  mean stability, upstream      : {np.mean(u) if u else float('nan'):.3f}  ({len(u)} edges)")
        print("  Jaccard matrix:")
        w = max(len(SHORT.get(n, n)) for n in names)
        for i, n in enumerate(names):
            print(f"    {SHORT.get(n, n):>{w}}  " + " ".join(f"{x:.2f}" for x in J[i]))
        print(f"  edges present in every circuit: {core[:20]}{' ...' if len(core) > 20 else ''}\n")

        with open(out / f"stability_top{k}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["edge", "stability", "into_logits"] + names)
            for e in sorted(union, key=lambda e: -stab[e]):
                wr.writerow([e, f"{stab[e]:.3f}", is_direct(e)] + [int(e in tops[n]) for n in names])

        summary[k] = {"mean_jaccard": float(off.mean()), "n_core": len(core),
                      "union": len(union),
                      "stability_direct": float(np.mean(d)) if d else None,
                      "stability_upstream": float(np.mean(u)) if u else None,
                      "jaccard": J.tolist(), "names": names, "core_edges": core}

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = args.k[0]
        J = np.array(summary[k]["jaccard"])
        labels = [SHORT.get(n, n) for n in names]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center",
                        color="white" if J[i, j] < 0.6 else "black", fontsize=8)
        ax.set_title(f"{args.method} on {args.model}/{args.task}: overlap of top-{k} edges\nacross counterfactuals (Jaccard)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / f"jaccard_top{k}.png", dpi=160)
        print(f"saved {out / f'jaccard_top{k}.png'}")
    except Exception as exc:                       # plotting is optional
        print(f"(heatmap skipped: {exc})")
    print(f"saved {out}/summary.json and stability_top*.csv")


if __name__ == "__main__":
    main()