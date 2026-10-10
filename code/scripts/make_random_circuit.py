"""
Random-baseline circuit in MIB's format.

Copies an existing MIB importances.json (any method, same model + task, so
the graph is identical) and replaces every edge score with a random number.
MIB then evaluates it exactly like a real method, so the random row in your
table is computed by the same code as everything else.

    python code/scripts/make_random_circuit.py \
        --like results/mib/circuits/EAP_patching_edge/ioi_gpt2/importances.json \
        --out  results/mib/circuits/random_patching_edge/ioi_gpt2/importances.json \
        --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, help="an existing importances.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.like) as f:
        g = json.load(f)

    rng = np.random.default_rng(args.seed)
    n = 0
    for key, val in g["edges"].items():
        if isinstance(val, dict):
            val["score"] = float(rng.standard_normal())
            if "in_graph" in val:
                val["in_graph"] = False
        else:
            g["edges"][key] = float(rng.standard_normal())
        n += 1
    for node in g.get("nodes", {}).values():
        if isinstance(node, dict) and "in_graph" in node:
            node["in_graph"] = False

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(g, f)
    print(f"random circuit: {n} edges, seed {args.seed} -> {out}")


if __name__ == "__main__":
    main()
