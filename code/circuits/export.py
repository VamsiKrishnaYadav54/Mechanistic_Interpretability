"""
Standard score export -- the ONE format every method writes.

Two files per run, in the run's results folder:

  scores_instance.csv.gz     one row per EDGE INSTANCE (src -> dest at one
                             token position). 487,365 rows for GPT-2 / IOI.
                             Columns: src, dest, seq_pos, token, score, is_inf

  scores_structural.json     one row per STRUCTURAL EDGE (src -> dest, the 15
                             positions collapsed). 32,491 rows for GPT-2 / IOI.
                             Sorted by importance, rank 1 = most important.

AGGREGATION RULE (instance -> structural):
    abs_score   = max over the 15 positions of |score|
    score       = the SIGNED score at that winning position
    best_pos    = the winning position, best_token = the token there
    is_inf      = True if ANY position is +/-inf (ACDC: survived every tao)
    For inf rows abs_score and score are null; use is_inf and sign.

Ranking: inf rows first, then by abs_score descending. Rows with exactly
equal keys are a TIE -- tied_with_next marks them. ACDC has large ties;
any "top N" cut that lands inside a tie is arbitrary unless the whole tie
group is taken (the visualiser does that).
"""

import csv
import gzip
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

AGGREGATION_RULE = (
    "max |score| over token positions; signed score and position kept "
    "from the winning position; inf at any position -> is_inf"
)


def export_scores(
    model,
    prune_scores: Dict,
    seq_labels: Sequence[str],
    out_dir,
    method: str,
    model_name: str,
    extra_meta: Optional[Dict] = None,
) -> Dict[str, str]:
    """Write scores_instance.csv.gz and scores_structural.json.

    Returns the two paths. Works for any method: it only reads the
    PruneScores dict (module name -> tensor) and the model's edge list.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = {m: v.detach().float().cpu().numpy() for m, v in prune_scores.items()}

    edges = [e for e in model.edges if e.dest.module_name in ps]
    edges.sort(key=lambda e: (e.dest.module_name, e.patch_idx))

    agg: Dict[tuple, Dict] = {}
    inst_path = out_dir / "scores_instance.csv.gz"
    with gzip.open(inst_path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dest", "seq_pos", "token", "score", "is_inf"])
        for e in edges:
            v = float(ps[e.dest.module_name][e.patch_idx])
            pos = e.seq_idx
            tok = seq_labels[pos] if (pos is not None and pos < len(seq_labels)) else ""
            is_inf = math.isinf(v)
            w.writerow([
                e.src.name, e.dest.name,
                "" if pos is None else pos, tok,
                ("inf" if v > 0 else "-inf") if is_inf else repr(v),
                int(is_inf),
            ])

            key = (e.src.name, e.dest.name)
            a = math.inf if is_inf else abs(v)
            cur = agg.get(key)
            if cur is None or a > cur["a"]:
                agg[key] = {"a": a, "v": v, "pos": pos, "tok": tok}

    rows: List[Dict] = []
    for (src, dest), r in agg.items():
        is_inf = math.isinf(r["a"])
        rows.append({
            "src": src,
            "dest": dest,
            "score": None if is_inf else r["v"],
            "abs_score": None if is_inf else r["a"],
            "sign": int(np.sign(r["v"])),
            "is_inf": is_inf,
            "best_pos": r["pos"],
            "best_token": r["tok"],
        })

    def sort_key(r):
        return (r["is_inf"], r["abs_score"] or 0.0)

    rows.sort(key=sort_key, reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["tied_with_next"] = (i + 1 < len(rows)
                               and sort_key(rows[i + 1]) == sort_key(r))

    n_inf = sum(r["is_inf"] for r in rows)
    meta = {
        "method": method,
        "model": model_name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregation_rule": AGGREGATION_RULE,
        "seq_len": len(seq_labels),
        "seq_labels": list(seq_labels),
        "n_instances": len(edges),
        "n_structural": len(rows),
        "n_structural_inf": n_inf,
        "n_distinct_structural_scores": len({sort_key(r) for r in rows}),
    }
    if extra_meta:
        meta.update(extra_meta)

    struct_path = out_dir / "scores_structural.json"
    with open(struct_path, "w") as f:
        json.dump({"meta": meta, "edges": rows}, f, indent=1)

    return {"instance": str(inst_path), "structural": str(struct_path)}
