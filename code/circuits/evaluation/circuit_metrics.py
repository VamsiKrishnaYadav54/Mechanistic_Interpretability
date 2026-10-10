"""
Corrected circuit evaluation metrics -- v2.

Changes from v1, after the 2026-10-08 rebuild run:

1. FIXED crash "FAILED: 0". faithfulness_curve seeded its point list with
   (k=0, f=0, n_edges=0) and then looked up m_by_count[0] when building the
   output. For ACDC, 0 is never an achievable circuit size (tied scores snap
   every request to a group boundary), so that lookup raised KeyError: 0.
   The k=0 anchor is now used only for the integral, never looked up.

2. FIXED check_patch_type. The old "matches_full_model_at_k=1" test was
   useless -- at k=1 every edge is in the circuit, so BOTH patch types equal
   the full model and both returned true. The decisive test is at a SMALL
   circuit: the faithfulness patch type lands between m(empty) and m(full),
   the other one sits at or above m(full) because it has only ablated a
   handful of edges. Now returns an explicit verdict.

3. DROPPED zero ablation from the defaults. On GPT-2 it is wildly
   off-distribution -- the 2026-10-08 run gave faithfulness swinging from
   +2.39 to -2.46 across circuit sizes, which is noise, not a curve. MIB
   uses counterfactual, mean and optimal ablation, not zero. Default is now
   resample (= MIB counterfactual); mean is available as a second row.

4. ADDED a monotonicity warning. A faithfulness curve that goes down as the
   circuit grows is worth knowing about rather than averaging over.
"""

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch as t
from sklearn.metrics import roc_auc_score

from auto_circuit.prune import run_circuits
from auto_circuit.types import AblationType, Edge, PatchType, PruneScores
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff

MIB_K_VALUES: Tuple[float, ...] = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0
)

# The two worth reporting on GPT-2. Zero ablation is deliberately absent.
REPORTABLE_ABLATIONS = {
    "resample": AblationType.RESAMPLE,                 # MIB counterfactual
    "mean": AblationType.TOKENWISE_MEAN_CORRUPT,       # Wang et al. style
}


# =====================================================================
# helpers
# =====================================================================

def _abs_scores_as_numpy(prune_scores: PruneScores) -> Dict[str, np.ndarray]:
    """|score| per module as numpy, inf mapped to a large finite value."""
    out: Dict[str, np.ndarray] = {}
    for mod, v in prune_scores.items():
        a = v.detach().abs().float().cpu().numpy()
        finite = a[np.isfinite(a)]
        big = (float(finite.max()) * 10.0 + 1.0) if finite.size else 1.0
        out[mod] = np.nan_to_num(a, nan=0.0, posinf=big, neginf=0.0)
    return out


def infer_ioi_word_idxs(seq_labels: Sequence[str]) -> Dict[str, int]:
    """Work out IO / S1 / S1+1 / S2 / end from the tokenised template."""
    stripped = [s.strip() for s in seq_labels]
    seen: Dict[str, List[int]] = defaultdict(list)
    for i, tok in enumerate(stripped):
        if tok[:1].isupper() and tok.isalpha() and tok != "When":
            seen[tok].append(i)

    repeated = {w: idxs for w, idxs in seen.items() if len(idxs) >= 2}
    if len(repeated) != 1:
        raise ValueError(
            f"could not find exactly one repeated name in {seq_labels!r}; "
            f"candidates={dict(seen)}. Pass word_idxs by hand."
        )
    _, s_idxs = next(iter(repeated.items()))
    singles = [w for w, idxs in seen.items() if len(idxs) == 1]
    if len(singles) != 1:
        raise ValueError(
            f"could not find exactly one non-repeated name in {seq_labels!r}. "
            f"Pass word_idxs by hand."
        )
    return {
        "IO": seen[singles[0]][0],
        "S1": s_idxs[0],
        "S1+1": s_idxs[0] + 1,
        "S2": s_idxs[1],
        "end": len(stripped) - 1,
    }


# =====================================================================
# AUROC  (unchanged from v1 -- these worked)
# =====================================================================

def structural_auroc(
    model: PatchableModel,
    prune_scores: PruneScores,
    official_edges: Set[Edge],
    aggregate: str = "max",
) -> Dict[str, float]:
    """AUROC over STRUCTURAL edges. Pair with ioi_true_edges(model)."""
    scores = _abs_scores_as_numpy(prune_scores)
    official_names = {e.name for e in official_edges}

    buckets: Dict[str, List[float]] = defaultdict(list)
    for e in model.edges:
        if e.dest.module_name not in scores:
            continue
        buckets[e.name].append(float(scores[e.dest.module_name][e.patch_idx]))

    names = sorted(buckets)
    reduce = {"max": max,
              "mean": lambda v: float(np.mean(v)),
              "sum": lambda v: float(np.sum(v))}[aggregate]
    y_score = np.array([reduce(buckets[n]) for n in names], dtype=np.float64)
    y_true = np.array([1 if n in official_names else 0 for n in names],
                      dtype=np.int8)

    if y_true.sum() in (0, y_true.size):
        raise ValueError("labels are all one class -- representation mismatch")

    return {"auroc": float(roc_auc_score(y_true, y_score)),
            "n_items": int(y_true.size),
            "n_positive": int(y_true.sum()),
            "aggregate": aggregate}


def instance_auroc(
    model: PatchableModel,
    prune_scores: PruneScores,
    official_edges_positioned: Set[Edge],
) -> Dict[str, float]:
    """AUROC over EDGE INSTANCES. Pair with token_positions=True."""
    scores = _abs_scores_as_numpy(prune_scores)
    labels = model.circuit_prune_scores(official_edges_positioned, bool=True)

    y_true_parts, y_score_parts = [], []
    for mod in sorted(scores):
        if mod not in labels:
            continue
        y_true_parts.append(
            labels[mod].detach().cpu().numpy().reshape(-1).astype(np.int8))
        y_score_parts.append(scores[mod].reshape(-1))

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    if y_true.sum() == 0:
        raise ValueError("no positive labels -- did you pass the "
                         "token_positions=False edge set?")

    return {"auroc": float(roc_auc_score(y_true, y_score)),
            "n_items": int(y_true.size),
            "n_positive": int(y_true.sum())}


# =====================================================================
# faithfulness, CPR, CMD
# =====================================================================

def _finite_scores(prune_scores: PruneScores) -> PruneScores:
    """Replace +/-inf with a large finite value, keeping sign and order.

    Needed for ACDC. AutoCircuit builds the 0-edge circuit by using a
    threshold of +inf and keeping every edge with |score| >= inf. ACDC
    stores inf for edges that survived every tao, so those edges all pass
    the test and the "empty" circuit actually contains them (223 edges in
    our run). With inf replaced by a finite value, nothing passes at
    threshold inf and the empty circuit is really empty.
    """
    out: PruneScores = {}
    for mod, v in prune_scores.items():
        v = v.detach().clone().float()
        finite = v[t.isfinite(v)]
        big = float(finite.abs().max()) * 10.0 + 1.0 if finite.numel() else 1.0
        v[t.isposinf(v)] = big
        v[t.isneginf(v)] = -big
        v[t.isnan(v)] = 0.0
        out[mod] = v
    return out


def _mean_logit_diff(dataloader, batch_outputs: Dict) -> float:
    vals = [batch_avg_answer_diff(batch_outputs[b.key], b).item()
            for b in dataloader]
    return float(np.mean(vals))


def full_model_logit_diff(model: PatchableModel, dataloader) -> float:
    vals = []
    with t.inference_mode():
        for batch in dataloader:
            vals.append(
                batch_avg_answer_diff(model(batch.clean)[model.out_slice],
                                      batch).item())
    return float(np.mean(vals))


def faithfulness_curve(
    model: PatchableModel,
    dataloader,
    prune_scores: PruneScores,
    k_values: Sequence[float] = MIB_K_VALUES,
    ablation_type: AblationType = AblationType.RESAMPLE,
    patch_type: PatchType = PatchType.TREE_PATCH,
) -> Dict[str, object]:
    """Faithfulness at each circuit size, plus CPR and CMD.

    patch_type must KEEP the circuit and ablate everything else. On the
    2026-10-08 run TREE_PATCH was confirmed to be that one: it rose from
    0.1177 at 0 edges to 3.3549 at all edges, while EDGE_PATCH gave 3.5734
    at 487 edges -- above the full model, so not a faithfulness curve.
    """
    prune_scores = _finite_scores(prune_scores)
    n = model.n_edges
    k_to_count = {k: min(n, max(1, int(round(k * n)))) for k in k_values}
    counts = sorted(set(k_to_count.values()) | {0})

    circ_outs = run_circuits(
        model=model, dataloader=dataloader, test_edge_counts=counts,
        prune_scores=prune_scores, patch_type=patch_type,
        ablation_type=ablation_type,
    )

    m_by_count = {c: _mean_logit_diff(dataloader, out)
                  for c, out in circ_outs.items()}
    if not m_by_count:
        raise ValueError("run_circuits returned nothing")

    m_full = full_model_logit_diff(model, dataloader)

    # m(empty): the 0-edge circuit when it exists, else the smallest built.
    # ACDC's tied scores mean 0 is often NOT achievable -- this is the line
    # that used to crash.
    empty_count = 0 if 0 in m_by_count else min(m_by_count)
    m_empty = m_by_count[empty_count]

    denom = m_full - m_empty
    if abs(denom) < 1e-8:
        raise ValueError(
            f"m(full)={m_full:.6f} equals m(empty)={m_empty:.6f}; "
            "the ablation is doing nothing")

    achieved = sorted(m_by_count)
    rows: List[Dict[str, object]] = []
    seen_counts = set()
    for k in k_values:
        req = k_to_count[k]
        got = min((c for c in achieved if c >= req), default=max(achieved))
        if got in seen_counts:
            continue
        seen_counts.add(got)
        rows.append({
            "k_requested": k,
            "k_actual": got / n,
            "n_edges": got,
            "faithfulness": (m_by_count[got] - m_empty) / denom,
            "logit_diff": m_by_count[got],
        })
    rows.sort(key=lambda r: r["n_edges"])

    # Integrate from k=0, f=0. The anchor is never looked up in m_by_count.
    ks = np.array([0.0] + [float(r["k_actual"]) for r in rows])
    fs = np.array([0.0] + [float(r["faithfulness"]) for r in rows])

    f_vals = [float(r["faithfulness"]) for r in rows]
    drops = [(rows[i - 1]["n_edges"], rows[i]["n_edges"],
              round(f_vals[i - 1], 4), round(f_vals[i], 4))
             for i in range(1, len(f_vals)) if f_vals[i] < f_vals[i - 1] - 0.02]

    warnings: List[str] = []
    if len(rows) < len(k_values):
        warnings.append(
            f"only {len(rows)} distinct circuit sizes were achievable out of "
            f"{len(k_values)} requested -- tied scores. CPR/CMD from this few "
            f"points are coarse and not comparable to a method with more.")
    if drops:
        warnings.append(
            f"faithfulness DROPS as the circuit grows at {len(drops)} "
            f"point(s): {drops}. Adding edges made the circuit worse.")
    if empty_count != 0:
        warnings.append(
            f"a 0-edge circuit was not achievable; m(empty) was taken at "
            f"{empty_count} edges, so faithfulness is measured against a "
            f"floor that already contains {empty_count} edges.")

    return {
        "curve": rows,
        "m_full": m_full,
        "m_empty": m_empty,
        "m_empty_at_n_edges": empty_count,
        "CPR": float(np.trapz(fs, ks)),
        "CMD": float(np.trapz(np.abs(1.0 - fs), ks)),
        "ablation_type": str(ablation_type),
        "patch_type": str(patch_type),
        "n_requested": len(k_values),
        "n_distinct_achieved": len(rows),
        "warnings": warnings,
    }


def faithfulness_at_counts(
    model: PatchableModel,
    dataloader,
    prune_scores: PruneScores,
    counts: Sequence[int],
    ablation_type: AblationType = AblationType.RESAMPLE,
    patch_type: PatchType = PatchType.TREE_PATCH,
) -> Dict[str, object]:
    """Faithfulness at specific circuit sizes.

    Used for the fair EAP-vs-ACDC comparison: ACDC can only build a few
    sizes (its tied scores snap every request to a group boundary), so
    evaluate BOTH methods at exactly those sizes instead of comparing a
    10-point CPR against a 3-point one.
    """
    prune_scores = _finite_scores(prune_scores)
    wanted = sorted(set(int(c) for c in counts) | {0})
    outs = run_circuits(
        model=model, dataloader=dataloader, test_edge_counts=wanted,
        prune_scores=prune_scores, patch_type=patch_type,
        ablation_type=ablation_type,
    )
    m = {c: _mean_logit_diff(dataloader, o) for c, o in outs.items()}
    m_full = full_model_logit_diff(model, dataloader)
    m_empty = m[0] if 0 in m else m[min(m)]
    denom = m_full - m_empty

    achieved = sorted(m)
    rows = []
    for c in sorted(set(int(x) for x in counts)):
        got = min((a for a in achieved if a >= c), default=max(achieved))
        rows.append({
            "n_requested": c,
            "n_edges": got,
            "logit_diff": m[got],
            "faithfulness": (m[got] - m_empty) / denom,
        })
    return {"m_full": m_full, "m_empty": m_empty,
            "m_empty_at_n_edges": 0 if 0 in m else min(m), "rows": rows}


def check_patch_type(model: PatchableModel, dataloader,
                     prune_scores: PruneScores) -> Dict[str, object]:
    """Decide which PatchType keeps the circuit. Run once.

    The test is at a SMALL circuit, not at k=1. At k=1 both patch types are
    the full model, which is why the v1 version of this check reported true
    for both and told you nothing.

    Correct (faithfulness) behaviour at a small circuit:
        m(empty) < m(small circuit) < m(full)
    The other patch type ablates only a handful of edges, so it sits at or
    above m(full).
    """
    n = model.n_edges
    small = max(1, n // 1000)
    counts = [0, small, n]
    m_full = full_model_logit_diff(model, dataloader)
    report: Dict[str, object] = {"m_full": m_full, "small_circuit_n_edges": small}
    verdict = None

    for pt in (PatchType.TREE_PATCH, PatchType.EDGE_PATCH):
        outs = run_circuits(
            model=model, dataloader=dataloader, test_edge_counts=counts,
            prune_scores=prune_scores, patch_type=pt,
            ablation_type=AblationType.RESAMPLE,
        )
        got = {c: _mean_logit_diff(dataloader, o) for c, o in outs.items()}
        m_empty = got[min(got)]
        m_small = got[min((c for c in got if c >= small), default=max(got))]
        between = min(m_empty, m_full) < m_small < max(m_empty, m_full)
        report[str(pt)] = {
            "logit_diff_by_n_edges": got,
            "m_small_between_empty_and_full": bool(between),
        }
        if between:
            verdict = str(pt)

    report["USE_THIS_PATCH_TYPE"] = verdict or "INCONCLUSIVE -- inspect by hand"
    return report
