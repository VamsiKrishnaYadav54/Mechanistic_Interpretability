"""
Unified experiment runner -- Phase 0 final version.

    python code/scripts/run_experiment.py --config code/configs/ioi/eap.yaml
    python code/scripts/run_experiment.py --config code/configs/ioi/acdc.yaml

Run from the mech_interp root.

ACDC SAFETY: if <output_dir>/acdc_prune_scores.pt exists it is LOADED, never
recomputed. A new ACDC search (~68 h on GPT-2) only starts with
--rerun-acdc, and even then it never overwrites an existing checkpoint.

Pipeline -- every step names the function that does it (see PIPELINE.md):
  1. model    load_model()                      TransformerLens + 3 hook flags
  2. data     load_datasets_from_json()         clean/corrupt prompt pairs
  3. graph    patchable_model()                 edges = src x dest x position
  4. scores   mask_gradient_prune_scores() (EAP) / acdc_prune_scores() (ACDC)
  5. export   circuits.export.export_scores()   instance csv + structural json
  6. AUROC    structural_auroc() / instance_auroc()           (IOI only)
  7. circuits faithfulness_curve() -> run_circuits(TREE_PATCH) -> CPR, CMD
  8. save     results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

CODE = Path(__file__).resolve().parents[1]          # mech_interp/code
sys.path.insert(0, str(CODE / "external/auto-circuit"))
sys.path.insert(0, str(CODE))

from auto_circuit.data import load_datasets_from_json
from auto_circuit.types import PatchType
from auto_circuit.utils.graph_utils import patchable_model

from circuits.evaluation.circuit_metrics import (
    REPORTABLE_ABLATIONS,
    faithfulness_at_counts,
    faithfulness_curve,
    infer_ioi_word_idxs,
    instance_auroc,
    structural_auroc,
)
from circuits.export import export_scores


def banner(s):
    print("\n" + "=" * 70 + f"\n  {s}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------- step 1
def load_model(model_name: str, device: torch.device):
    """HookedTransformer with the three hook flags edge patching needs.

    use_attn_result      -> each head's output kept separate (A9.9 is a node)
    use_split_qkv_input  -> separate Q / K / V inputs per head (destinations)
    use_hook_mlp_in      -> MLP input is a destination
    """
    import transformer_lens as tl
    model = tl.HookedTransformer.from_pretrained(model_name)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    return model.to(device).eval()


def dataset_path(cfg: dict) -> Path:
    p = Path(cfg["dataset_file"]).expanduser()
    return p if p.is_absolute() else (CODE / "external/auto-circuit" / p).resolve()


# ---------------------------------------------------------------- step 4
def compute_scores(method, model, train_loader, cfg):
    if method == "eap":
        from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
        return mask_gradient_prune_scores(
            model=model, dataloader=train_loader, official_edges=None,
            grad_function=cfg.get("grad_function", "logit"),
            answer_function=cfg.get("answer_function", "avg_diff"),
            mask_val=cfg.get("mask_val", 0.0),
        )
    if method == "acdc":
        from auto_circuit.prune_algos.ACDC import acdc_prune_scores
        return acdc_prune_scores(
            model=model, dataloader=train_loader, official_edges=None,
            tao_exps=cfg.get("tao_exps", [-4, -3, -2]),
            tao_bases=cfg.get("tao_bases", [1]),
            faithfulness_target=cfg.get("faithfulness_target", "kl_div"),
        )
    if method == "eap_ig":
        raise SystemExit(
            "EAP-IG is not wired in yet (Phase 1). AutoCircuit's "
            "parameter_integrated_grads_prune_scores integrates over WEIGHTS "
            "and is not EAP-IG. Use external/EAP-IG once integrated.")
    raise SystemExit(f"unknown method: {method}")


def get_scores(method, model, train_loader, cfg, out_dir, args, device):
    """Load a saved checkpoint when allowed; otherwise compute and save."""
    ckpt = out_dir / f"{method}_prune_scores.pt"

    use_ckpt = ckpt.exists() and (
        (method == "acdc" and not args.rerun_acdc) or args.reuse_scores)
    if use_ckpt:
        print(f"  loading scores from checkpoint: {ckpt}")
        raw = torch.load(ckpt, map_location="cpu")
        return {k: v.to(device) for k, v in raw.items()}, "checkpoint", None

    if method == "acdc" and not args.rerun_acdc:
        raise SystemExit(
            f"No ACDC checkpoint at {ckpt}. Refusing to start a ~68 h search. "
            f"Pass --rerun-acdc if you really mean it.")

    t0 = time.time()
    scores = compute_scores(method, model, train_loader, cfg)
    secs = time.time() - t0
    print(f"  scored in {secs:.1f}s")

    save_to = ckpt
    if ckpt.exists():                       # never overwrite a checkpoint
        save_to = out_dir / f"{method}_prune_scores_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({k: v.cpu() for k, v in scores.items()}, save_to)
    print(f"  saved: {save_to}")
    return scores, "computed", secs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reuse-scores", action="store_true",
                    help="load <method>_prune_scores.pt if present (any method)")
    ap.add_argument("--rerun-acdc", action="store_true",
                    help="really start a new ACDC search (~68 h)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    method, task = cfg["method"], cfg["task"]
    out_dir = Path(cfg["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    banner(f"{cfg['name']} | task={task} method={method} model={cfg['model']}")

    # 1-3: model, data, graph
    model = load_model(cfg["model"], device)
    train_loader, test_loader = load_datasets_from_json(
        model=model, path=dataset_path(cfg), device=device, prepend_bos=True,
        batch_size=cfg.get("batch_size", 16),
        train_test_size=(cfg.get("train_size", 128), cfg.get("test_size", 128)),
        return_seq_length=True,
    )
    seq_labels = list(train_loader.seq_labels)
    model = patchable_model(
        model, factorized=True, slice_output="last_seq",
        seq_len=test_loader.seq_len, separate_qkv=True, device=device,
    )
    print(f"  seq_len={test_loader.seq_len}  edges={model.n_edges}")
    print(f"  tokens={seq_labels}")

    # 4: scores
    banner("STEP 4  scores")
    prune_scores, source, secs = get_scores(
        method, model, train_loader, cfg, out_dir, args, device)

    # 5: export
    banner("STEP 5  export")
    files = export_scores(model, prune_scores, seq_labels, out_dir,
                          method=method, model_name=cfg["model"])
    for k, v in files.items():
        print(f"  {k:10s} -> {v}")

    results = {
        "name": cfg["name"], "task": task, "method": method,
        "model": cfg["model"], "n_edges": model.n_edges,
        "scores_source": source, "scoring_seconds": secs,
        "patch_type": "TREE_PATCH", "files": files,
        "auroc": {}, "faithfulness": {}, "matched": {},
    }

    # 6: AUROC (needs a ground-truth circuit -> IOI only for now)
    if task == "ioi":
        banner("STEP 6  AUROC vs official IOI circuit")
        from auto_circuit.metrics.official_circuits.circuits.ioi_official import (
            ioi_true_edges, ioi_true_edges_mlp_0_only)
        word_idxs = infer_ioi_word_idxs(seq_labels)
        results["word_idxs"] = word_idxs
        print(f"  word_idxs = {word_idxs}")
        variants = [
            ("structural / MLP 0 only", structural_auroc, ioi_true_edges_mlp_0_only(model)),
            ("instance / MLP 0 only",   instance_auroc,
             ioi_true_edges_mlp_0_only(model, token_positions=True, word_idxs=word_idxs)),
            ("structural / all MLPs",   structural_auroc, ioi_true_edges(model)),
            ("instance / all MLPs",     instance_auroc,
             ioi_true_edges(model, token_positions=True, word_idxs=word_idxs)),
        ]
        for label, fn, edges in variants:
            r = fn(model, prune_scores, edges)
            results["auroc"][label] = r
            print(f"  {label:26s} {r['auroc']:.4f}  ({r['n_positive']} pos)")
    else:
        print("\n  STEP 6 skipped: no ground-truth circuit wired for this task")

    # 7: circuits -> behaviour -> CPR / CMD
    banner("STEP 7  faithfulness (TREE_PATCH)")
    match_counts = cfg.get("match_counts", [])
    for abl_name, abl in REPORTABLE_ABLATIONS.items():
        r = faithfulness_curve(model, test_loader, prune_scores,
                               ablation_type=abl, patch_type=PatchType.TREE_PATCH)
        results["faithfulness"][abl_name] = r
        print(f"\n  [{abl_name}]  CPR={r['CPR']:.4f}  CMD={r['CMD']:.4f}  "
              f"m(full)={r['m_full']:.4f}  m(empty)={r['m_empty']:.4f}  "
              f"sizes={r['n_distinct_achieved']}/{r['n_requested']}")
        for p in r["curve"]:
            print(f"    edges={p['n_edges']:7d}  f={p['faithfulness']:7.4f}")
        for w in r["warnings"]:
            print(f"    !! {w}")
        if match_counts:
            m = faithfulness_at_counts(model, test_loader, prune_scores,
                                       match_counts, ablation_type=abl)
            results["matched"][abl_name] = m
            print("    matched sizes: " + "  ".join(
                f"{row['n_edges']}->{row['faithfulness']:.4f}" for row in m["rows"]))

    # 8: save
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    banner(f"DONE -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
