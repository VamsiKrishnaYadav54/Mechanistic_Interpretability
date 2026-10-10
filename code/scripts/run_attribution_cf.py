"""
MIB circuit discovery with a CHOSEN counterfactual.

Identical to MIB-circuit-track/run_attribution.py except for one thing:
--counterfactual is passed to HFEAPDataset(counterfactual_type=...).
MIB's own script never passes it, so IOI always uses s2_io_flip.

The MIB submodule is not modified; this file lives in our code/scripts/.

    python code/scripts/run_attribution_cf.py --model gpt2 --task ioi \
        --method EAP-IG-inputs --counterfactual abc_counterfactual \
        --circuit-dir results/mib/circuits_cf/abc_counterfactual

IOI counterfactuals in mib-bench/ioi:
    s2_io_flip_counterfactual            (MIB default)
    abc_counterfactual
    s1_io_flip_counterfactual
    random_names_counterfactual
    random_names_s1_ioi_flip_counterfactual
    random_names_s2_ioi_flip_counterfactual
    s1_ioi_flip_s2_ioi_flip_counterfactual
    random_names_s1_ioi_flip_s2_ioi_flip_counterfactual
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

MIB_DIR = Path(__file__).resolve().parents[1] / "external" / "MIB-circuit-track"
sys.path.insert(0, str(MIB_DIR))

from eap.attribute import attribute                                     # noqa: E402
from eap.graph import Graph                                             # noqa: E402
from MIB_circuit_track.dataset import HFEAPDataset                      # noqa: E402
from MIB_circuit_track.metrics import get_metric                        # noqa: E402
from MIB_circuit_track.utils import (                                   # noqa: E402
    COL_MAPPING, MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--counterfactual", required=True)
    ap.add_argument("--ig-steps", type=int, default=5)
    ap.add_argument("--ablation", default="patching")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--num-examples", type=int, default=100)
    ap.add_argument("--circuit-dir", required=True)
    args = ap.parse_args()

    if f"{args.task.replace('_', '-')}_{args.model}" not in COL_MAPPING:
        raise SystemExit(f"{args.task}/{args.model} is not a MIB combination")

    if args.model in ("qwen2.5", "gemma2", "llama3"):
        model = HookedTransformer.from_pretrained(
            MODEL_NAME_TO_FULLNAME[args.model],
            attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[args.model])
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    graph = Graph.from_model(model)
    dataset = HFEAPDataset(
        f"mib-bench/{TASKS_TO_HF_NAMES[args.task]}", model.tokenizer,
        split=args.split, task=args.task, model_name=args.model,
        num_examples=args.num_examples,
        counterfactual_type=args.counterfactual,           # <-- the only change
    )
    print(f"counterfactual={args.counterfactual}  examples={len(dataset)}")
    try:                                   # show one pair so the log proves which CF ran
        item = dataset[0]
        print(f"  example clean  : {item[0]}")
        print(f"  example corrupt: {item[1]}")
    except Exception as exc:
        print(f"  (could not print example: {exc})")

    dataloader = dataset.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, model.tokenizer, model)
    attribute(model, graph, dataloader, partial(metric, mean=True, loss=True),
              args.method, args.ablation, ig_steps=args.ig_steps,
              intervention_dataloader=dataloader)

    out = os.path.join(args.circuit_dir, f"{args.method}_{args.ablation}_edge",
                       f"{args.task.replace('_', '-')}_{args.model}")
    os.makedirs(out, exist_ok=True)
    graph.to_json(os.path.join(out, "importances.json"))
    print(f"saved -> {out}/importances.json")


if __name__ == "__main__":
    main()