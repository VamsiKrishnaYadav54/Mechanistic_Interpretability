# Pipeline: from activations to a number

This file answers one question for any method: **how does a raw activation become a
reported number, and which function does each step?** If a step is not in this table,
it does not happen.

Entry point for everything: `code/scripts/run_experiment.py --config <yaml>`.

## The eight steps

| # | Step | Function (file) | What it does |
|---|------|-----------------|--------------|
| 1 | Model | `load_model()` (`code/scripts/run_experiment.py`) | Loads GPT-2 as a TransformerLens `HookedTransformer` and turns on three hook flags: `use_attn_result` (each head's output separate), `use_split_qkv_input` (separate Q/K/V inputs per head), `use_hook_mlp_in` (MLP input). |
| 2 | Data | `load_datasets_from_json()` (`auto_circuit/data.py`) | Loads clean/corrupt prompt pairs with the correct and wrong answer token per prompt. IOI vanilla template: 15 tokens. |
| 3 | Graph | `patchable_model()` (`auto_circuit/utils/graph_utils.py`) | Replaces each hook point with a `PatchWrapperImpl`. Edges = every (source, destination, token position) with source earlier than destination: 32,491 structural edges x 15 positions = 487,365. Each destination owns a mask tensor, e.g. `blocks.10.hook_v_input` = (15 positions, 12 heads, 131 sources). An edge is an address into one cell: `Edge.patch_idx`. Mask 0 = real activation, 1 = ablated activation. |
| 4 | Scores | EAP: `mask_gradient_prune_scores()` (`auto_circuit/prune_algos/mask_gradient.py`). ACDC: `acdc_prune_scores()` (`auto_circuit/prune_algos/ACDC.py`) | Fills a `PruneScores` dict (destination module -> tensor, same shape as its mask). **EAP:** one forward and one backward pass; score = gradient of the logit difference w.r.t. the mask at 0, a linear estimate of the effect of ablating that edge. **ACDC:** removes edges one by one from the output backwards and keeps a removal if the KL increase stays below `tao`; score = the `tao` at which the edge was removed, `inf` = never removed. Saved as `<method>_prune_scores.pt`. |
| 5 | Export | `export_scores()` (`code/circuits/export.py`) | Writes `scores_instance.csv.gz` (487,365 rows) and `scores_structural.json` (32,491 rows). Aggregation rule below. |
| 6 | Ranking + AUROC | `structural_auroc()`, `instance_auroc()` (`code/circuits/evaluation/circuit_metrics.py`) | Compares the score ranking against the official IOI circuit (`ioi_true_edges*` in `auto_circuit/.../ioi_official.py`). Scores and labels are **always in the same representation**: structural scores vs position-blind labels, or instance scores vs position-aware labels (`token_positions=True`, `word_idxs`). |
| 7 | Circuit | `run_circuits()` (`auto_circuit/prune.py`) via `faithfulness_curve()` | For a target size k, `prune_scores_threshold()` (`auto_circuit/utils/tensor_ops.py`) finds the k-th largest \|score\|; every edge at or above it is in the circuit. **Ties put more than k edges in** (ACDC: 223, 575, 2,384). With `PatchType.TREE_PATCH`, circuit edges keep their clean activation and every other edge gets the ablated one (`RESAMPLE` = activation from the corrupted prompt). Output: logits at the last token. |
| 8 | Number | `faithfulness_curve()`, `faithfulness_at_counts()` (`circuit_metrics.py`) | m = mean logit difference (correct - wrong), `batch_avg_answer_diff()`. Faithfulness f = (m(circuit) - m(empty)) / (m(full) - m(empty)). CPR = area under f over MIB's k grid {.001 .002 .005 .01 .02 .05 .1 .2 .5 1}; CMD = area under \|1 - f\|; trapezoid rule. |

## Aggregation rule (instance -> structural)

For each structural edge (src -> dest), over its 15 token positions:

- `abs_score` = max \|score\|
- `score` = the signed score at the winning position; `best_pos`, `best_token` = that position
- `is_inf` = true if any position is +/-inf (ACDC survivors); then `score` and `abs_score` are null
- Rank: inf first, then `abs_score` descending. `tied_with_next` marks equal keys.

## Conventions that were wrong before 2026-10-08

| Thing | Correct | Was |
|-------|---------|-----|
| Patch type for faithfulness | `TREE_PATCH` (keeps the circuit) | `EDGE_PATCH` |
| AUROC labels | same representation as the scores | position-blind labels vs position-specific scores -> 0.59 instead of 0.94 |
| ACDC `inf` before `run_circuits` | replaced by a large finite value (`_finite_scores`) | the "empty" circuit contained 223 edges |
| Ablation for headline numbers | `RESAMPLE` (MIB counterfactual) | zero ablation also reported; off-distribution, dropped |
| Headline AUROC labels | MLP-0-only (the two label versions agree: 242 edges) | all-MLPs, where structural (739) and positioned (771) differ because of `dest_layer = conn.qkv[0][1]` in `ioi_official.py` |

## Results layout

```
results/<run>/
  config.yaml                the config used
  <method>_prune_scores.pt   raw PruneScores (ACDC: never overwritten)
  scores_instance.csv.gz     step 5
  scores_structural.json     step 5, input to the visualiser
  results.json               steps 6-8
results/figures/             visualiser output (.png, .pdf, .csv)
```

## Visualise

```
python code/scripts/visualize_circuit.py --scores results/ioi_eap_gpt2/scores_structural.json --top 40 --out results/figures/eap_top40
python code/scripts/visualize_circuit.py --compare results/ioi_eap_gpt2/scores_structural.json results/ioi_acdc_gpt2/scores_structural.json --top 40 --out results/figures/eap_vs_acdc
```

## Rules

- Never delete or overwrite `results/ioi_acdc_gpt2/acdc_prune_scores.pt` (68 h). A backup lives in `~/VAMSI/backups/`.
- `run_experiment.py` loads the ACDC checkpoint automatically; a new search needs `--rerun-acdc`.
- EAP-IG is not wired in. AutoCircuit's `parameter_integrated_grads_prune_scores` is **not** EAP-IG.
- A high attribution score is not causal proof. AUROC (ranking) and faithfulness (circuit) answer different questions.
