"""Faithfulness metrics for evaluating circuits."""
import torch
from transformer_lens import HookedTransformer


def circuit_performance_recovery(
    model: HookedTransformer,
    circuit_logits: torch.Tensor,
    full_model_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    answer_token_id: int
) -> float:
    """CPR: how much of the full model's performance does the circuit recover?
    
    CPR = (circuit_score - baseline_score) / (full_model_score - baseline_score)
    CPR = 1.0 means circuit perfectly recovers full model behavior
    CPR = 0.0 means circuit is no better than baseline (random/ablated)
    """
    def logit_score(logits):
        return logits[:, -1, answer_token_id].mean().item()

    circuit_score    = logit_score(circuit_logits)
    full_score       = logit_score(full_model_logits)
    baseline_score   = logit_score(baseline_logits)

    denom = full_score - baseline_score
    if abs(denom) < 1e-8:
        return 0.0
    return (circuit_score - baseline_score) / denom