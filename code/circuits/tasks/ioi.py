"""Indirect Object Identification (IOI) task.

Classic MI task from Wang et al. 2022.
Prompt: "When Mary and John went to the store, John gave a drink to"
Answer: " Mary"  (not John)
Corrupted: swap the names so the correct answer changes.
"""
from circuits.tasks.base import BaseTask, TaskData
import torch
from typing import List


# Small fixed dataset for reproducibility
IOI_EXAMPLES = [
    {
        "clean":     "When Mary and John went to the store, John gave a drink to",
        "corrupted": "When John and Mary went to the store, Mary gave a drink to",
        "answer":    " Mary",
        "wrong":     " John",
    },
    {
        "clean":     "When Sarah and Tom went to the park, Tom gave a ball to",
        "corrupted": "When Tom and Sarah went to the park, Sarah gave a ball to",
        "answer":    " Sarah",
        "wrong":     " Tom",
    },
    {
        "clean":     "When Emma and James went to the office, James gave a pen to",
        "corrupted": "When James and Emma went to the office, Emma gave a pen to",
        "answer":    " Emma",
        "wrong":     " James",
    },
]


class IOITask(BaseTask):

    def get_data(self, n_examples: int = 3) -> TaskData:
        examples = IOI_EXAMPLES[:n_examples]
        return TaskData(
            clean_prompts    = [e["clean"]     for e in examples],
            corrupted_prompts= [e["corrupted"] for e in examples],
            answer_tokens    = [e["answer"]    for e in examples],
            wrong_tokens     = [e["wrong"]     for e in examples],
        )

    def metric(self, logits: torch.Tensor, data: TaskData) -> torch.Tensor:
        """Logit difference: logit(correct) - logit(wrong).
        Higher = model is more confident about the correct answer.
        """
        # logits shape: [batch, seq_len, vocab_size]
        # We care about the last token position
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        return last_logits.mean(0)       # placeholder — will implement properly in Phase 2
