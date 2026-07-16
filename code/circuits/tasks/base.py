"""Abstract base class for all circuit discovery tasks."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class TaskData:
    """Holds clean/corrupted prompt pairs for a task."""
    clean_prompts: List[str]
    corrupted_prompts: List[str]
    answer_tokens: List[int]      # correct answer token ids
    wrong_tokens: List[int]       # wrong answer token ids (for logit diff)


class BaseTask(ABC):
    """Every task (IOI, Greater-Than, etc.) inherits from this."""

    @abstractmethod
    def get_data(self, n_examples: int) -> TaskData:
        """Return clean/corrupted prompt pairs."""
        pass

    @abstractmethod
    def metric(self, logits: torch.Tensor, data: TaskData) -> torch.Tensor:
        """Compute task performance metric from logits.
        Returns a scalar — higher = better performance.
        """
        pass