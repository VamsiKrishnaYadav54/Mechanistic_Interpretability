"""Abstract base class for all edge scoring methods."""
from abc import ABC, abstractmethod
import torch
from transformer_lens import HookedTransformer
from circuits.tasks.base import BaseTask


class BaseScorer(ABC):
    """ACDC, EAP, Fisher — all inherit from this."""

    def __init__(self, model: HookedTransformer, task: BaseTask):
        self.model = model
        self.task = task

    @abstractmethod
    def score_edges(self, n_examples: int = 10) -> dict:
        """Return dict mapping edge_name -> importance_score."""
        pass