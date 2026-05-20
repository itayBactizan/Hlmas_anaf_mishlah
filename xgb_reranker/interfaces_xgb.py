import pandas as pd
import torch
from typing import Protocol, List, Tuple, Optional, Any

class DataPreprocessor(Protocol):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and serializes raw input data for the retrieval step."""
        ...

class RuleEngine(Protocol):
    def evaluate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits data into a resolved DataFrame (fast-path) and an unresolved DataFrame (ML-path)."""
        ...

class CandidateRetriever(Protocol):
    def search(self, embeddings: torch.Tensor, top_k: int) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        """Retrieves top-k candidate codes, labels, and similarity scores for a batch of dense vectors."""
        ...

class ContextEnricher(Protocol):
    def enrich(self, queries: List[str], candidate_codes: List[List[str]], candidate_labels: List[List[str]],
                scores=Optional[List[List[float]]] = None, rows_ids: Optional[List[List[Any]]] =None ) -> Tuple[List[str], List[List[str]]]:
        """Fuses retrieved candidates with external dictionary definitions to create context-rich strings."""
        ...

class DynamicTruncatorProtocol(Protocol):
    def truncate(self, codes: List[List[str]], labels: List[List[str]], scores: List[List[float]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Applies margin thresholds to dynamically reduce the candidate list size, saving compute."""
        ...

class CrossEncoderScorer(Protocol):
    def score(self, queries: List[str], candidates: List[List[str]], feature_prefix: str, logit_name: str) -> torch.Tensor:
        """Executes the forward pass of the cross-encoder model and returns raw logits."""
        ...

class PredictionPostProcessor(Protocol):
    def resolve(self, row_ids: List[any], logits: torch.Tensor, candidate_codes: List[List[str]], task_name: str, is_multi_task: bool) -> pd.DataFrame:
        """Translates raw model logits into final business metrics (predictions and precision values)."""
        ...