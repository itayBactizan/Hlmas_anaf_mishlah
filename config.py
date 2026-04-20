import os
import torch
from dataclasses import dataclass, field
from typing import Literal
from .exceptions import PipelineInitializationError

@dataclass
class InferenceConfig:
    """
    Centralized configuration for the Dual Multiple Choice Inference Pipeline.
    Validates all critical parameters upon instantiation to ensure fail-fast behavior.
    """
    model_id: str
    temp_dir: str
    logs_dir: str
    key_column_name: str
    prediction_column_name: str
    precision_column_name: str
    file_suffix: str
    
    simul_type: Literal['anaf', 'mishlah', 'both'] = 'both'
    precision: Literal["int8", "fp16", "fp32"] = "int8"
    batch_size: int = 64          
    io_chunk_size: int = 10000     
    top_k_candidates: int = 10    
    use_chunked: bool = True
    input_col_name: str = "input"
    
    feature_prefix: str = field(init=False)
    logit_name: str = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        """Validates types and logic immediately after dataclass creation."""
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        if not isinstance(self.io_chunk_size, int) or self.io_chunk_size <= 0:
            raise ValueError(f"io_chunk_size must be a positive integer, got {self.io_chunk_size}")
            
        if self.simul_type not in ('anaf', 'mishlah', 'both'):
            raise ValueError(f"Invalid simul_type '{self.simul_type}'. Expected 'anaf', 'mishlah', or 'both'.")

        # Map domain-specific prefixes based on the active task
        if self.simul_type in ('mishlah', 'mishlach'):
            self.feature_prefix = "occ"
            self.logit_name = "occ_logits"
        else:
            self.feature_prefix = "sec"
            self.logit_name = "sec_logits"

        # Force CPU execution to prevent GPU thrashing in threaded environments
        self.device = torch.device("cpu")
        
        # Ensure critical directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)