
class InferencePipelineError(Exception):
    """Base exception for all errors"""

class PipelineInitializationError(InferencePipelineError):
    """Raised when the pipeline fails to initialize (e.g., missing weights, bad config)."""
    def __init__(self, message: str) -> None:
        super().__init__(f"Critical: System failed to initialize. {message}")

class DataSchemaError(Exception):
    """Raised when the input dataframe is missing required columns."""
    def __init__(self, missing_col: str) -> None:
        super().__init__(f"Schema Error: Missing requirment column. {missing_col}")

class InferenceExecutionError(Exception):
    """Raised when a specific batch fails during the ML forward pass."""
    def __init__(self, chunk_idx: int, original_exception: Exception) -> None:
        super().__init__(f"Execution Error: Failed at chunk {chunk_idx}. Root Cause: {original_exception}")

