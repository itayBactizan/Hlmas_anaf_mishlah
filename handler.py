import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer

#current_dir = os.path.dirname(os.path.abspath(__file__))
#src_path = os.path.abspath(os.path.join(current_dir, '../../../'))

from src.models.ReRanker.ReRanker_model_v1 import BertForDualMultipleChoice, DualMultipleChoiceCollator
from src.utils.logger import LoggerBuilder

from .config import InferenceConfig
from .pipeline_orchestrator import DualMCModelOrchestrator
from .exceptions import PipelineInitializationError
from .components import (
    BasePreprocessor, ExactMatchRuleEngine, PyTorchExactRetriever, 
    DynamicTruncator, DefinitionContextEnricher, CPUOptimizedDualMCScorer, 
    StaticPrecisionPostProcessor, PassThroughContextEnricher, PassThroughRuleEngine
)



class DualMCModelHandler:
    """
    Facade class designed to interface with the external pipeline orchestrator.
    Handles the heavy lifting of model instantiation and dependency injection.
    """
    def __init__(
        self,
        model_id: str,
        preprocessed_data_name: str,
        temp_dir: str,
        device: str, 
        logs_dir: str,
        key_column_name: str,
        prediction_column_name: str,
        precision_column_name: str,
        file_suffix: str, 
        simul_type: str = 'both',
        num_threads: int = 4,
        precision: str = "int8",
        use_chunked: bool = True,
        **kwargs
    ):
        try:
            self.config = InferenceConfig(
                model_id=model_id, temp_dir=temp_dir, logs_dir=logs_dir,
                key_column_name=key_column_name, prediction_column_name=prediction_column_name,
                precision_column_name=precision_column_name, file_suffix=file_suffix,
                simul_type=simul_type, precision=precision, use_chunked=use_chunked
            )
        except Exception as e:
            raise PipelineInitializationError(f"Invalid configuration parameters provided: {e}")
            
        self.input_filename = preprocessed_data_name
        self.logger = LoggerBuilder.get_logger(f'{os.path.basename(model_id)}_{simul_type}', out_path=logs_dir)
        self.logger.info("Initializing high-throughput CPU-optimized pipeline adapter.")
        
        self._initialize_pipeline(num_threads)

    def _initialize_pipeline(self, num_threads: int):
        """
        Loads models and instantiates the internal orchestrator using Dependency Injection.
        Automatically falls back to 'Pass-Through' components if rules or dictionaries are missing.
        """
        try:
            self.logger.info("Initializing Tokenizer and Quantized Model...")
            tokenizer = AutoTokenizer.from_pretrained("/home/nfsdisk/lm/alephbertgimmel-base-512/")
            
            # Load the base BERT model
            raw_model = BertForDualMultipleChoice.from_pretrained(self.config.model_id)
            
            # CPU Optimization: Apply INT8 dynamic quantization if specified
            if self.config.precision == "int8":
                self.logger.info("Applying INT8 Dynamic Quantization for CPU...")
                quantized_model = torch.quantization.quantize_dynamic(
                    raw_model, {torch.nn.Linear}, dtype=torch.qint8
                )
            else:
                quantized_model = raw_model
                
            collator = DualMultipleChoiceCollator(tokenizer=tokenizer)

            # 1. Resolve Data Paths (Handles both Anaf and Mishlah variants)
            task_tag = "mishlah" if "mishlah" in self.config.simul_type else "anaf"
            dict_path = f"{self.config.model_id}/dic{task_tag.capitalize()}.csv"
            embed_path = f"{self.config.model_id}/{task_tag}_embeddings.pt"
            metrics_path = f"{self.config.model_id}/metrics_{task_tag}.csv"
            rules_path = f"{self.config.model_id}/exact_matches.json"
            defs_path = f"{self.config.model_id}/definitions.json"

            # 2. Initialize Retriever (Core Dependency)
            try:
                reference_df = pd.read_csv(dict_path)
                reference_embeddings = torch.load(embed_path, weights_only=True)
            except FileNotFoundError as e:
                self.logger.critical(f"Missing critical reference file: {e}")
                raise PipelineInitializationError(f"Could not find vector search assets: {e}")

            # 3. Dynamic Component Selection (The "Skip" Logic)
            
            # --- Rule Engine ---
            if os.path.exists(rules_path):
                self.logger.info(f"Rules found at {rules_path}. Activating ExactMatchRuleEngine.")
                rule_engine = ExactMatchRuleEngine(
                    input_col=self.config.input_col_name,
                    id_col=self.config.key_column_name,
                    pred_col=self.config.prediction_column_name,
                    prec_col=self.config.precision_column_name,
                    rules_dict_path=rules_path,
                    logger=self.logger
                )
            else:
                self.logger.info("No rules found. Activating PassThroughRuleEngine.")
                rule_engine = PassThroughRuleEngine(
                    id_col=self.config.key_column_name,
                    pred_col=self.config.prediction_column_name,
                    prec_col=self.config.precision_column_name
                )

            # --- Context Enricher ---
            if os.path.exists(defs_path):
                self.logger.info(f"Definitions found at {defs_path}. Activating DefinitionContextEnricher.")
                context_enricher = DefinitionContextEnricher(
                    dictionary_path=defs_path,
                    logger=self.logger
                )
            else:
                self.logger.info("No definitions found. Activating PassThroughContextEnricher.")
                context_enricher = PassThroughContextEnricher()

            # 4. Assemble the Full Pipeline
            self.internal_orchestrator = DualMCModelOrchestrator(
                config=self.config,
                base_preprocessor=BasePreprocessor(text_col=self.config.input_col_name),
                rule_engine=rule_engine,
                retriever=PyTorchExactRetriever(
                    mapping_df=reference_df, 
                    embeddings_tensor=reference_embeddings, 
                    device=self.config.device
                ),
                truncator=DynamicTruncator(),
                context_enricher=context_enricher,
                scorer=CPUOptimizedDualMCScorer(
                    model=quantized_model,
                    tokenizer=tokenizer,
                    collator=collator,
                    num_threads=num_threads
                ),
                post_processor=StaticPrecisionPostProcessor(
                    precision_csv_path=metrics_path,
                    prediction_col_base=self.config.prediction_column_name,
                    precision_col_base=self.config.precision_column_name,
                    id_col_name=self.config.key_column_name,
                    logger=self.logger
                ),
                logger=self.logger
            )
            self.logger.info("Pipeline assembly complete.")

        except Exception as e:
            self.logger.critical(f"Catastrophic initialization failure: {e}", exc_info=True)
            raise PipelineInitializationError(f"System failed to boot: {e}")

    def process_and_predict(self, **kwargs) -> str:
        """
        Primary execution method triggered by the external system.
        Returns:
            str: Absolute path to the generated Parquet file containing predictions.
        """
        input_file = kwargs.get('input_filename', self.input_filename)
        input_path = os.path.join(self.config.temp_dir, input_file)
        
        output_filename = f"{os.path.basename(self.config.model_id)}_{self.config.file_suffix}"
        output_path = os.path.join(self.config.temp_dir, output_filename)

        try:
            self.internal_orchestrator.process_file(input_parquet_path=input_path, output_parquet_path=output_path)
        except Exception as e:
            self.logger.error(f"process_and_predict aborted due to unrecoverable error: {e}")
            raise

        return output_path