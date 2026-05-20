import os
import sys
import torch
import pickle
import json
import pandas as pd
from transformers import AutoTokenizer

#current_dir = os.path.dirname(os.path.abspath(__file__))
#src_path = os.path.abspath(os.path.join(current_dir, '../../../'))

from ...ReRanker.ReRanker_model_v1 import BertForDualMultipleChoice, DualMultipleChoiceCollator
from utils.logger import LoggerBuilder

from .config_xgb import InferenceConfig
from .pipeline_orchestrator_xgb import DualMCModelOrchestrator
from .exceptions_xgb import PipelineInitializationError
from .components_xgb import (
    BasePreprocessor, ExactMatchRuleEngine, PyTorchExactRetriever, 
    DynamicTruncator, PassThroughTruncator, DefinitionContextEnricher, CPUOptimizedDualMCScorer, 
    StaticPrecisionPostProcessor, PassThroughContextEnricher, PassThroughRuleEngine,  XgbFeatureEnricher, XGBRerankerScorer
)



class XGBRerankerHandler:
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
        cols_to_chg_categorial: list= ['MakorSachar','MenaheletMi'],
        key_column_reranker: str= 'mezaheReshumaRatz', # in the amir code is 'mezaheReshumaRaz'
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
        self.cols_to_chg_categorial = cols_to_chg_categorial
        self.key_column_reranker=key_column_reranker
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
            retriver_path = f"{self.config.model_id}/retriver_dual"
            prefix_tag = "occ" if task_tag=="mishlah" else "sec"
            xgb_model_path=f"{self.config.model_id}/xgb_ranker_model_for_{task_tag}.pkl"
            feat_cols_path= f"{self.config.model_id}/feature_columns_{task_tag}.json"
            median_path = f"{self.config.model_id}/median_shnotlimud_by_gil.json"

            # 2. Initialize Retriever (Core Dependency)
            try:
                reference_df = pd.read_csv(dict_path)
                reference_embeddings = torch.load(embed_path, weights_only=True) if os.path.exists(embed_path) else None
            except FileNotFoundError as e:
                self.logger.critical(f"Missing critical reference file: {e}")
                raise PipelineInitializationError(f"Could not find vector search assets: {e}")

            # load XGB model, feature columns and median eduction by age lookup
            try:
                with open(xgb_model_path, 'rb') as f:
                    xgb_model=pickle.load(f)
            except FileNotFoundError as e:
                raise PipelineInitializationError(f"XGb model file not found: {e}")

            try: 
                with open(feat_cols_path, 'r') as f:
                    feature_columns=json.load(f)
            except FileNotFoundError as e:
                raise PipelineInitializationError(f"Feature columns file not found {e}")

            try: 
                with open(median_path, 'r') as f:
                    map_shnotlimud=json.load(f)
                    map_shnotlimud={int(k): v for k,v in map_shnotlimud.items()}
                    median_shnotlimud_by_gil=pd.Series(map_shnotlimud)
            except FileNotFoundError as e:
                raise PipelineInitializationError(f"Median Shanotlimud file not found {e}")

            try: 
                numeric_feature_df=pd.read_parquet(
                    os.path.join(self.config.temp_dir, self.input_filename)
                )
            except FileNotFoundError as e :
                raise PipelineInitializationError(f"can't read input file in {self.input_filename}")

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
            self.logger.info(f"Activating XgbFeatureEnricher.")
            context_enricher = XgbFeatureEnricher(
                numeric_features_df=numeric_feature_df,
                key_column=self.key_column_reranker,
                feature_prefix=prefix_tag,
                median_shanotlimud_by_gil=median_shnotlimud_by_gil,
                cols_to_chg_categorial=self.cols_to_chg_categorial,
                logger=self.logger,
            )

            # -- xgb scorer
            scorer= XGBRerankerScorer(
                model=xgb_model,
                feature_columns=feature_columns,
                enricher=context_enricher,
                logger=self.logger,
            )

            # 4. Assemble the Full Pipeline
            self.internal_orchestrator = DualMCModelOrchestrator(
                config=self.config,
                base_preprocessor=BasePreprocessor(text_col=self.config.input_col_name),
                rule_engine=rule_engine,
                retriever=PyTorchExactRetriever(
                    mapping_df=reference_df, 
                    embeddings_tensor=reference_embeddings, 
                    device=self.config.device,
                    model_path=retriver_path,
                    dict_path=dict_path if not reference_embeddings else None
                ),
                truncator=PassThroughTruncator(),
                context_enricher=context_enricher,
                scorer=scorer,
                post_processor=StaticPrecisionPostProcessor(
                    precision_csv_path=metrics_path,
                    prediction_col_base=self.config.prediction_column_name,
                    precision_col_base=self.config.precision_column_name,
                    id_col_name=self.config.key_column_name,
                    top_k_output=1,
                    logger=self.logger
                ),
                logger=self.logger
            )
            self.logger.info("XGB Pipeline assembly complete.")

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

        return output_filename