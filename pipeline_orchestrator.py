import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from .components import BatchFlattener

class DualMCModelOrchestrator:
    """
    Coordinates data flow through the decoupled ML components.
    Guarantees 1:1 row output mapping via a streaming Parquet append strategy.
    """
    def __init__(
        self,
        config: 'InferenceConfig',
        base_preprocessor: 'DataPreprocessor',
        rule_engine: 'RuleEngine',
        retriever: 'CandidateRetriever',
        truncator: 'DynamicTruncatorProtocol',
        context_enricher: 'ContextEnricher',
        scorer: 'CrossEncoderScorer',
        post_processor: 'PredictionPostProcessor',
        logger: logging.Logger
    ):
        self.config = config
        self.preprocessor = base_preprocessor
        self.rule_engine = rule_engine
        self.retriever = retriever
        self.truncator = truncator
        self.context_enricher = context_enricher
        self.scorer = scorer
        self.post_processor = post_processor
        self.logger = logger

    def process_file(self, input_parquet_path: str, output_parquet_path: str):
        self.logger.info(f"Initiating streaming inference. Source: {input_parquet_path}")
        
        try:
            parquet_file = pq.ParquetFile(input_parquet_path)
        except Exception as e:
            self.logger.critical(f"Failed to open source file {input_parquet_path}: {e}")
            raise

        writer = None
        processed_count = 0

        for batch_idx, record_batch in enumerate(parquet_file.iter_batches(batch_size=self.config.io_chunk_size)):
            df_chunk = record_batch.to_pandas()
            try:
                results_df = self._execute_pipeline_step(df_chunk)
            except Exception as e:
                self.logger.error(f"Catastrophic failure in processing chunk {batch_idx}: {e}. Routing to dead-letter format.", exc_info=True)
                results_df = self._generate_error_dataframe(df_chunk, str(e))

            # Stream results to disk immediately to maintain horizontal memory profile
            try:
                table = pa.Table.from_pandas(results_df)
                if writer is None:
                    writer = pq.ParquetWriter(output_parquet_path, table.schema)
                writer.write_table(table)
            except Exception as e:
                self.logger.critical(f"Failed to write results to disk for chunk {batch_idx}: {e}")
                if writer: writer.close()
                raise
            
            processed_count += len(df_chunk)
            self.logger.info(f"Processed {processed_count} rows...")

        if writer: 
            writer.close()
            
        self.logger.info(f"Inference complete. Results safely persisted to {output_parquet_path}")

    def _execute_pipeline_step(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """Executes the standard pipeline logic for a single DataFrame chunk."""
        
        df_clean = self.preprocessor.process(df_chunk)
        df_resolved, df_unresolved = self.rule_engine.evaluate(df_clean)
        
        # Short-circuit if all rows in this chunk were solved by regex/rules
        if df_unresolved.empty: 
            return df_resolved

        queries = df_unresolved[self.config.input_col_name].tolist()
        row_ids = df_unresolved[self.config.key_column_name].tolist()
        
        raw_codes, raw_labels, raw_scores = self.retriever.search(queries, self.config.top_k_candidates)
        trunc_codes, trunc_labels = self.truncator.truncate(raw_codes, raw_labels, raw_scores)
        enriched_queries, enriched_cands = self.context_enricher.enrich(queries, trunc_codes, trunc_labels)
        
        flat_q, flat_c, flat_l, mapping = BatchFlattener.flatten(enriched_queries, trunc_codes, enriched_cands)
        
        flat_logits = self.scorer.score(
            queries=flat_q,
            candidates=[[lbl] for lbl in flat_l],
            feature_prefix=self.config.feature_prefix,
            logit_name=self.config.logit_name
        )
        
        reshaped_logits = BatchFlattener.unflatten_to_tensor(flat_logits, mapping, len(queries))
        
        df_ml_results = self.post_processor.resolve(
            row_ids, reshaped_logits, trunc_codes, self.config.simul_type, self.config.simul_type == 'both'
        )
        
        return pd.concat([df_resolved, df_ml_results], ignore_index=True)

    def _generate_error_dataframe(self, df_chunk: pd.DataFrame, error_msg: str) -> pd.DataFrame:
        """Fallback generator to ensure the pipeline doesn't drop rows if a chunk fails."""
        return pd.DataFrame({
            self.config.key_column_name: df_chunk[self.config.key_column_name],
            self.config.prediction_column_name: None,
            self.config.precision_column_name: 0.0,
            "error_reason": error_msg
        })