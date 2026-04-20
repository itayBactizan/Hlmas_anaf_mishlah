import json
import logging
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Tuple, Dict, Any
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from .exceptions import DataSchemaError, InferenceExecutionError

class BasePreprocessor:
    """Handles initial text cleaning and NaN imputation."""
    def __init__(self, text_col: str):
        self.text_col = text_col

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_col not in df.columns:
            raise DataSchemaError(self.text_col)
            
        df_clean = df.copy()
        # Ensure all inputs are strings and strip surrounding whitespace
        df_clean[self.text_col] = df_clean[self.text_col].fillna("").astype(str).str.strip()
        return df_clean

class ExactMatchRuleEngine:
    """Short-circuits the ML pipeline for known, exact string matches."""
    def __init__(self, input_col: str, id_col: str, pred_col: str, prec_col: str, rules_dict_path: str = None, logger: logging.Logger = None):
        self.input_col = input_col
        self.id_col = id_col
        self.pred_col = pred_col
        self.prec_col = prec_col
        self.logger = logger or logging.getLogger(__name__)
        self.rules = {}
        
        if rules_dict_path:
            try:
                with open(rules_dict_path, 'r', encoding='utf-8') as f:
                    self.rules = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.warning(f"Rule engine initialization failed ({e}). Proceeding with empty rule set.")

    def evaluate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.rules:
            # If no rules exist, return an empty resolved df and pass everything to unresolved
            empty_resolved = pd.DataFrame(columns=[self.id_col, self.pred_col, self.prec_col])
            return empty_resolved, df.copy()

        # Map inputs against the rules dictionary
        df['__rule_match'] = df[self.input_col].map(self.rules)
        resolved_mask = df['__rule_match'].notna()
        
        df_resolved = df[resolved_mask].copy()
        df_unresolved = df[~resolved_mask].copy()
        
        if not df_resolved.empty:
            df_resolved[self.pred_col] = df_resolved['__rule_match']
            df_resolved[self.prec_col] = 1.0 # Exact matches get 100% precision
            df_resolved = df_resolved[[self.id_col, self.pred_col, self.prec_col]]
        else:
            df_resolved = pd.DataFrame(columns=[self.id_col, self.pred_col, self.prec_col])
            
        df_unresolved = df_unresolved.drop(columns=['__rule_match'])
        return df_resolved, df_unresolved

class PyTorchExactRetriever:
    """Executes high-speed, exact inner-product vector search in memory."""
    def __init__(self, mapping_df: pd.DataFrame, embeddings_tensor: torch.Tensor, device: torch.device):
        if len(mapping_df) != embeddings_tensor.shape[0]:
            raise ValueError(f"Mapping rows ({len(mapping_df)}) do not match embedding rows ({embeddings_tensor.shape[0]})")
            
        self.reference_matrix = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1).to(device)
        self.codes = mapping_df['code'].tolist()
        self.labels = mapping_df['label'].tolist()
        self.device = device
        self.query_encoder = SentenceTransformer("imvladikon/alephbertgimmel-base-512", device=str(device))

    def search(self, queries: List[str], top_k: int) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        if not queries:
            return [], [], []
            
        try:
            query_embeddings = self.query_encoder.encode(queries, convert_to_tensor=True, show_progress_bar=False)
            query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1).to(self.device)

            similarities = torch.matmul(query_embeddings, self.reference_matrix.T)
            scores, indices = torch.topk(similarities, k=top_k, dim=1)
            
            scores_cpu = scores.cpu().numpy()
            indices_cpu = indices.cpu().numpy()
            
            batch_codes, batch_labels, batch_scores = [], [], []
            for i, row in enumerate(indices_cpu):
                batch_codes.append([self.codes[idx] for idx in row])
                batch_labels.append([self.labels[idx] for idx in row])
                batch_scores.append(scores_cpu[i].tolist())
                
            return batch_codes, batch_labels, batch_scores
        except Exception as e:
            raise InferenceExecutionError(chunk_idx=1, original_exception=e)

class DynamicTruncator:
    """Optimizes compute by shrinking candidate buckets when retrieval confidence is high."""
    def __init__(self, high_conf: float = 0.85, margin: float = 0.15, mid_conf: float = 0.70, clear_margin: float = 0.15, tight_margin: float = 0.03):
        self.high_conf = high_conf
        self.margin = margin
        self.mid_conf = mid_conf
        self.clear_margin = clear_margin
        self.tight_margin = tight_margin

    def truncate(self, codes: List[List[str]], labels: List[List[str]], scores: List[List[float]]) -> Tuple[List[List[str]], List[List[str]]]:
        trunc_codes, trunc_labels = [], []
        for row_codes, row_labels, row_scores in zip(codes, labels, scores):
            if not row_scores or len(row_scores) == 0:
                trunc_codes.append([]); trunc_labels.append([]); continue

            top_score = row_scores[0]
            k = 16 
            if len(row_scores) >= 3:
                score_1 = row_scores[1]
                score_2 = row_scores[2]
                score_3 = row_scores[3]

                if (score_1 - score_2) <= self.tight_margin and (score_2 - score_3) > self.clear_margin:
                    k = 2 
                
                elif top_score > self.high_conf and (score_1 - score_3) > self.clear_margin:
                    k = 4
                
                elif top_score > self.mid_conf:
                    k = 8 

            else:
                k = 2 

            trunc_codes.append(row_codes[:k])
            trunc_labels.append(row_labels[:k])
        return trunc_codes, trunc_labels

class DefinitionContextEnricher:
    """Fuses base survey text with external metadata definitions for the cross-encoder."""
    def __init__(self, dictionary_path: str = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.definitions = {}
        if dictionary_path:
            try:
                with open(dictionary_path, 'r', encoding='utf-8') as f:
                    self.definitions = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.warning(f"Context dictionary failed to load ({e}). Relying solely on candidate labels.")

    def enrich(self, queries: List[str], candidate_codes: List[List[str]], candidate_labels: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        enriched_candidates = []
        for row_codes, row_labels in zip(candidate_codes, candidate_labels):
            row_enriched = []
            for code, label in zip(row_codes, row_labels):
                definition = self.definitions.get(str(code), "")
                # Create a dense string representation. e.g., "Software Engineer: Develops and maintains..."
                fused_text = f"{label}: {definition}" if definition else label
                row_enriched.append(fused_text)
            enriched_candidates.append(row_enriched)
        return queries, enriched_candidates

class BatchFlattener:
    """Utility class to transform ragged array structures into compute-dense flat lists."""
    @staticmethod
    def flatten(queries: List[str], candidate_codes: List[List[str]], candidate_labels: List[List[str]]):
        flat_q, flat_c, flat_l, mapping = [], [], [], []
        for row_idx, (q, codes, labels) in enumerate(zip(queries, candidate_codes, candidate_labels)):
            for code, label in zip(codes, labels):
                flat_q.append(q)
                flat_c.append(code)
                flat_l.append(label)
                mapping.append(row_idx) # Track lineage for unflattening
        return flat_q, flat_c, flat_l, mapping

    @staticmethod
    def unflatten_to_tensor(flat_logits: torch.Tensor, row_mapping: List[int], num_original_rows: int) -> torch.Tensor:
        grouped_logits = [[] for _ in range(num_original_rows)]
        for logit, row_idx in zip(flat_logits.tolist(), row_mapping):
            grouped_logits[row_idx].append(logit)
            
        max_len = max((len(g) for g in grouped_logits), default=0)
        # Pad with negative infinity so argmax completely ignores these dummy positions
        padded_logits = [g + [-float('inf')] * (max_len - len(g)) for g in grouped_logits]
        return torch.tensor(padded_logits)

class CPUOptimizedDualMCScorer:
    """Handles thread-safe, micro-batched forward passes for cross-encoder models."""
    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizer, collator: Any, internal_batch_size: int = 32, num_threads: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.internal_batch_size = internal_batch_size
        torch.set_num_threads(num_threads)
        self.model.eval()

    def _prepare_features(self, queries: List[str], candidates: List[List[str]], prefix: str) -> List[Dict[str, Any]]:
        features = []
        for i, text in enumerate(queries):
            input_ids, attention_masks, token_type_ids = [], [], []
            for lbl in candidates[i]:
                enc = self.tokenizer(text, text_pair=lbl, truncation=True, max_length=128)
                input_ids.append(enc["input_ids"])
                attention_masks.append(enc["attention_mask"])
                if "token_type_ids" in enc: token_type_ids.append(enc["token_type_ids"])
                
            feature = {f"{prefix}_input_ids": input_ids, f"{prefix}_attention_mask": attention_masks}
            if token_type_ids: feature[f"{prefix}_token_type_ids"] = token_type_ids
            features.append(feature)
        return features

    @torch.inference_mode()
    def score(self, queries: List[str], candidates: List[List[str]], feature_prefix: str, logit_name: str) -> torch.Tensor:
        all_logits = []
        try:
            for i in range(0, len(queries), self.internal_batch_size):
                batch_q = queries[i : i + self.internal_batch_size]
                batch_c = candidates[i : i + self.internal_batch_size]
                
                feats = self._prepare_features(batch_q, batch_c, feature_prefix)
                collated = self.collator(feats)
                outputs = self.model(**collated)
                
                if logit_name not in outputs:
                    raise KeyError(f"Logit '{logit_name}' missing from model output. Available: {outputs.keys()}")
                    
                all_logits.append(outputs[logit_name])
                
            return torch.cat(all_logits, dim=0) if all_logits else torch.tensor([])
        except Exception as e:
            raise InferenceExecutionError(chunk_idx=1, original_exception=e)

class StaticPrecisionPostProcessor:
    """Translates model logits into business metrics via vectorized dataframe operations."""
    def __init__(self, precision_csv_path: str, prediction_col_base: str, precision_col_base: str, id_col_name: str, logger: logging.Logger = None):
        self.pred_col_base = prediction_col_base
        self.prec_col_base = precision_col_base
        self.id_col = id_col_name
        self.logger = logger or logging.getLogger(__name__)
        
        try:
            df = pd.read_csv(precision_csv_path)
            # Safely cast keys to strings to ensure matching against candidate codes
            self.precision_map = {str(row['code']): float(row['precision']) for _, row in df.iterrows()}
        except Exception as e:
            self.logger.error(f"Failed to load precision mapping from {precision_csv_path} ({e}). Precision will default to 0.0.")
            self.precision_map = {} 

    def resolve(self, row_ids: List[Any], logits: torch.Tensor, candidate_codes: List[List[str]], task_name: str, is_multi_task: bool) -> pd.DataFrame:
        if logits.numel() == 0:
            return pd.DataFrame(columns=[self.id_col, self.pred_col_base, self.prec_col_base])
            
        try:
            best_indices = torch.argmax(logits, dim=1).tolist()
            winning_codes = [candidate_codes[i][idx] for i, idx in enumerate(best_indices)]
            
            # Dynamic column naming prevents collision if evaluating both occupation and sector simultaneously
            pred_col = f"{self.pred_col_base}_{task_name}" if is_multi_task else self.pred_col_base
            prec_col = f"{self.prec_col_base}_{task_name}" if is_multi_task else self.prec_col_base
            
            batch_df = pd.DataFrame({self.id_col: row_ids, pred_col: winning_codes})
            batch_df[prec_col] = batch_df[pred_col].map(self.precision_map).fillna(0.0)
            return batch_df
        except IndexError as e:
            raise InferenceExecutionError(chunk_idx=1, original_exception=e)




class PassThroughRuleEngine:
    """
    A Null Object implementation of the RuleEngine Protocol.
    It resolves 0 rows and passes 100% of the data to the ML pipeline.
    """
    def __init__(self, id_col: str, pred_col: str, prec_col: str):
        self.id_col = id_col
        self.pred_col = pred_col
        self.prec_col = prec_col

    def evaluate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Return an empty DataFrame with the correct schema for the resolved path
        empty_resolved = pd.DataFrame(columns=[self.id_col, self.pred_col, self.prec_col])
        
        # Return the original DataFrame completely untouched for the unresolved (ML) path
        return empty_resolved, df.copy()


class PassThroughContextEnricher:
    """
    A Null Object implementation of the ContextEnricher Protocol.
    It performs no enrichment, passing the raw candidate labels directly to the Scorer.
    """
    def enrich(
        self, 
        queries: List[str], 
        candidate_codes: List[List[str]], 
        candidate_labels: List[List[str]]
    ) -> Tuple[List[str], List[List[str]]]:
        
        # Simply return the exact inputs without modifying them
        return queries, candidate_labels