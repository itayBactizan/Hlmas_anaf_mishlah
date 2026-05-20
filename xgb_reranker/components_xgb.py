import json
import logging
import re
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from .exceptions_xgb import DataSchemaError, InferenceExecutionError
from .utils_xgb import _serialize_data_vectorized

class BasePreprocessor:
    """Handles initial text cleaning and NaN imputation."""
    def __init__(self, text_col: str):
        self.text_col = text_col

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        # Ensure all inputs are strings and strip surrounding whitespace
        df_clean[self.text_col] = _serialize_data_vectorized(df_clean, "/home/nfsdisk1/Simul_AI/other_data_Amir/yishuv_dict.csv")
        if self.text_col not in df_clean.columns:
            raise DataSchemaError(self.text_col)
        return df_clean

class ExactMatchRuleEngine:
    """Short-circuits the ML pipeline for known, exact string matches."""
    def __init__(self, input_col: str, id_col: str, pred_col: str, prec_col: str, rules_dict_path: str = None, top_k: int = 3, logger: logging.Logger = None):
        self.input_col = input_col
        self.id_col = id_col
        self.pred_col = pred_col
        self.prec_col = prec_col
        self.logger = logger or logging.getLogger(__name__)
        self.rules = {}
        self.top_k = top_k

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
            df_resolved[self.pred_col] = df_resolved['__rule_match'].apply(lambda code: [code] + [None] * (self.top_k - 1))
            df_resolved[self.prec_col] = df_resolved['__rule_match'].apply(lambda _: [1.0] + [0.0] * (self.top_k - 1)) # Exact matches get 100% precision
            df_resolved = df_resolved[[self.id_col, self.pred_col, self.prec_col]]
        else:
            df_resolved = pd.DataFrame(columns=[self.id_col, self.pred_col, self.prec_col])
            
        df_unresolved = df_unresolved.drop(columns=['__rule_match'])
        return df_resolved, df_unresolved

class PyTorchExactRetriever:
    """Executes high-speed, exact inner-product vector search in memory."""
    def __init__(self, mapping_df: pd.DataFrame, embeddings_tensor: Optional[torch.Tensor] , device: torch.device, model_path: str, dict_path: Optional[str]=None):

        if embeddings_tensor and len(mapping_df) != embeddings_tensor.shape[0]:
            raise ValueError(f"Mapping rows ({len(mapping_df)}) do not match embedding rows ({embeddings_tensor.shape[0]})")
        
        self.query_encoder = SentenceTransformer(model_path, device=str(device))
        if not embeddings_tensor:
            corpus_df = pd.read_csv(dict_path, encoding='utf-8-sig')
            if "anaf" in model_path.lower():
                code_col, label_col =  'SemelAnaf','ShemAnaf'  
            else:  
                code_col, label_col =  'SemelMishlachYad','ShemMishlachYad'

            corp = corpus_df[label_col]
            embeddings_tensor = self.query_encoder.encode(corp, convert_to_tensor=True, show_progress_bar=False)  

        self.reference_matrix = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1).to(device)
        self.codes = mapping_df[code_col].tolist()
        self.labels = mapping_df[label_col].tolist()
        self.device = device
        

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
                
            self.logger.info(f"retriver completed,  number of codes are {len(batch_codes)} ")
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

    def enrich(self, queries: List[str], candidate_codes: List[List[str]],
                candidate_labels: List[List[str]], scores: List[List[float]] = None,
                  row_ids: List[List[Any]] = None) -> Tuple[List[str], List[List[str]]]:
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
                flat_q.append(str(q))
                flat_c.append(str(code))
                flat_l.append(str(label))
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
    def __init__(self, precision_csv_path: str, prediction_col_base: str, precision_col_base: str, id_col_name: str, top_k_output: int = 3, logger: logging.Logger = None):
        self.pred_col_base = prediction_col_base
        self.prec_col_base = precision_col_base
        self.id_col = id_col_name
        self.logger = logger or logging.getLogger(__name__)
        self.top_k = top_k_output
        
        try:
            df = pd.read_csv(precision_csv_path)
            #df['code']= df['code'].str.lstrip('0').replace('', '0')
            # Safely cast keys to strings to ensure matching against candidate codes
            self.precision_map = {str(row['code']): float(row['precision']) for _, row in df.iterrows()}
        except Exception as e:
            self.logger.error(f"Failed to load precision mapping from {precision_csv_path} ({e}). Precision will default to 0.0.")
            self.precision_map = {} 

    def resolve(self, row_ids: List[Any], logits: torch.Tensor, candidate_codes: List[List[str]], task_name: str, is_multi_task: bool) -> pd.DataFrame:
        
        # Dynamic column naming prevents collision if evaluating both occupation and sector simultaneously
        pred_col = self.pred_col_base #f"{self.pred_col_base}_{task_name}" 
        prec_col = self.prec_col_base #f"{self.prec_col_base}_{task_name}" 

        if logits.numel() == 0:
            return pd.DataFrame(columns=[self.id_col, pred_col, prec_col])
            
        try:
            k = min(self.top_k, logits.shape[1])
            _, best_indices = torch.topk(logits, k=k, dim=1)
            best_indices = best_indices.tolist()

            rows = []
            for i, indxs in enumerate(best_indices):
                codes_list = []
                prec_list = []
                for idx in indxs:
                    if logits[i, idx] == -float('inf'):
                        codes_list.append(None)
                        prec_list.append(0.0)
                    else:
                        code = candidate_codes[i][idx]
                        codes_list.append(code)
                        prec_list.append(self.precision_map.get(str(code), 0.0))
                
                while len(codes_list) < self.top_k:
                    codes_list.append(None)
                    prec_list.append(0.0)

                rows.append({
                    self.id_col: row_ids[i],
                    pred_col: codes_list,
                    prec_col: prec_list,
                    })
            return pd.DataFrame(rows)
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

class PassThroughTruncator:
    """ A Null Object implementation of the Truncator Protocol """
    def truncate(self, codes: List[List[str]], labels: List[List[str]], scores: List[List[float]]) -> Tuple[List[List[str]], List[List[str]]]:
        # retrun the same codes and lables without truncate it 
        return codes, labels


class PassThroughContextEnricher:
    """
    A Null Object implementation of the ContextEnricher Protocol.
    It performs no enrichment, passing the raw candidate labels directly to the Scorer.
    """
    def enrich(
        self, 
        queries: List[str], 
        candidate_codes: List[List[str]], 
        candidate_labels: List[List[str]],
        scores: List[List[float]] = None,
        row_ids: List[List[Any]] = None
    ) -> Tuple[List[str], List[List[str]]]:
        
        # Simply return the exact inputs without modifying them
        return queries, candidate_labels

class XgbFeatureEnricher:  
    """
     
    """
    def __init__(self,
                numeric_features_df: pd.DataFrame,
                key_column: str,
                feature_prefix: str,
                median_shanotlimud_by_gil: pd.Series,
                cols_to_chg_categorial:Optional[List[str]]= None,
                cols_to_convert_int:Optional[List[str]]= None,
                logger: logging.Logger = None
                ):
        self.numeric_features_df=numeric_features_df
        self.key_column=key_column
        self.feature_prefix=feature_prefix
        self.candidates_col=f'{feature_prefix}_candidates_code'
        self.scores_col= f'{feature_prefix}_scores_normlize'
        self.cols_to_chg_categorial= cols_to_chg_categorial or ['MakorSachar','MenaheletMi']
        self.cols_to_convert_int=cols_to_convert_int or [f"{feature_prefix}_candidates_code",
                                                        "YeshuvAvoda",
                                                        "MaamadAvoda",
                                                        "TeudaGvoha",
                                                        "shnotlimud",
                                                        "Gil",
                                                        f"{feature_prefix}_scores_normlize",
                                                        f"{feature_prefix}_candidates_code_1",
                                                        f"{feature_prefix}_candidates_code_2",
                                                        f"{feature_prefix}_candidates_code_3",
                                                        "MakorSachar",
                                                        "MenaheletMi",
                                                        ]
        self.median_shanotlimud_by_gil= median_shanotlimud_by_gil
        self.logger = logger or logging.getLogger(__name__)
        self._cached_feature_df:pd.DataFrame = pd.DataFrame()

    # =================================================== #
    #            feature enineering helpers               #
    # =================================================== #

    def _normlize_similarity(self, similarity_list:List[float]) ->List[float] :
        """
        normlize list of similarity scores
        """
        #intilize standardScalar
        similarity_array=np.array(similarity_list, dtype=float)
        
        #transform similarity to normlize by the formula (x-mean)/sd
        mean=np.mean(similarity_array)
        std_dev=np.std(similarity_array)
        normized_similarity=(similarity_array-mean)/std_dev if std_dev > 0 else  (similarity_array-mean)

        normized_similarity=[round(num,5) for num in normized_similarity]
        
        return normized_similarity
    
    def _split_target_to_four_columns(self, df: pd.DataFrame,col: str):
        """ split the code of 4 digit to seperate 4 columns of agg digits like the second column contains two digits """
        temp=df[col].astype(str).str.strip()

        if (temp.str.len()!=4).any():
            raise ValueError("target value does not equal to 4 digits")
        padded=temp.str.pad(4, side='right', fillchar=" ")

        # split into charachter
        new_cols=padded.apply(lambda x: list(x))
        df[f"{col}_1"]=new_cols.apply(lambda x: x[0])
        df[f"{col}_2"]=new_cols.apply(lambda x: x[0]+x[1])
        df[f"{col}_3"]=new_cols.apply(lambda x: x[0]+x[1]+x[2])
        return df
    
    def _convert_text_to_number(self, val):
        # update text of לא ידוע
        if pd.isna(val) or val=="לא ידוע" :
            return -100
        elif val=="XXXX" :
            return -200
        elif val=="X":
            return -300
        elif val=="XX":
            return -400
        elif val=="XXX":
            return -400
        
        # if it's a number leave it as is
        elif isinstance(val, (int, float)):
            return val
        
        elif isinstance(val, str) and re.fullmatch(r"\d+XXX", val):
            number_part=int(val.replace("XXX",""))
            return number_part*-1000
        
        elif isinstance(val, str) and re.fullmatch(r"\d+XX", val):
            number_part=int(val.replace("XX",""))
            return number_part*-100
        
        elif isinstance(val, str) and re.fullmatch(r"\d+X", val):
            number_part=int(val.replace("X",""))
            return number_part*-10
        else:
            # try to convert text to number
            try:
                if isinstance(val,str):
                    val=val.strip()
                    f_val=float(val)
                if f_val.is_integer():
                    return int(f_val)  
                else:
                    return f_val
            except Exception:
                raise ValueError (f"cannot convert value '{val}' to number")
            
    def _convert_text_df(self, df: pd.DataFrame) -> pd.DataFrame:
        
        cols_to_convert= self.cols_to_convert_int
        df[cols_to_convert]=df[cols_to_convert].map(self._convert_text_to_number)
        # convert columns after update the values
        columns_convert_to_int=[c for c in ['TeudaGvoha','YeshuvAvoda','shnotlimud','gil'] if c in df.columns]
        try:
            tmp=df[columns_convert_to_int].apply(pd.to_numeric, errors='raise')
            if not  ((tmp % 1 )==0).all().all():
                raise ValueError("Non-integer values {}")
            df[columns_convert_to_int]=df[columns_convert_to_int].astype("Int64")
        except Exception as e:
            print("error  when convert to numeric is ",e)
            raise
        return df
    
    def _update_categorial_by_onehotencoder(self, df: pd.DataFrame, cols_to_chg :List[str]) -> pd.DataFrame:
        for col in cols_to_chg:
            if col not in df.columns:
                self.logger.warning(f"categoial column '{col}' not found in Dataframe, skipping")
                continue
            encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            column_encoded=encoder.fit_transform(df[[col]])
            df_columns_encoded=pd.DataFrame(column_encoded,
                  columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                    index=df.index)
            df= pd.concat([df,df_columns_encoded], axis=1)
            # drop orginal column
            df=df.drop(col, axis=1)
            self.logger.info(f"categorial column '{col} updated to new Encoder")
        return df
    
    def _create_eduction_gap(self, df:pd.DataFrame) -> pd.DataFrame:
        if 'shnotlimud' not in df.columns or 'Gil' not in df.columns:
            missing=[c for c in ('shnotlimud', 'Gil') if c not in df.columns]
            self.logger.error(f" requierd columns {missing} missing from Dataframe, canot compute eduction_gap")
            raise DataSchemaError(str(missing))
        # update series with the median shnotlimud
        series_meidan=df['Gil'].map(self.median_shanotlimud_by_gil)
        
        # create the new column education gap, if the feature are outliers set -50
        df['education_gap']=np.where((df['shnotlimud']>-1) & (df['shnotlimud']<26)& (df['Gil']>0),
                                    df['shnotlimud']-series_meidan,-50)
        return df
    # =================================================== #
    #            Protocol method                          #
    # =================================================== #

    def enrich(self,
               queries: List[str],
               candidate_codes: List[List[str]],
               candidate_labels: List[List[str]],
               scores: List[List[float]],
               row_ids: List[List[Any]] ) -> Tuple[List[str], List[List[str]]]:
        if scores is None or row_ids is None: 
            self.logger.error(f"XgbFeatureEnricher: requierd inputs of scores and row _ids cannot prepare feature cache")
            raise ValueError(f"XgbFeatureEnricher: requires bothe scores and row_ids")
        
        normlized_scores= [self._normlize_similarity(s) for s in scores]

        df=pd.DataFrame({
            'id': row_ids, # worked in iim brancg wheb batch is small the 10k :self.numeric_features_df[self.key_column]
            self.candidates_col: candidate_codes,
            self.scores_col: normlized_scores,
        })

        # one row per candidate
        df=df.explode([self.candidates_col, self.scores_col], ignore_index=True)

        df=df.merge(
            self.numeric_features_df,
            left_on='id',
            right_on= self.key_column,
            how='left',
            suffixes=('','_dupp'),
        )

        df=self._split_target_to_four_columns(df, self.candidates_col)
        df= self._convert_text_df(df)
        df=self._update_categorial_by_onehotencoder(df, self.cols_to_chg_categorial)
        df= self._create_eduction_gap(df)

        self._cached_feature_df=df
        self.logger.info(f"XgbFeatureEnricher: prepared {len(df)}  candidates rows for xgb ranker")
        return queries, candidate_labels

class XGBRerankerScorer:
    """implement the xgb ranker after get k candidates from retriver of reranker model
    """
    def __init__(self,
                 model,
                 feature_columns: List[str],
                 enricher: 'XgbFeatureEnricher',
                 logger: logging.Logger =None):
        self.model = model
        self.feature_columns=feature_columns
        self.enricher = enricher
        self.logger = logger or logging.getLogger(__name__)
    def _check_tie_score(self, scores: np.ndarray, df: pd.DataFrame):
        """
        test if there are two candidate with similar score
        """
        scores_rounded= np.round(scores,2)
        df_tie_check=pd.DataFrame({'id':df[self.enricher.key_column].values, 'score_rounded': scores_rounded })
        queries_with_ties = (
            df_tie_check.groupby('id')['score_rounded'].apply(lambda x: (x == x.max()).sum() >1).sum()
        )
        if queries_with_ties> 0:
            self.logger.warning(f"XGBRerankerScorer: {queries_with_ties} queries have a tied top score so argmax tie breaking is arbitrary"
            )

    def score(self,
              queries: List[str],
              candidates: List[List[str]],
              feature_prefix: str,
              logit_name: str,) ->torch.Tensor:
        df=self.enricher._cached_feature_df

        if df.empty:
            raise InferenceExecutionError(chunk_idx=1,
                                          original_exception=ValueError("XgbFeatureEnricher is empty"),)
        
        # fill feature columns
        critical_cols={self.enricher.candidates_col, self.enricher.scores_col}
        missing_cols=set(self.feature_columns) - set(df.columns)
        critical_missing= critical_cols - set(df.columns)
        non_critical_missing=missing_cols- critical_cols
        
        if critical_missing:
            self.logger.error("crtical columns are missing from the dataframe")
            raise DataSchemaError(str(critical_missing))
        
        if non_critical_missing:
            self.logger.warning(f"the missing columns {str(missing_cols)} filled with 0")
            for col in missing_cols:
                df[col] = 0
        
        # get predict scores 
        X= df[self.feature_columns].astype(float)
        scores= self.model.predict(X)
        self.logger.info(f"scored {len(scores)} candidate rows")
        
        # warn when scores in first places are similar 
        self._check_tie_score( scores, df)


        return torch.tensor(scores, dtype=torch.float32)