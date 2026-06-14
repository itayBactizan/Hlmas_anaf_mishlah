import os
import pickle
import numpy as np
import pandas as pd
import torch
import json
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple,Literal, Optional


class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4

    RETREIVER_MODEL_PATH = '/home/itayb@lamas.gov.il/my_code/models/retriver_dual'

    TRAIN_DATA = "/home/nfsdisk/Simul_AI/new_datas/processed_2018_2025/2018_2024_train_data.csv"
    VALIDATION_DATA = "/home/nfsdisk/Simul_AI/new_datas/processed_2018_2025/2018_2024_val_anaf_data.csv"
    TEST_DATA = "/home/nfsdisk/Simul_AI/new_datas/processed_2018_2025/2025_test_data.csv"
    SIMUL_TYPE= "anaf"
    VERSION_NUM="4"
    OUTPUT_PATH = f'/home/itayb@lamas.gov.il/my_code/models/xgb_reranker_{SIMUL_TYPE}_v{VERSION_NUM}'


    BATCH_SIZE = 32
    MAX_LENGTH = 512

    SKIP_TRAINING = True

    MODEL_NAME = f'xgb_ranker_model_for_{SIMUL_TYPE}.pkl'

os.makedirs(Config.OUTPUT_PATH, exist_ok=True)


class XgbFeatureEnricher:  
    """
     create class for update the original 
    """
    def __init__(self,
                numeric_features_df: pd.DataFrame,
                key_column: str,
                feature_prefix: str,
                median_shanotlimud_by_gil: pd.Series,
                cols_to_chg_categorial:Optional[List[str]]= None,
                cols_to_convert_numbers:Optional[List[str]]= None
                ):
        self.numeric_features_df=numeric_features_df
        self.key_column=key_column
        self.feature_prefix=feature_prefix
        self.candidates_col=f'{feature_prefix}_candidates_code'
        self.scores_col= f'{feature_prefix}_scores_normlize'
        self.cols_to_chg_categorial= cols_to_chg_categorial or ['MakorSachar','MenaheletMi']
        self.cols_to_convert_numbers=cols_to_convert_numbers or [f"{feature_prefix}_candidates_code",
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
        
        cols_to_convert= self.cols_to_convert_numbers
        df[cols_to_convert]=df[cols_to_convert].map(self._convert_text_to_number)
        # convert columns after update the values
        columns_convert_to_int=[c for c in ['TeudaGvoha','YeshuvAvoda','shnotlimud','Gil'] if c in df.columns]
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
                print(f"Warning :categoial column '{col}' not found in Dataframe, skipping")
                continue
            encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            column_encoded=encoder.fit_transform(df[[col]])
            df_columns_encoded=pd.DataFrame(column_encoded,
                  columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                    index=df.index)
            df= pd.concat([df,df_columns_encoded], axis=1)
            # drop orginal column
            df=df.drop(col, axis=1)
            print(f"categorial column '{col} updated to new Encoder")
        return df
    
    def _create_eduction_gap(self, df:pd.DataFrame) -> pd.DataFrame:
        if 'shnotlimud' not in df.columns or 'Gil' not in df.columns:
            missing=[c for c in ('shnotlimud', 'Gil') if c not in df.columns]
            raise RuntimeError(f" requierd columns {missing} missing from Dataframe, canot compute eduction_gap")
        # update series with the median shnotlimud
        series_meidan=df['Gil'].map(self.median_shanotlimud_by_gil)
        
        # create the new column education gap, if the feature are outliers set -50
        df['education_gap']=np.where((df['shnotlimud']>-1) & (df['shnotlimud']<26)& (df['Gil']>0),
                                    df['shnotlimud']-series_meidan,-50)
        return df
    
    def _filter_by_minus_1_in_score(self, df_from_retriver: pd.DataFrame):
        print(f'shape before filtering is {df_from_retriver.shape}')

        df_from_retriver=df_from_retriver[~df_from_retriver['sec_scores'].apply(lambda x: -1 in x)].copy()
        
        print(f'shape after filtering is {df_from_retriver.shape}')
    
        return df_from_retriver
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
            raise ValueError(f"XgbFeatureEnricher: requires bothe scores and row_ids")
        
        print(f"XgbFeatureEnricher: started to enrich numeric data")
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

def load_and_process_data():
    df_train = pd.read_csv(Config.TRAIN_DATA)
    df_val = pd.read_csv(Config.VALIDATION_DATA)
    df_test = pd.read_csv(Config.TEST_DATA)


    X_train = generate_embeddings(df_train_processed['free_text_description'], Config.EMBEDDING_MODEL_PATH, Config.BATCH_SIZE)
    X_val = generate_embeddings(df_val_preprocess['free_text_description'], Config.EMBEDDING_MODEL_PATH, Config.BATCH_SIZE)
    X_test = generate_embeddings(df_test_preprocess['free_text_description'], Config.EMBEDDING_MODEL_PATH, Config.BATCH_SIZE)

    y_train = le.transform(df_train_processed['SemelAnafSofi'])
    y_val = le.transform(df_val_preprocess['SemelAnafSofi'])
    y_test = le.transform(df_test_preprocess['SemelAnafSofi'])

    with open(os.path.join(Config.OUTPUT_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)



    return X_train, y_train, X_val, y_val, X_test, y_test, le

def train_model(X_train, y_train, X_val, y_val, label_encoder):
    weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    num_classes = len(label_encoder.classes_)

    train_classes = np.unique(y_train)
    all_classes = np.arange(len(label_encoder.classes_))
    missing = np.setdiff1d(all_classes, train_classes)
    missing_label = label_encoder.inverse_transform(missing)

    base_params = {
        'device': 'cuda',
        'tree_method': 'hist',
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'n_estimators': 2000,
        'num_class': num_classes,
        'verbosity': 1
    }

    param_grid = [
        {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'learning_rate': 0.1, 'max_depth': 8, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'learning_rate': 0.01, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.7}
    ]

    best_score = -1
    best_model = None
    best_params = None

    for i, params in enumerate(param_grid):
        print(f'\nTrainin Config {i+1}/{len(param_grid)}: {params}')

        current_params = {**base_params, **params}

        checkpoint_dir = os.path.join(Config.OUTPUT_PATH, f"checkpoints_cgf_{i}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        clf = xgb.XGBClassifier(
            **current_params,
            early_stopping_rounds=50,
            callbacks=[xgb.callback.TrainingCheckPoint(
                directory=checkpoint_dir,
                interval=100,
                name='model'
            )]
        )

        clf.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        val_preds = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"Config {i+1} Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_model = clf
            best_params = params

    print(f'\nBest Validation Accuracy: {best_score:.4f}')
    print(f'Best Parameters: {best_params}')

    return best_model




class XGBRerankerScorer:
    """implement the xgb ranker after get k candidates from retriver of reranker model
    """
    def __init__(self,
                 model,
                 feature_columns: List[str],
                 enricher: 'XgbFeatureEnricher'
                 ):
        self.model = model
        self.feature_columns=feature_columns
        self.enricher = enricher

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
            raise ValueError("XgbFeatureEnricher is empty")
        
        print(f"XGBRerankerScorer: started of getting scores for candidates")
        # fill feature columns
        critical_cols={self.enricher.candidates_col, self.enricher.scores_col}
        missing_cols=set(self.feature_columns) - set(df.columns)
        critical_missing= critical_cols - set(df.columns)
        non_critical_missing=missing_cols- critical_cols
        
        if critical_missing:
            raise RuntimeError(f"crtical columns are missing from the dataframe{critical_missing}")
        
        if non_critical_missing:
            print(f" Warning: the missing columns {str(missing_cols)} filled with 0")
            for col in non_critical_missing:
                df[col] = 0
        
        # get predict scores 
        X= df[self.feature_columns].astype(float)
        scores= self.model.predict(X)
        print(f"XGBRerankerScorer: get scores of {len(scores)} candidate rows")
        
        # warn when scores in first places are similar 
        self._check_tie_score( scores, df)


        return torch.tensor(scores, dtype=torch.float32)

if __name__ == "__main__":
    if Config.SKIP_TRAINING:
        if not os.path.exists(os.path.join(Config.OUTPUT_PATH, Config.MODEL_NAME)):
            raise FileNotFoundError(f'Model {Config.MODEL_NAME} not found in {Config.OUTPUT_PATH}. Please change SKIP_TRAINING=False in Config and start over.')
        else:
            with open(os.path.join(Config.OUTPUT_PATH, Config.MODEL_NAME)) as f:
                best_model=pickle.load(f)

            #best_model = xgb.XGBClassifier()
            #best_model.load_model(os.path.join(Config.OUTPUT_PATH, Config.MODEL_NAME))
            #best_model=xgb.Booster()
            #best_model.load_model(os.path.join(Config.OUTPUT_PATH, Config.MODEL_NAME))

        try:
            X_test = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'X_test.csv'))
            y_test = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'y_test.csv'))
        except FileNotFoundError as e:
            print(f'Test files (X_test.csv and y_test.csv) are not found in {Config.OUTPUT_PATH}.')
            exit(1)

        # itay changed
        # try:
        #      with open(os.path.join(Config.OUTPUT_PATH, 'label_encoder.pkl'), 'rb') as f:
        #         le = pickle.load(f)
        # except FileNotFoundError as e:
        #     print(f'Label encoder (le) are not found in {Config.OUTPUT_PATH}')
        #     exit(1)
    
    # not skipped the training
    else:
        if os.path.exists(os.path.join(Config.OUTPUT_PATH, 'X_train.csv')):
            print(f'Loading pre-compute embeddings from {Config.OUTPUT_PATH}')
            X_train = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'X_train.csv'))
            y_train = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'y_train.csv'))
            
            X_val = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'X_val.csv'))
            y_val = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'y_val.csv'))

            X_test = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'X_test.csv'))
            y_test = pd.read_csv(os.path.join(Config.OUTPUT_PATH, 'y_test.csv'))

            # with open(os.path.join(Config.OUTPUT_PATH, 'label_encoder.pkl'), 'rb') as f:
            #     le = pickle.load(f)

        else:
            X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_data()

        best_model = train_model(X_train, y_train, X_val, y_val)

        model_save_path = os.path.join(Config.OUTPUT_PATH, Config.MODEL_NAME)
        best_model.save_model(model_save_path)

    print('Running final evaluation on Test-set')
    #test_preds = best_model.predict(X_test)
    dtest=xgb.DMatrix(X_test)
    test_preds = best_model.predict(dtest).astype(int)
 

    test_preds_decoded = le.inverse_transform(test_preds)
    test_preds_decoded = test_preds_decoded.astype(str)

    y_test_decoded = le.inverse_transform(y_test)
    y_test_decoded = y_test_decoded.astype(str)

    statistic_per_class = classification_report(y_true=y_test_decoded, y_pred=test_preds_decoded, zero_division=0, output_dict=True)
    statistic_per_class.pop('accuracy', None)
    statistic_per_class.pop('macro avg', None)
    statistic_per_class.pop('weighted avg', None)
    statistic_per_class_final = {k: float(f"{v['precision']:.4f}") for k, v in statistic_per_class.items()}

    with open(os.path.join(Config.OUTPUT_PATH, 'percision_per_class_anaf_on_anaf_model.json'), 'w') as f:
        json.dump(statistic_per_class_final, f)
    
    with open(os.path.join(Config.OUTPUT_PATH,'xgb_model_for_anaf_on_fine_tuned_anaf.pkl'),'wb') as f:
        pickle.dump(best_model,f)

    pd.DataFrame(statistic_per_class).transpose().to_csv(os.path.join(Config.OUTPUT_PATH, 'anaf_v4_stats.csv'))

    print('Complete!')
