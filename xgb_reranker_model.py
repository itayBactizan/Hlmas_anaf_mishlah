# =================================================== #
#                      Imports                        #
# =================================================== #
from typing import List, Dict, Any, Tuple,Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import re
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from sentence_transformers.evaluation import SentenceEvaluator
import pandas as pd
from datasets import load_from_disk
import os
from joblib import Parallel, delayed
import joblib
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker, plot_importance
import time
import matplotlib.pyplot as plt
from datetime import datetime
import json
import shap
import yaml
from scipy.stats import entropy
import scipy.stats as stats
import random
import logging

COLUMNS_IN_MODEL=['ShnatSeker',
       'ChodeshSeker', 'YeshuvAvoda',
       'MaamadAvoda',  'SamlanNochechi',
       'SemelAnafSofi', 'ShemAvoda', 'SugAvoda', 'ShemMachlaka', 'SugMachlaka',
       'TeurAnafMale', 'SemelMishlahSofi', 'TeudaGvoha', 
       'TeudaGvohaAcher', 'MakorSachar',  'EzoAvoda',
       'TeurPeula', 'TeurTafkid', 'TeurMishlahMale', 'shnotlimud',
       'TaarichSimul',  'gil', 'MenahelEtMi']
class XGBRerankerClassifier:
    def __init__(
            self,
            file_suffix: str,
            model_id: str, 
            temp_dir: str,
            prediction_column_name: str,
            precision_column_name: str,
            key_column_name: str,
            simul_type: str,
            preprocessed_data_name: str,
            path_retriver: str,
            logger: logging.Logger = None
                 ):
        init_time = datetime.now()

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.logger.info("Seccessfuly cretaed logger")
        self.file_suffix = file_suffix
        self.output_file_name = f"{os.path.basename(model_id)}_{file_suffix}"
        self.temp_dir=temp_dir
        self.data_input_for_inference = pd.read_parquet(os.path.join(self.temp_dir, preprocessed_data_name))
        dataset_from_retriver=load_from_disk(path_retriver)
        self.retriever_data=dataset_from_retriver.to_pandas()
        self.predictions_column_name = prediction_column_name
        self.precision_class_column_name = precision_column_name
        
        self.key_column = key_column_name
        self.anaf_or_mishlah = simul_type
        self.model, self.model_parameters=self.upload_xgb_model()
        self.occ_or_sec="sec" if self.anaf_or_mishlah=="anaf" else "occ"
        self.score_normlize_column_name= f'{self.occ_or_sec}_scores'
        self.candidates_column_name= f'{self.occ_or_sec}_candidates_code'
        self.rank_column_name=f'rank_{self.occ_or_sec}'
        self.columns_to_use=[ f'{self.occ_or_sec}_candidates_code', 'YeshuvAvoda','MaamadAvoda','TeudaGvoha',
                                  'MakorSachar','shnotlimud', 'gil','MenahelEtMi','id',f'rank_{self.occ_or_sec}',
                                    f'{self.occ_or_sec}_scores_normlize']
        self.cols_to_chg_categorial=['MakorSachar','MenahelEtMi']
        self.mishlah_dict_label_to_code=dict()
        self.anaf_dict_label_to_code= dict()

        self.meidan_shnotlimud_by_gil=pd.DataFrame() #need to add path to file in models folder
        with open(os.path.join(model_id,f'precision_per_class_for_xgb_ reranker_model_on_{simul_type}.json'), 'r') as f:
            self.percision_per_class = json.load(f) 

    # Returning XGB model 
    def upload_xgb_model(self):

        model_name = f'xgb_reranker_model_for_{self.anaf_or_mishlah}.pkl'


        model_path = os.path.join(self.models_location,model_name)

        with open(model_path,'rb') as f:
            model = pickle.load(f)

        # get models hyperparameters
        with open (os.path.join(os.getcwd(),path,"xgb_ranker.yaml")) as f:
            cfg= yaml.safe_load(f)
        model_parameters= {**cfg["model"]["params"]}

        return model,model_parameters
    

    
    # =================================================== #
    #            update scores by retriver                #
    # =================================================== #

    def _normlize_similarity(self, similarity_list:List[float]) ->List[float] :
        """
        normlize list of similarity
        """
        #intilize standardScalar
        similarity_array=np.array(similarity_list)
        
        #transform similarity to normlize by the formula (x-mean)/sd
        mean=np.mean(similarity_array)
        std_dev=np.std(similarity_array)
        normized_similarity=(similarity_array-mean)/std_dev

        normized_similarity=[round(num,5) for num in normized_similarity]
        
        return normized_similarity

    def normlize_columns(self, df: pd.DataFrame ):
        df[self.score_normlize_column_name+'_normlize']=Parallel(n_jobs=-2,
                                                                  backend="threading")(delayed(self._normlize_similarity)(x) for x in df[self.score_normlize_column_name])
        return df 




    def filter_by_minus_1_in_score(self, df_from_retriver: pd.DataFrame):
        self.logger.info(f'shape before filtering is {df_from_retriver.shape}')

        df_from_retriver=df_from_retriver[~df_from_retriver['sec_scores'].apply(lambda x: -1 in x)].copy()
        
        print(f'shape after filtering is {df_from_retriver.shape}')
        
        return df_from_retriver
    

    def update_length_of_candidates(self, df_retriver: pd.DataFrame, verbose=False) -> pd.DataFrame:
        df_retriver['length_of_occ_scores']=df_retriver['occ_scores_normlize'].apply(len)
        df_retriver['length_of_occ_candidates']=df_retriver['occ_candidates'].apply(len)
        df_retriver['length_of_sec_scores']=df_retriver['sec_scores_normlize'].apply(len)
        df_retriver['length_of_sec_candidates']=df_retriver['sec_candidates'].apply(len)

        # test if  num of candidates is the same as num of scores
        test_length_occ=df_retriver[df_retriver['length_of_occ_scores']!=
                                            df_retriver['length_of_occ_candidates']].shape[0]
        test_length_sec=df_retriver[df_retriver['length_of_sec_scores']!=
                                            df_retriver['length_of_sec_candidates']].shape[0]

        # test of errors regarding number of candidates
        if test_length_occ!=0 or test_length_sec!=0:
            raise ValueError(f"""number of misallignment between scores and candidates in anaf is  ({test_length_sec})
                            and in mishlah is ({test_length_occ})""")
        
        num_of_diff_candidates=df_retriver[df_retriver['length_of_occ_candidates']!= df_retriver['length_of_sec_candidates']].shape[0]

        if num_of_diff_candidates>0:
            raise ValueError(f"""number of misallignment between anaf and mishlah is  ({num_of_diff_candidates})""")
        
        # how many rows per number of candidates
        df_group_sec=df_retriver.groupby('length_of_sec_scores')['id'].count().reset_index()
        df_group_sec['n_rows']=df_group_sec['length_of_sec_scores'] * df_group_sec['id']
        if verbose:
            print(f'total number or rows by sec is {df_group_sec.n_rows.sum()}')

        df_group_occ=df_retriver.groupby('length_of_occ_scores')['id'].count().reset_index()
        df_group_occ['n_rows']=df_group_occ['length_of_occ_scores'] * df_group_occ['id']
        if verbose:
            print(f'total number or rows by sec is {df_group_occ.n_rows.sum()}')

        return df_retriver
    
    # =================================================== #
    #                      product cartesian of scores    #
    # =================================================== #
    def cartesian_product_by_similarities(self, df: pd.DataFrame) ->pd.DataFrame:
        """

        """
        # start with mislah with exploding rows
        # save the orginal lists
        start=time.time()

        # expand the list of similarities by mishlah
        df=df.explode(['occ_scores_normlize','occ_candidates'],ignore_index=True)
        
        # get the code by dict mishlah_dict_label_to_code for each candidate
        if self.anaf_or_mishlah=="mishlah":
            df[self.candidates_column_name]=df['occ_candidates'].map(lambda x: self.mishlah_dict_label_to_code.get(x,None))
        else:
            df[self.candidates_column_name]=df['sec_candidates'].map(lambda x: self.anaf_dict_label_to_code.get(x,None))
        print(f"explode took {time.time()-start:.4f} seconds")

        return df
    

    def merge_to_numeric_features_and_set_rank(self, df: pd.DataFrame, df_numeric: pd.DataFrame, id_retriver:str,
                                                id_origanl: str ) -> pd.DataFrame:
        df=df.merge(df_numeric,left_on=id_retriver,right_on=id_origanl, how='left',suffixes=('','_dupp'))
        print('df shape after merge with df numeric is ',df.shape)
        
        # add rank of occupation
        df['rank_occ']=np.where(df['SemelMishlahSofi']== df['occ_candidates_code'],1,0)
        # add rank of sector 
        df['rank_sec']=np.where(df['SemelAnafSofi']==df['sec_candidates_code'],1,0)
        
        df['count_rank_occ']=df.groupby(id_retriver)['rank_occ'].transform('sum')

        df['count_rank_sec']=df.groupby(id_retriver)['rank_sec'].transform('sum')
        
        # test if per id group there are more then 1 rank=1 , will add sec later!!!
        # delete duplicate correct in sec 
        if df[df['count_rank_occ']>1].shape[0]>0:
            test_rank_more_then_1_chosen= df[df['count_rank_occ']>1].groupby([ 'count_rank_occ','count_rank_sec'])[id_retriver].nunique().reset_index()
            print('occ, number of group id with more then one is ', test_rank_more_then_1_chosen[id_retriver].sum(), ' by the next options:')
            display(df[df['count_rank_occ']>1].head(2))
            df=df.sort_values(by=[id_retriver,'rank_sec'], ascending=[True,False])
            # remove candidates with duplicate ranking
            df=df[((df .groupby(id_retriver)['rank_occ'].transform('cumsum')==1) | (df['rank_occ']==0))]
            print("df shape after filtering is ",df.shape )
        
        # delete duplicate correct in sec 
        if df[df['count_rank_sec']>1].shape[0]>0:
            test_rank_more_then_1_chosen= df[df['count_rank_sec']>1].groupby([ 'count_rank_occ','count_rank_sec'])[id_retriver].nunique().reset_index()
            print('sec, number of group id with more then one is ', test_rank_more_then_1_chosen[id_retriver].sum(), ' by the next options:')
            display(df[df['count_rank_sec']>1].head(2))
            df=df.sort_values(by=[id_retriver,'rank_occ'], ascending=[True,False])
            # remove candidates with duplicate ranking
            df=df[((df .groupby(id_retriver)['rank_sec'].transform('cumsum')==1) | (df['rank_sec']==0))]
            print("df shape after filtering is ",df.shape )

        # occ, test if per id group there are no rank=1
        if df[df['count_rank_occ']==0].shape[0]>0:
            test_missing_rank = df[df['count_rank_occ']==0][[id_retriver, 'count_rank_occ','count_rank_sec']].drop_duplicates()
            print('number of group id with no ranking is ', test_missing_rank.shape,  'and two examples:')
            display(df[df['count_rank_occ']==0].head(2))
            # remove all group id with no ranking
            df=df[df['count_rank_occ']!=0]
            print("df shape after filtering is ",df.shape )
        
        # sec, test if per id group there are no rank=1
        if df[df['count_rank_sec']==0].shape[0]>0:
            test_missing_rank = df[df['count_rank_sec']==0][[id_retriver, 'count_rank_sec','count_rank_occ']].drop_duplicates()
            print('number of group id with no ranking is ', test_missing_rank.shape,  'and two examples:')
            display(test_missing_rank.head(2))
            # remove all group id with no ranking
            df=df[df['count_rank_sec']!=0]
            print("df shape after filtering is ",df.shape )

        return df
    
    # =================================================== #
    #                      feature engeneering            #
    # =================================================== #
    def filter_rows_and_choose_columns(self, df: pd.DataFrame, cols_to_use: list,target_value: str, candidate_col: str,
                                   is_chosen_col: str) -> pd.DataFrame:
        
        # remove rows where SmselMishlahSofi is None
        df_filter=df[~df[target_value].isnull()].copy()
        print( "num of rows filtered out by target ", df.shape[0]-df_filter.shape[0] )
        
        # remove rows where candidate is None
        if len(df_filter[(df_filter[candidate_col].isnull()) & (df_filter[is_chosen_col]==1) ])>0:
            raise RuntimeError('candidate column is null, verify the text candidate is  in th dict')
            
        
        df_filter=df_filter[~df_filter[candidate_col].isnull()].copy()
        print( "num of rows filtered out by candidate", df.shape[0]-df_filter.shape[0] )
        
        # choose relevant columns for the model
        df_filter=df_filter.loc[:, cols_to_use]
        return df_filter
    
    def split_target_to_four_columns(self, df: pd.DataFrame,col: str):
        temp=df[col].astype(str).str.strip()

        if (temp.str.len()!=4).any():
            raise ValueError("target value does not equal  4 digits")
        padded=temp.str.pad(4, side='right', fillchar=" ")

        # split into charachter
        new_cols=padded.apply(lambda x: list(x))

        df[f"{col}_1"]=new_cols.apply(lambda x: x[0])
        df[f"{col}_2"]=new_cols.apply(lambda x: x[0]+x[1])
        df[f"{col}_3"]=new_cols.apply(lambda x: x[0]+x[1]+x[2])
        #df[f"{col}_4"]=new_cols.apply(lambda x: x[3])

        return df
    
    def convert_text_to_number(self, val):
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
                print("")
                raise ValueError (f"cannot convert value '{val}' to number")
            
    def convert_text_df(self, df: pd.DataFrame):
        cols_to_convert=df.columns.difference(['id'])
        df[cols_to_convert]=df[cols_to_convert].map(self.convert_text_to_number)
        # convert columns after update the values
        columns_convert_to_int=['TeudaGvoha','YeshuvAvoda','shnotlimud','gil']
        try:
            tmp=df[columns_convert_to_int].apply(pd.to_numeric, errors='raise')
            if not  ((tmp % 1 )==0).all().all():
                raise ValueError("Non-integer values {}")
            df[columns_convert_to_int]=df[columns_convert_to_int].astype("Int64")
        except Exception as e:
            print("error is ",e)
            raise
        return df
    
    def update_categorial_by_onehotencoder(self, df: pd.DataFrame, cols_to_chg :List, verbose=True) -> pd.DataFrame:
        print(cols_to_chg)
        for col in cols_to_chg:
            
            encoder=OneHotEncoder(sparse_output=False)
            column_encoded=encoder.fit_transform(df[[col]])
            df_columns_encoded=pd.DataFrame(column_encoded, columns=[f"{col}_{cat}" for cat in encoder.categories_[0]], index=df.index)
            df= pd.concat([df,df_columns_encoded], axis=1)
            if verbose:
                print(f"update column '{col}")
                display(df.loc[:,df.columns[df.columns.str.contains(col)]].drop_duplicates())
            # drpo orginal column
            df=df.drop(col, axis=1)
        return df
    
    def create_eduction_gap(self, df:pd.DataFrame, is_train:bool= False):
        # update series with the median shnotlimud
        series_meidan_shnotlimud_by_gil=df['gil'].map(self.meidan_shnotlimud_by_gil)
        
        # create the new column education gap, if the feature are outliers set -50
        
        df['education_gap']=np.where((df['shnotlimud']>-1) & (df['shnotlimud']<26)& (df['gil']>0),
                                    df['shnotlimud']-series_meidan_shnotlimud_by_gil,-50)
        return df 

    # =================================================== #
    #                      model prediction               #
    # =================================================== #

    def predict_ranking_fast(self, df, model, feature_list, y_test: pd.Series | None, X_train, model_type='occ'):

        # add columns missing in the test/valid df and fill with 0
        missing_cols=set(feature_list)- set(df.columns)
        if len(missing_cols)>0: 
            print(f"missing columns in test/valid are {missing_cols}")
        for col in missing_cols:
            df[col]=0
        
        df["_pos"]=df.groupby("id").cumcount()
        df= df.sort_values(["id","_pos"])
        
        df["score_pred"]=model.predict(df[feature_list])

        
        if model_type=='occ':
            rank_pred_col='rank_occ_pred'
            candidate_col='occ_candidates_code'
            similarity_col='occ_scores_normlize'
        else:
            rank_pred_col='rank_sec_pred'
            candidate_col='sec_candidates_code'
            similarity_col='sec_scores_normlize'

        ## adding freq_map if model score is not unique
        if candidate_col in X_train.columns:
            freq_map=(
                X_train[candidate_col].value_counts(normalize=True)
            )

            df['candidate_freq']=df[candidate_col].map(freq_map).fillna(0)
        else:
            df['candidate_freq']=1
        df[rank_pred_col]=df.sort_values(['id','score_pred',similarity_col, 'candidate_freq'], ascending=[True,False,False,False]).\
            groupby('id').cumcount()+1
        # old code
        #df[rank_pred_col]=df.groupby('id')['score_pred'].rank(method='dense', ascending=False)
        
        shape_input=df.shape[0]
        if y_test is not None:
            df=pd.concat([df, y_test], axis=1)
        if shape_input!= df.shape[0]:
            raise RuntimeError(f"the shape of input df ({shape_input}) is different from output df ({df.shape})")
        return df
    
    # Attaching the models' precision per class to the prediction output file
    def get_percision_per_class(self,data):

        percision_param = self.percision_per_class

        data[self.precision_class_column_name] = [percision_param[x] if x in percision_param.keys() else 0.0 for x in data[self.predictions_column_name]]
        self.logger.info("presicion per class was added succesfully")
        return data

    # =================================================== #
    #   run infrence                                      #
    # =================================================== #
    def run_inference_process(self):
                        
            # normlize scores
            df_retriver=self.retriever_data.copy()
            df_retriver=self.normlize_columns(df_retriver)

            # filter out minus 1 scores- should I remove it from inference? 
            df_retriver=self.filter_by_minus_1_in_score(df_retriver)

            # check number of candidates 
            df_retriver=self.update_length_of_candidates(df_retriver)

            # product cartesian of scores ## need to change so mishalh and anaf multiple seperataly
            df_retriver_explode=self.cartesian_product_by_similarities(df_retriver)

            # merge retriver and orginal input with numeric features
            df_retriver_explode=self.merge_to_numeric_features_and_set_rank(df_retriver_explode,
                                                                             self.data_input_for_inference,
                                                                               'id', self.key_column )
            # filter nulls and choose columns, ### need to send mishalh or anaf
            df_model=self.filter_rows_and_choose_columns(df_retriver_explode, self.columns_to_use,
                                                             self.predictions_column_name, self.candidates_column_name,
                                                             self.rank_column_name)
            #split target to digits
            df_model=self.split_target_to_four_columns(df_model,self.candidates_column_name)

            # convert text to numbers
            df_model=self.convert_text_df(df_model)

            # update  categorial columns
            df_model=self.update_categorial_by_onehotencoder(df_model,self.cols_to_chg_categorial,False)
            
            # create column eductaion gap-- need to update the function
            df_model=self.create_eduction_gap(df_model,False)

            # 
            df_model=self.predict_ranking_fast(df_model, self.model, X_train_occ.columns.tolist(), None, X_train_occ,
                                                'occ')
