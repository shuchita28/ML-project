import sys
sys.path.insert(0, '/Users/shuchitamishra/Desktop/Jobs/Study /ML-project/src')
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()
    
    def get_DataTransformer_obj(self):
        try:
            num_cols = ["B12", "Age", "Gender", "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050","DPQ060","DPQ070","DPQ080","DPQ090","SLQ050","SLQ060","IND235"]
            cat_cols = []

            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                           ("scaler", StandardScaler())])
            logging.info("Numerical columns Standardized")

            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                           ("one_hot_encoder", OneHotEncoder()),
                                           ("scaler", StandardScaler())])
            logging.info("Categorical columns encoded")
            
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, num_cols), 
                 ("cat_pipeline", cat_pipeline, cat_cols)])
            return preprocessor
        except Exception as e:
            raise(CustomException(e, sys))
    
    def intiate_dataTransformation(self, train_path, test_path):
        try:
            train_df = pd.red_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully")

            preprocessor_obj = self.get_DataTransformer_obj()
            target_col = "B12"
            num_cols = ["B12", "Age", "Gender", "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050","DPQ060","DPQ070","DPQ080","DPQ090","SLQ050","SLQ060","IND235"]

            input_feature_train_df = train_df.drop(columns = [target_col], axis = 1)
        except Exception as e:
            raise CustomException(e, sys)
