import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class dataTransformConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")
    
    
class DataTransform:
    def __init__(self):
        self.data_transform_config = dataTransformConfig()
        
    def get_data_transform_obj(self):
        try:
            numerical_colums = ["writing score","reading score"]
            categorical_columns=[
               "gender",
               "race/ethnicity",
               "parental level of education",
               "lunch",
               "test preparation course"
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())
                ]
            )
            
            logging.info("Numerical columns Standard Scaling Completed ....")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical Columns encoding Compeleted ....")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_colums),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed...")
            
            preprocessing_obj = self.get_data_transform_obj()
            
            target_column_nam = "math score"
            numerical_columns = ["reading score","writing score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_nam], axis=1)
            target_feature_train_df = train_df[target_column_nam]
            
            input_feature_test_df = test_df.drop(columns=[target_column_nam])
            target_feature_test_df = test_df[target_column_nam]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_arc = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arc = preprocessing_obj.transform(input_feature_test_df)
            
            train_arc = np.c_[
                input_feature_train_arc, np.array(target_feature_train_df)
            ]
            
            test_arc = np.c_[
                input_feature_test_arc, np.array(target_feature_test_df)
            ]
            
            logging.info("Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transform_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return {
               "train_arc" : train_arc,
                "test_arc" : test_arc,
                "preprocessing_obj" : self.data_transform_config.preprocessor_obj_file_path
            }
        except Exception as e:
            raise CustomException(e, sys)