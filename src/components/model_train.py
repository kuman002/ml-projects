import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainconfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")
    

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainconfig()
        
    def initiate_model_train(self, train_array, test_arrary):
        try:
            logging.info("Split training and testing input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_arrary[:, :-1],
                test_arrary[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "X-Neighbours classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=0),
                "AdaBoost Classifier": AdaBoostRegressor()
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                },
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                "Linear Regression": {},
                "KNeighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # 1=Manhattan, 2=Euclidean
                },
                "XGBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                "CatBoost Regressor": {
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200, 300]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                }
            }
            
            model_report: dict=evaluate_model(x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, models=models, params=params)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_train_config.train_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        
        
        except Exception as e:
            raise CustomException(e, sys)