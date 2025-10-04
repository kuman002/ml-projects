import sys
import os
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        
        for model_name, model in models.items():
            model.fit(x_train, y_train)

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # ✅ Use model name as dictionary key
            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)