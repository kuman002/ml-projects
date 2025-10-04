import sys
import os
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    """
    Trains and evaluates multiple models with hyperparameter tuning.

    Args:
        x_train, y_train: Training data
        x_test, y_test: Test data
        models (dict): Dictionary of model name -> model instance
        params (dict): Dictionary of model name -> parameter grid

    Returns:
        model_report (dict): model_name -> best r2_score
    """

    try:
        logging.info("Starting model evaluation with hyperparameter tuning.")
        model_report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            param_grid = params.get(model_name, {})

            # Skip GridSearchCV if no params
            if len(param_grid) > 0:
                gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
                gs.fit(x_train, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
                logging.info(f"Best Params for {model_name}: {best_params}")
            else:
                model.fit(x_train, y_train)
                best_model = model

            # Evaluate model on test data
            y_pred = best_model.predict(x_test)
            r2 = r2_score(y_test, y_pred)

            model_report[model_name] = r2

            logging.info(f"{model_name} R2 Score: {r2}")

        logging.info("Model evaluation complete.")
        return model_report
    except Exception as e:
        raise CustomException(e, sys)