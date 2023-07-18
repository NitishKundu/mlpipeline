import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_models

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig():
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer():
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        
        try:
            logging.info('Splitting training and test input data')
            
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], 
                train_array[:,-1], 
                test_array[:,:-1], 
                test_array[:,-1]
            )
            
            models = {
                "RandomForest Regressor": RandomForestRegressor(),
                "DecisionTree Regressor": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()               
            }
            
            
            
            model_report : dict = evaluate_models(
                                                X_train=X_train, y_train=y_train, 
                                                X_test=X_test, y_test=y_test, 
                                                models=models
                                                )
            
            # To get the best model score
            best_model_score = max(sorted(model_report.values()))
            
            # To get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException('No best model found as Best model score is less than 0.6')
            logging.info(f'Best model found is {best_model_name} with score {best_model_score}')
            
            save_obj(
                self.model_train_config.trained_model_file_path, 
                obj=best_model
                )
                
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square   
                    
        
        except Exception as error:
            raise CustomException(error, sys)