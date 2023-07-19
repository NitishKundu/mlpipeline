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
            
            
            
            
            params = {
                
                "DecisionTree Regressor": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'splitter':['best','random'],
                        'max_features':['sqrt','log2'],
                },
                
                "RandomForest Regressor":{
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
                        'max_features':['sqrt','log2',None],
                        'n_estimators': [8,16,32,64,128,256]
                },
                
                "Gradient Boosting Regressor":{
                        'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'criterion':['squared_error', 'friedman_mse'],
                        'max_features':['sqrt','log2'],
                        'n_estimators': [8,16,32,64,128,256]
                },
                
                "Linear Regression":{},
                
                
                "K-Neighbors Regressor":{
                        'n_neighbors': range(1, 21),  # Values from 1 to 20 (inclusive)
                        'weights': ['uniform', 'distance'],  # Two options: 'uniform' and 'distance'
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Four algorithm options
                        'leaf_size': range(10, 51, 10),  # Values: 10, 20, 30, 40, 50
                        'p': [1, 2],  # Two distance metric options: Manhattan (1) and Euclidean (2)
                        'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']  # Four distance metrics
                },
                
                "XGBoost Regressor":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256]
                },
                
                "CatBoost Regressor":{
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                },
                
                "AdaBoost Regressor":{
                        'learning_rate':[.1,.01,0.5,.001],
                        'loss':['linear','square','exponential'],
                        'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            
            
            
            
            model_report : dict = evaluate_models(
                                                X_train=X_train, y_train=y_train, 
                                                X_test=X_test, y_test=y_test, 
                                                models=models,
                                                params=params
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
            
            return f" The best model is {best_model_name} and the R_Squared is {r2_square}"   
                    
        
        except Exception as error:
            raise CustomException(error, sys)