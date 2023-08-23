import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object
from dataclasses import dataclass
import sys
from src.utilis import evaluate_model

@dataclass
class ModelTrainingConfig:
    traing_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('splitting dependent and independent variables')
            X_train,y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
                'LinearRegression' :LinearRegression(),
                'Lasso' :Lasso(),
                'Rigde' :Ridge(),
                'Elasticnet':ElasticNet()
            }

            model_report:dict=evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n========================================================================\n')
            logging.info(f'Model Report : {model_report}')

            #to get the best model
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            print(f'Best model found,model name :{best_model_name}, R2 score={best_model_score}')
            print('\n=====================================================================\n')
            logging.info(f'Best model found,model name :{best_model_name}, R2 score={best_model_score}')

            save_object(
                file_path=self.model_trainer_config.traing_model_file_path,
                obj=best_model
            )

        except Exception as e :
            logging.info('Error occured in model trainer')
            raise CustomException(e,sys)