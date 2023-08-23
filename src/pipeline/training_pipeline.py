import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTrafromation
from src.components.model_training import ModelTrainer

## run data ingestion

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.Initiate_data_Ingestion()
    data_tranformation=DataTrafromation()
    train_arr,test_arr,_=data_tranformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)