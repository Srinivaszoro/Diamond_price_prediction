import os
import sys
import pandas as pd


from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_tranformation import DataTrafromation
from src.logger import logging
#Intitialize the Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

#create a class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def Initiate_data_Ingestion(self):
        logging.info('data ingestion step started')
        try:
            df=pd.read_csv("https://raw.githubusercontent.com/Srinivaszoro/FSDSRegression/main/notebooks/data/gemstone.csv")
            logging.info('data read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split started')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=30)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Data ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e,sys)


## run data ingestion

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.Initiate_data_Ingestion()
    data_tranformation=DataTrafromation()
    train_arr,test_arr,_=data_tranformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
    
    