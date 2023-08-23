import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer 

from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import os
from dataclasses import dataclass
from src.utilis import save_object
from src.logger import logging
@dataclass
class DataTranformatiomConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTrafromation:
    def __init__(self):
        self.data_tranformation_config=DataTranformatiomConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data tranformatiopn initiated')
            #define categorical colomns and numerical colomns
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']

            #define custom ranking for categorical colomns

            cut_categories=['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline initiated')

            ##numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]
            )

            ##categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('Ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

            logging.info('Pipeline completed')


        except Exception as e:
            logging.info('Error in get data transformation')
            raise CustomException(e,sys)


    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('onbtaining preprocessor object')

            preprocessing_obj=self.get_data_transformation_object()

            target_coloumn_name='price'
            drop_columns=[target_coloumn_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_coloumn_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_coloumn_name]

            ##tranformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing obj on traing and testing datasets')

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('preprocessor pickel file saved')

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('exception occured in the initiate tranformation cell')
            raise CustomException(e,sys)