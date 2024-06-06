# Handle Missing value
# Outliers treatment
#Handle Imblanced dataset
#Convert categorical columns into numerical columns

import os
import sys
import numpy as np
import pandas as pd
from censusIncome import logger
from censusIncome.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from censusIncome.utils.common import save_objet


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/data_transformation',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        ''' 
        This function is responsible for data trnasformation
        '''

        try:
            logger.info(" Data Transformation Started")
            numerical_columns = ['age', 'workclass',  'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain',
            'capital_loss', 'hours_per_week']

            # categorical_columns = [
            #     "I have already converted catogerical to num
            #     using labelEncoder() and convert it in to income_cleandata.csv
            #     -- categorical column here"
            # ]


            ### Numerical feature Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )


            ### Categorical feature Pipeline
            # cat_pipeline=Pipeline(
            #     steps=[
            #     ("imputer",SimpleImputer(strategy="most_frequent")),
            #     ("one_hot_encoder",OneHotEncoder()),
            #     ("scaler",StandardScaler(with_mean=False))
            #     ]

            # )


            # logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")


            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                # ("cat_pipelines",cat_pipeline,categorical_columns)

                ]

            )

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)

    
    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lowwer_limit = Q1 - 1.5 * iqr

            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lowwer_limit), col] = lowwer_limit

            return df
        
        except Exception as e:
            logger.info("Outluers handling")
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            numerical_columns = ['age', 'workclass',  'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain',
            'capital_loss', 'hours_per_week']

            for col in numerical_columns:
                self.remove_outliers_IQR(col = col, df = train_df)
            
            logger.info("Outliers capped on our train data")

            for col in numerical_columns:
                self.remove_outliers_IQR(col = col, df = test_df)
            
            logger.info("Outliers capped on our test data")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "income"

            logger.info("Splitting train data into dependent and independent features")

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply transfpormation on our train data and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Apply preprocessor object on our train data and test data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info(f"Saved preprocessing object.")

            save_objet(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


