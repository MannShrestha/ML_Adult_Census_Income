import os
import sys
from censusIncome import logger
from censusIncome.exception import CustomException
from censusIncome.components.data_ingestion import DataIngestion
from censusIncome.components.data_transformation import DataTransformation
from censusIncome.components.model_trainer import ModelTrainer
from dataclasses import dataclass


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _=data_transformation.initiate_data_transformation(train_data, test_data)
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))



### Run in terminal>> python src/censusIncome/pipeline/training_pipeline.py