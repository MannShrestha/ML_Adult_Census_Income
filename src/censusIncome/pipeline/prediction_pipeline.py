import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from censusIncome.exception import CustomException
from censusIncome.utils.common import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    
    def predict(self, features):
        try:

            preprocessor_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
            model_path = os.path.join("artifacts/model_trainer", "model.pkl")
            print("Before Loading")

            preprocessor = load_object(file_path = preprocessor_path)
            model = load_object(file_path = model_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                    age:int,
                    workclass:int, 
                    education_num:int, 
                    marital_status:int, 
                    occupation:int,
                    relationship:int,  
                    race:int,
                    sex:int,  
                    capital_gain:int, 
                    capital_loss:int,
                    hours_per_week:int):
        
        self.age = age
        self.workclass = workclass
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week

    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education_num":[self.education_num],
                "marital_status":[self.marital_status],
                "occupation":[self.occupation],
                "relationship":[self.relationship],
                "race":[self.race],
                "sex":[self.sex],
                "capital_gain":[self.capital_gain],
                "capital_loss":[self.capital_loss],
                "hours_per_week":[self.hours_per_week],
            }

            data = pd.DataFrame(custom_data_input_dict)

            return data


        except Exception as e:
            raise CustomException(e, sys)



        