import os
import sys
import numpy as np
import pandas as pd
from censusIncome import logger
from censusIncome.exception import CustomException

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score

from censusIncome.utils.common import save_objet, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBRegressor": XGBClassifier(),
                "Logistic Regression": LogisticRegression(),
                "KNeighborsRegressor": KNeighborsClassifier(),
            }


            params = {
                "Random Forest": {
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },

                "Decision Tree": {
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    
                },

                "AdaBoost": {
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "XGBRegressor": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Logistic Regression": {
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },

                "KNeighborsRegressor":{
                    'n_neighbors': [3,4,5,7,9,11],
                    'weights': ['uniform', 'distance'],
                    'p': [1,2]
                    # p=1 is equivalent to manhattan distance
                    # p=2 is equivalent to euclidean distance
                }

            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            ## set Threshold point 
            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")

            logger.info(f"Best found model on both training and testing dataset")

            save_objet(
                file_path = self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        
        except Exception as e:
            raise CustomException(e, sys)
