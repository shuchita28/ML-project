import os
import sys
from dataclasses import dataclass
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model

from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

@dataclass
class model_trainer_config:
    train_model_file_path = os.path.join('artifact', 'model.pkl')

class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:,-1], test_arr[:, :-1], test_arr[:,-1])
            models = {
                DecisionTreeClassifier(), "Decision Tree",
                RandomForestClassifier(n_estimators=100), "Random Forest",
                GaussianNB(), "Gauissian Naive Bayes",
                KNeighborsClassifier(n_neighbors=10), "kNN",
                LogisticRegression(), "Logistic Regression",
                SVC(), "Support Vector Machines"
                }
            model_report:dict= evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)
            
        except:
            pass
