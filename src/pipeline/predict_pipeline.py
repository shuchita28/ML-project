import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preproccessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preproccessor = load_object(file_path = preproccessor_path)
            data_scaled = preproccessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                B12 : int,
                Age : int,
                Gender : str,
                DPQ010 : int,
                DPQ020 : int,
                DPQ030 : int,
                DPQ040 : int,
                DPQ050 : int,
                DPQ060 : int,
                DPQ070 : int,
                DPQ080 : int,
                DPQ090 : int,
                SLQ050 : int,
                SLQ060 : int,
                IND235 : int):
        self.B12 = B12
        self.Age = Age
        self.Gender = Gender
        self.DPQ010 = DPQ010
        self.DPQ020 = DPQ020
        self.DPQ030 = DPQ030
        self.DPQ040 = DPQ040
        self.DPQ050 = DPQ050
        self.DPQ060 = DPQ060
        self.DPQ070 = DPQ070
        self.DPQ080 = DPQ080
        self.DPQ090 = DPQ090
        self.SLQ050 = SLQ050
        self.SLQ060 = SLQ060
        self.IND235 = IND235

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "B12":[self.B12],
                "Age": [self.Age],
                "Gender": [self.Gender],
                "DPQ010": [self.DPQ010],
                "DPQ020": [self.DPQ020],
                "DPQ030": [self.DPQ030],
                "DPQ040": [self.DPQ040],
                "DPQ050": [self.DPQ050],
                "DPQ060": [self.DPQ060],
                "DPQ070": [self.DPQ070],
                "DPQ080": [self.DPQ080],
                "DPQ090": [self.DPQ090],
                "SLQ050": [self.SLQ050],
                "SLQ060": [self.SLQ060],
                "IND235": [self.IND235]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        

