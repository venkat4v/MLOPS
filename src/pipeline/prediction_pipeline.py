import sys
import pandas as pd
import os
from src.exception.exception import customexception
from src.logger.logger import logging
from src.utils.utils import load_object 

class PredictPipeline:
   
    def __init__(self):
        print("You are learning MLOPS")
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor_obj=load_object(preprocessor_path)
            model_obj=load_object(model_path)

            scaled_feature = preprocessor_obj.transform(features)
            prediction = model_obj.predict(scaled_feature)

            return prediction

        except Exception as e:
            raise customexception(e,sys)

class CustomData():
    def __init__(self,carat:float,depth:float,table:float,x:float,y:float,z:float,cut:str,color:str,clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {'carat':[self.carat],
                                    'depth':[self.depth],
                                    'table':[self.table],
                                    'x':[self.x],
                                    'y':[self.y],
                                    'z':[self.z],
                                    'cut':[self.cut],
                                    'color':[self.color],
                                    'clarity':[self.clarity]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)