
import os
import joblib


class Ops:
    
    data_dir=''

    def __init__(self):
        self.data_dir = os.path.abspath(os.path.dirname(__file__))
        return


    def save_model(self, model,name):


        joblib.dump(model, os.path.join(self.data_dir, name))
        return

    def get_model(self,name):
        model = joblib.load(os.path.join(self.data_dir, name))
        return model


    