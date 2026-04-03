from src.utils.save_load import load_path
import numpy as np
from src.components.model_trainer import ModelTraining

class Prediction:
    def __init__(self,model_path="artifacts/model.pkl"): 
        try:
            obj = load_path(model_path)
        except Exception as e:
            raise RuntimeError(f" Failed to load model from '{model_path}'. Original error: {e}")
        if "params" not in obj or "layers" not in obj:
            raise FileNotFoundError("Model file missing required keys: 'params' and 'layers'")
        self.layers , self.params= obj['layers'] , obj['params']
        self.model = ModelTraining(self.layers)
        self.model.params =self.params
    
    def _preprocess_input(self , data : list ):
        input_size = self.layers[0]
        try:
            data = np.array(data,dtype=float)
        except:
            raise ValueError("Conversion fails!")
        if data.size == 0:
            raise ValueError("Input data cannot be empty")
        if data.ndim ==1:
            data =data.reshape(-1,1)
        if np.max(data) >1:
            data = data /255
        if data.shape[0] == 1:
            data = data.T
       
        if data.shape[1] >1:
            raise ValueError("Only single sample allowed (column > 1 detected)")
        
        if data.shape[0] != input_size:
            raise ValueError(f"input must have {input_size} pixels")
        

        return data
    def predict(self,X):
        processed_data = self._preprocess_input(X)
        prediction,_ =self.model.predict(processed_data)
        return int(prediction[0])

        



    