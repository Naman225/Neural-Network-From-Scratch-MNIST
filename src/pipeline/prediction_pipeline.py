from src.utils.save_load import load_path
import numpy as np
from src.components.model_trainer import ModelTraining
from src.utils.logger import get_logger

logger=get_logger(__name__)

class Prediction:
    def __init__(self,model_path="artifacts/model.pkl"): 
        try:
            logger.info(f"Loading model from {model_path}")
            obj = load_path(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f" Failed to load model from '{model_path}'. Original error: {e}")
        if "params" not in obj or "layers" not in obj:
            raise FileNotFoundError("Model file missing required keys: 'params' and 'layers'")
        self.layers , self.params= obj['layers'] , obj['params']
        self.model = ModelTraining(self.layers)
        self.model.params =self.params
    
    def _preprocess_input(self , data : list ):
        input_size = self.layers[0]
        logger.info("Received input for preprocessing")
        try:
            data = np.array(data,dtype=float)
            logger.debug(f"Input converted to numpy with shape: {data.shape}")
        except:  
            raise ValueError("Conversion fails!")
        if data.size == 0:
            logger.error("Input data is empty")
            raise ValueError("Input data cannot be empty")
        if data.ndim ==1:
            data =data.reshape(-1,1)
        if np.max(data) >1:
            
            data = data /255
            logger.debug("Input normalized")
        if data.shape[0] == 1:
            data = data.T
        logger.info(f"Input shape after preprocessing: {data.shape}")
        
       
        if data.shape[1] >1:
            logger.error("Multiple columns detected")
            raise ValueError("Only single sample allowed (column > 1 detected)")
        
        if data.shape[0] != input_size:
            logger.error("Invalid input size")
            raise ValueError(f"input must have {input_size} pixels")
        

        return data
    def predict(self,X):
        logger.info("Prediction started")
        processed_data = self._preprocess_input(X)
        prediction,_ =self.model.predict(processed_data)
        logger.info(f"Prediction result: {prediction[0]}")
        return int(prediction[0])

        



    