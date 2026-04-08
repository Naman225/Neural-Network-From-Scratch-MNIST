from fastapi import FastAPI,UploadFile,File,HTTPException
from pydantic import BaseModel
from src.pipeline.prediction_pipeline import Prediction
from src.utils.logger import get_logger
from PIL import Image
import numpy as np
import io
import uvicorn

logger = get_logger(__name__)

app =FastAPI()
predictor = Prediction()

class InputData(BaseModel):
    data : list[float] 

@app.get('/')
def home():
    return {'message' : 'API is running'}
@app.post('/predict')
def predict(input_data : InputData):
    logger.info('Request recieved')
    try:
        result = predictor.predict(input_data.data)
        logger.info("Prediction Done")
        return {"prediction" : result}
    except Exception as e:
        logger.error(f"Error occured : {e}")
        return {"error" : str(e)}

ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg"]
@app.post('/predict-image')
async def predict_image(file :UploadFile = File()):
    if file.content_type not in ALLOWED_TYPES:
        logger.error("Invalid File Type")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Only PNG and JPEG are supported."
        )
        
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data))
        logger.info("Image Received")
        img = img.resize((28,28)).convert('L')
        
        img_np = np.array(img)
        
        img.close()
        input_vector = img_np.flatten().reshape(784,1) /255.0
        logger.info("Preprocessing Done")
        result = predictor.predict(input_vector)
        logger.info("Prediction Done")
        
        return {'prediction' : result}
    except Exception as e:
        logger.error(f"Error occured : {e}")
        raise HTTPException(status_code=500, detail=f"Error occured at {e}")


if __name__ == "__main__":
    uvicorn.run("app:app" , port=8000 , reload=True)



