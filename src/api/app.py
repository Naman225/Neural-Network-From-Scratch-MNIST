from fastapi import FastAPI,UploadFile,File,HTTPException
from pydantic import BaseModel
from src.pipeline.prediction_pipeline import Prediction
from src.utils.logger import get_logger
from PIL import Image
import numpy as np
import io
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.config.config_loader import config

logger = get_logger(__name__)

app =FastAPI()
predictor = Prediction()
app.mount("/static", StaticFiles(directory="src/frontend"), name="static")

class InputData(BaseModel):
    data : list[float] 

@app.get('/',response_class=HTMLResponse)
def home():
    with open("src/frontend/index.html", "r") as f:
        return f.read()

@app.get('/health')
def health():
    return {"status": "ok"}

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

ALLOWED_TYPES = config["file"]["allowed_types"]
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
        img = img.resize(tuple(config["image"]["size"])).convert('L')

        
        img_np = np.array(img)


        if np.mean(img_np) > config["image"]["inversion_threshold"]:
            logger.info("White background detected → Inverting image")
            img_np = 255 - img_np
        else:
            logger.info("Dark background detected → No inversion")

        
        input_vector = img_np.flatten().reshape(784,1) / config["image"]["normalization"]
        
        logger.info("Preprocessing Done")
        probs = predictor.predict_proba(input_vector)

        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))

        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [
            {"digit": int(i), "confidence": float(probs[i])}
            for i in top3_idx
        ]

        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "top3": top3
        }
        
    except Exception as e:
        logger.error(f"Error occured : {e}")
        raise HTTPException(status_code=500, detail=f"Error occured at {e}")


if __name__ == "__main__":
    uvicorn.run("api.app:app" , port=8000 , reload=True)



