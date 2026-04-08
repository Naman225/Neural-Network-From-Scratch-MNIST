# Neural Network From Scratch (MNIST)

A NumPy-based, from-scratch implementation of a multi-layer neural network for handwritten digit recognition on MNIST, packaged with a FastAPI backend and a simple web frontend for image-based inference.

## Live Demo

Try the deployed app here: [neural-network-from-scratch-mnist.onrender.com](https://neural-network-from-scratch-mnist.onrender.com)

## Highlights

- End-to-end neural network implementation without TensorFlow/PyTorch
- Vectorized forward and backward propagation
- ReLU + Softmax architecture with categorical cross-entropy loss
- Training and evaluation pipeline with artifact-based model loading
- FastAPI inference API for uploaded digit images
- Frontend UI showing prediction confidence and top-3 classes

## Tech Stack

- Python, NumPy, Pandas
- FastAPI, Uvicorn
- HTML, CSS, JavaScript
- Pickle-based model artifact loading

## Project Structure

```text
src/
├── api/
│   └── app.py                    # FastAPI app and prediction endpoints
├── components/
│   ├── data_ingestion.py         # Dataset loading
│   ├── data_transformation.py    # Normalization / preprocessing
│   ├── model_trainer.py          # Network training logic
│   └── evaluate.py               # Metrics and error analysis
├── config/
│   ├── config.yaml               # Image/file/runtime settings
│   └── config_loader.py          # YAML config loader
├── frontend/
│   ├── index.html                # UI
│   ├── script.js                 # Upload and prediction rendering
│   └── style.css                 # Styling
├── pipeline/
│   ├── train_pipeline.py         # Training workflow
│   └── prediction_pipeline.py    # Model loading + inference wrapper
└── utils/
    ├── logger.py
    └── save_load.py
```

## Model Overview

- Input: `28x28` grayscale image (flattened to `784`)
- Hidden layers: ReLU activations
- Output: `10` classes (digits `0-9`) via Softmax
- Objective: categorical cross-entropy minimization with gradient descent

## API Endpoints

- `GET /` - serves the frontend
- `POST /predict` - predicts from numeric vector input
- `POST /predict-image` - predicts from uploaded image and returns:
  - `prediction` (top class)
  - `confidence` (top class probability)
  - `top3` (top-3 class probabilities)

## Local Setup

```bash
git clone https://github.com/Naman225/Neural-Network-From-Scratch-MNIST.git
cd Neural-Network-From-Scratch-MNIST
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### 1) Train (optional, if artifact is missing)

```bash
python3 src/main.py
```

### 2) Start API + frontend

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://127.0.0.1:8000`

## Configuration

Runtime settings are in `src/config/config.yaml`, including:

- image size
- normalization factor
- inversion threshold
- allowed upload MIME types

## Roadmap

- Mini-batch training and learning-rate scheduling
- Better experiment tracking
- Optional CNN implementation for higher accuracy
- More robust frontend error handling and UX improvements

## Author

Naman Tiwari
