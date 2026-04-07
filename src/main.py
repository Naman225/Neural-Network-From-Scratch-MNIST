import time
from src.pipeline.train_pipeline import TrainingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)
if __name__ =='__main__':
    tic = time.time()
    train=TrainingModel()
    train.train()
    toc=time.time()
    logger.info(f"Training completed in {toc -tic : 2f} seconds")
