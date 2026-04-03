import time
from src.pipeline.train_pipeline import TrainingModel

if __name__ =='__main__':
    tic = time.time()
    train=TrainingModel()
    train.train()
    toc=time.time()
    print("Training time :" ,toc-tic)
