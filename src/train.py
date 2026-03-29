from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model import ModelTraining
from evaluate import Evaluate
import time
class TrainingModel:
    def __init__(self):
        pass

    def start(self):
        layers=[784,64,32,10]
        ingestion=DataIngestion()
        train_df, test_df = ingestion.run()
        transformer = DataTransformation()
        X_train, y_train, X_test, y_test = transformer.preprocess(train_df, test_df)
        training = ModelTraining(layers)
        training.initialize_parameters()
        for i in range(1500):
            AL,caches=training.full_linear_activation_forward(X_train)
            cost=training.compute_cost(AL,y_train)
            grads=training.full_backward_activation(AL,y_train,caches)
            training.update_parameters(grads)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")
            
        preds=training.predict(X_test)
        evaluate = Evaluate()
        accuracy =evaluate.accuracy(preds,y_test)
        confusion = evaluate.confusion_matrix(preds,y_test)
        print("Metrics for accuracy is :",accuracy)
        print("Metrics for confusion is :",confusion)

            

if __name__ =='__main__':
    tic = time.time()
    train=TrainingModel()
    train.start()
    toc=time.time()
    print(toc-tic)
