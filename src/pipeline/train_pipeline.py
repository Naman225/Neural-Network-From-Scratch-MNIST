from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining
from src.components.evaluate import Evaluate
from src.utils.save_load import save_object
import pandas as pd
import copy
from src.utils.logger import get_logger

logger=get_logger(__name__)
class TrainingModel:
    def __init__(self):
        pass

    def train(self,epoch=70,batch_size=64,tolerance = 1e-4):
        logger.info("Training Started with Epochs: 70, Batch size: 64")
        layers=[784,128,64,32,10]
        ingestion=DataIngestion()
        train_df, test_df = ingestion.run()
        logger.info("Data loaded successfully")
        
        transformer = DataTransformation()
        X_train, y_train, X_test, y_test = transformer.preprocess(train_df, test_df)
        logger.info("Data preprocessing completed")
        logger.info(f"X_train shape: {X_train.shape} y_train shape : {y_train.shape}")
        training = ModelTraining(layers)
        training.initialize_parameters()
        logger.info("Model initialized with layers: [784,128,64,32,10]")
        best_cost=float('inf')
        patience = 5
        counter = 0
        best_parameters=None
        for i in range(epoch):
            logger.info(f"Epoch {i} started")
            if i < 10:
                
                learning_rate=0.02
            elif i <20:
                learning_rate=0.08
            else :
                learning_rate = 0.002
            mini_batch = training.create_mini_batches(X_train,y_train,batch_size)
            epoch_cost = 0
            for  X_batch,y_batch in mini_batch:
                AL,caches=training.full_linear_activation_forward(X_batch)
                cost=training.compute_cost(AL,y_batch)
                epoch_cost +=cost
                grads=training.full_backward_activation(AL,y_batch,caches)
                
                training.update_parameters(grads,learning_rate)
               
            logger.debug(f"Learning rate: {learning_rate}")
            epoch_cost /= len(mini_batch)
            logger.info(f"Epoch  {i} - Cost : {epoch_cost:.6f}")
            if epoch_cost  < best_cost - tolerance:
                best_cost=epoch_cost
                best_parameters = copy.deepcopy(training.params)
                counter =0
                logger.debug("Model improved, updating best parameters")
                
            else :
                counter +=1
            if counter >= patience:
                logger.info(f"Stopped at epoch {i} after {patience} epochs without improvement")
                break
        if best_parameters is not None:
            training.params = best_parameters
            logger.info(f"Restored best model with cost: {best_cost:.6f}")
        logger.info("Training completed successfully")
        model_data=training.retrieve_data()

        logger.info("Saving model to artifacts/model.pkl")
        save_object("artifacts/model.pkl",model_data)
        logger.info("Model saved successfully")

        preds,AL=training.predict(X_test)
        logger.info("Evaluation started")
        evaluate = Evaluate()
        accuracy =evaluate.accuracy(preds,y_test)
        confusion = evaluate.confusion_matrix(preds,y_test)
        logger.debug("AL is ",AL.min(), AL.max())
        logger.info(f"Accuracy: {accuracy}")

        error =evaluate.get_wrong_predictions(preds,y_test)
        error_img=evaluate.analyze_patterns(error,X_test)
        evaluate.visualize_error(error_img[:10])
        error_imgs = evaluate.add_confidence(AL,error_img)

        evaluate.visualize_error_with_confidence(error_imgs[:10])
        evaluation_data = {
            'accuracy':accuracy,
            'confusion' : confusion
        }
        confusion_df=pd.DataFrame(confusion)
        confusion_df.to_csv('artifacts/confusion_matrix.csv', index=False)
        evaluate.visualize_confusion_matrix(confusion_df)

        logger.info("Saving evaluation data to artifacts/evaluate.pkl")
        save_object("artifacts/evaluate.pkl",evaluation_data)
        
        