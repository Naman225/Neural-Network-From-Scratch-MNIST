import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        pass
    def preprocess(self ,train_df , test_df,epsilon=0.03):       
        logger.info("Data preprocessing started")
        X_train = train_df.drop('label', axis=1).values / 255
        y_train = train_df['label']
        y_train = pd.get_dummies(y_train)
        X_test = test_df.drop('label', axis=1).values / 255
        y_test = test_df['label']
        y_test = pd.get_dummies(y_test)

       

        y_train = y_train.reindex(columns=range(10), fill_value=0)
        y_test  = y_test.reindex(columns=range(10), fill_value=0)
        y_train = y_train.values.T
        y_test  = y_test.values.T
        num_classes = y_train.shape[0]
        y_train = y_train * (1 - epsilon) + epsilon / num_classes
        y_test = y_test * (1 - epsilon) + epsilon / num_classes
        X_train=X_train.T
        X_test=X_test.T
        
        logger.info(
            f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, "
            f"X_test: {X_test.shape}, y_test: {y_test.shape}"
        )

        return X_train,y_train,X_test,y_test


if __name__ == '__main__':
    ingestion=DataIngestion()
    train_df, test_df = ingestion.run()
    transformer = DataTransformation()
    X_train, y_train, X_test, y_test = transformer.preprocess(train_df, test_df)

