import numpy as np
import pandas as pd
from data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        pass
    def preprocess(self ,train_df , test_df):       
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
        X_train=X_train.T
        X_test=X_test.T
        
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        return X_train,y_train,X_test,y_test


if __name__ == '__main__':
    ingestion=DataIngestion()
    train_df, test_df = ingestion.run()
    transformer = DataTransformation()
    X_train, y_train, X_test, y_test = transformer.preprocess(train_df, test_df)

