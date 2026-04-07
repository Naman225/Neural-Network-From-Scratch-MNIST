import numpy as np
class DataValidation:
    def __init__(self):
        pass

    def validate(self,X_train,X_test,y_train,y_test):
        ## Checking shape

        if X_train.shape[0] != 784 or X_test.shape[0] != 784:
            raise Exception("Rows is not 784")
        if y_train.shape[0] != 10 or y_test.shape[0] != 10:
            raise Exception("Labels must be one-hot encoded with 10 classes")
        if X_train.shape[1] != y_train.shape[1]:
            raise Exception("columns are different for both X_train and y_train")
        if X_test.shape[1] != y_test.shape[1]:
            raise Exception("columns are different for both X_test and y_test")
        
        ## Checking labels after one-hot + optional label smoothing
        if not (
            (y_train.min() >= 0 and y_train.max() <= 1) and
            (y_test.min() >= 0 and y_test.max() <= 1)
        ):
            raise Exception("Label values must be in [0, 1]")
        if not (
            np.allclose(np.sum(y_train, axis=0), 1.0, atol=1e-6) and
            np.allclose(np.sum(y_test, axis=0), 1.0, atol=1e-6)
        ):
            raise Exception("Each label column must sum to 1")
       
        
        ## Checking values for X_train if its between 0 and 1
        if not ( X_train.min() >= 0  and X_train.max() <=1):
            raise Exception("Train values are not between 0 and 1")
        if not ( X_test.min() >= 0  and X_test.max() <=1):
            raise Exception("Test values are not between 0 and 1")
        
        ##Checking nan values

        if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
            raise Exception('null values present')
        
        ## Checking for X_train and X_test
        if X_train.shape[0] != X_test.shape[0]:
            raise Exception ("Shapes not matched")
