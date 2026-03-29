import numpy as np
class DataValidation:
    def __init__(self):
        pass

    def validate(self,X_train,X_test,y_train,y_test):
        ## Checking shape

        if X_train.shape[0] != 784 or X_test.shape[0] != 784:
            raise Exception("Rows is not 784")
        if y_train.shape[0] != 1 or y_test.shape[0]!=1:
            raise Exception("Labels not reshaped")
        if X_train.shape[1] != y_train.shape[1]:
            raise Exception("columns are different for both X_train and y_train")
        if X_test.shape[1] != y_test.shape[1]:
            raise Exception("columns are different for both X_test and y_test")
        
        ## Checking labels of y_train
        if not (set(y_train.flatten()).issubset({0,1}) and set(y_test.flatten()).issubset({0,1})):
            raise Exception(f"Labels not converted! ")
       
        
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
