import pandas as pd
import kagglehub
import os
class DataIngestion:
    def __init__(self):
        pass

    def download_data(self):

        path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
        return path
    
    def load_data(self,path):
        print(f"Checking directory: {path}")
        print(f"Files found: {os.listdir(path)}")
        if os.path.isdir(path):
            train_path=os.path.join(path , 'mnist_train.csv')
            test_path=os.path.join(path , 'mnist_test.csv')
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)
            else: 
                raise Exception("mnist_train.csv not found in dataset path")
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
            else: 
                raise Exception("mnist_test.csv missing")
        else:
            raise Exception("Dataset path does not exist or download failed")
        return train_df, test_df
            
    def run(self): 
        path = self.download_data()
        train_df , test_df = self.load_data(path)
        return train_df,test_df
    


        

