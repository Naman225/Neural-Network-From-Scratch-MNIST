import os
import pickle

def save_object(path,obj):
    dir_path=os.path.dirname(path)
    os.makedirs(dir_path,exist_ok=True)

    with open(path,'wb') as file:
        pickle.dump(obj,file)

def load_path(path):
    with open(path,'rb') as file:
        obj = pickle.load(file)
        return obj
            
    
