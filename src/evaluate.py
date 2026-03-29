import numpy as np
class Evaluate:
    def __init__(self):
        pass
    def accuracy(self,preds,y):
        print("unique preds is :",np.unique(preds))
        y = np.argmax(y,axis=0)
        assert preds.shape == y.shape
        accuracy = np.mean(preds == y)       
        return accuracy 
    
    def confusion_matrix(self,preds,y):
        y = np.argmax(y,axis=0)
        confusion = np.zeros((10,10))
        for i in range(len(preds)):
            actual = y[i]
            predicted = preds[i]
            confusion[actual][predicted] +=1

        return confusion
    
    