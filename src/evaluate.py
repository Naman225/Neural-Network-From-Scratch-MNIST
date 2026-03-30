import numpy as np
import matplotlib.pyplot as plt
import os
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
    
    def get_wrong_predictions(self,preds,y):
        preds = preds.flatten()
        y=y.flatten()
        wrong_predictions= []
        index = len(preds)
        for i in range(index):
            if preds[i] != y[i]:
                info = {
                    'index' : i,
                    'pred' : preds[i],
                    'true' : y[i]
                }
                wrong_predictions.append(info)
        return wrong_predictions
    
    def analyze_patterns(self,error,X):
        error_img = []
        for i in range(len(error)):
            image = X[:,error[i]['index']]
            images = image.reshape(28,28)
            error_img.append({
                'index':error[i]['index'],
                'error_imgs':images,
                'predicted' : error[i]['pred'],
                'true' : error[i]['true']
                })
        return error_img
    
    def create_folder(self):
        folder_name = 'error_prone_images'
        os.makedirs(folder_name ,exist_ok=True)
        return folder_name


    def visualize_error(self,error_img,num_samples=10):
        folder = self.create_folder()
        for i in range(min(num_samples,len(error_img))):
            image = error_img[i]['error_imgs']
            pred = error_img[i]['predicted']
            true = error_img[i]['true']
            plt.imshow(image , cmap='gray')
            plt.title(f"Pred: {pred} | True: {true}")
            plt.axis('off')
            filename = f"{folder}/error_{i}_pred_{pred}_true_{true}.png"
            plt.savefig(filename)
            plt.close()

    def add_confidence(self ,AL,error_img):
        for i in range(len(error_img)):
            confidence = np.max(AL[:,error_img[i]['index']])
            error_img[i]['confidence'] = confidence

        error_img.sort(key=lambda x: x['confidence'] ,reverse=True)
        return  error_img
    
    def visualize_error_with_confidence(self,error_img,num_samples=10):
        folder = self.create_folder()
        for i in range(min(num_samples, len(error_img))):
            image = error_img[i]['error_imgs']
            pred = error_img[i]['predicted']
            true = error_img[i]['true']
            confidence = error_img[i]['confidence']
            plt.imshow(image , cmap='gray')
            plt.title(f"Pred: {pred} | True: {true} | Conf: {confidence:.2f}")
            plt.axis('off')
            filename = f"{folder}/error_{i}_pred_{pred}_true_{true}_conf_{confidence:.2f}.png"
            plt.savefig(filename)
            plt.close()

