import numpy as np
class ModelTraining:
    def __init__(self,layers,learning_rate=0.2):
        self.layers=layers
        self.learning_rate=learning_rate
        self.params={}
    def initialize_parameters(self):
        L=len(self.layers)
        for l in range(1,L):
            self.params['W' +str(l)]=np.random.randn(self.layers[l],self.layers[l-1]) *np.sqrt(2/self.layers[l-1])
            self.params['b' +str(l)]=np.zeros((self.layers[l],1))
            
        return self.params

    def softmax(self,Z):
        Z = Z - np.max(Z,axis=0,keepdims=True)
        exp_Z = np.exp(Z)
        sum_exp = np.sum(exp_Z,axis=0,keepdims=True)
        A=exp_Z/sum_exp
        cache =Z
        return A,cache

    
    def sigmoid(self,Z):
        A= 1/(1+np.exp(-Z))
        cache = Z
        return A , cache

    
    def relu(self,Z):
        A=np.maximum(Z,0)
        cache = Z
        return A,cache
    
    def linear_forward(self,A_prev,W,b):
        Z = np.dot(W,A_prev) + b 
        cache = (A_prev , W , b)
        return Z,cache
    
    def linear_activation_forward(self,A_prev,W,b,activation):
        Z,linear_cache = self.linear_forward(A_prev,W,b)

        if activation == 'relu':    
            A , activation_cache = self.relu(Z)
        
        elif activation == 'sigmoid':
            A , activation_cache = self.sigmoid(Z)
        elif activation == 'softmax':
            A, activation_cache = self.softmax(Z)

        cache = (linear_cache , activation_cache)
        
        return A, cache
    
    def full_linear_activation_forward (self,X):
        caches =[]
        A=X
        L = len(self.layers) -1
        for l in range(1,L):
            A_prev = A
            A ,cache = self.linear_activation_forward(A_prev, self.params['W' + str(l)] , self.params['b' + str(l)],'relu')
            caches.append(cache)
        
        AL ,cache = self.linear_activation_forward(A, self.params['W' + str(L)] , self.params['b' + str(L)],'softmax')
        caches.append(cache)
        return AL ,caches 
    
    ## Backward propogation

    def compute_cost(self , AL ,y):
        AL = np.clip(AL, 1e-15, 1-1e-15)
        m=y.shape[1]
        costs = -(1/m) * np.sum(y * np.log(AL))
        cost = np.squeeze(costs)
        return cost

    # def sigmoid_backward(self,dA ,cache):
    #     Z=cache
    #     A , _=self.sigmoid(Z)
    #     dZ = dA *A*(1-A)
    #     return dZ

    def relu_backward(self,dA,cache):
        Z=cache
        dZ=np.array(dA,copy=True)
        dZ[Z<=0] =0
        return dZ
        
    
    def linear_backward(self,dZ,cache):
        A_prev,W,b=cache
        m=A_prev.shape[1]

        dW = (1/m) * np.dot(dZ,A_prev.T)
        db = (1/m) *np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        return dA_prev,dW,db
    
    def linear_backward_activation(self,dA,cache,activation):
        linear_cache,activation_cache=cache
        if activation == 'relu':
            dZ=self.relu_backward(dA,activation_cache)
        else:
            raise Exception('Wrong activation name for linear backward')

        dA_prev,dW,db=self.linear_backward(dZ,linear_cache)
        return dA_prev,dW,db
    
    def full_backward_activation(self,AL,y,caches):
        grads={}
        L=len(caches)
        
        dZ = AL-y
        linear_cache = caches[L-1][0]
        dA_prev ,grads['dW' + str(L)],grads['db' + str(L)]= self.linear_backward(dZ,linear_cache)


        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp,dW_temp,db_temp=self.linear_backward_activation(dA_prev,current_cache,'relu')
            dA_prev =dA_prev_temp
            grads['dW' + str(l+1)]=dW_temp
            grads['db' + str(l+1)] = db_temp
          
        return grads
    
    def update_parameters(self,grads):
        L = len(self.params) //2
        for l in range(1,L+1):
            self.params['W'+str(l)]=self.params['W'+str(l)]-self.learning_rate*grads['dW' + str(l)]
            self.params['b'+str(l)]=self.params['b'+str(l)]-self.learning_rate*grads['db' + str(l)]
        return self.params
    
    def predict(self,X):
        AL ,_ =self.full_linear_activation_forward(X)
        print("AL is ",AL.min(), AL.max())
        predictions = np.argmax(AL,axis=0)
        return predictions
    
