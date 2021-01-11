import numpy as np
import math

class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape
        
        #Initiallize initial zero p array
        p_xy = np.zeros((D, 2, 2))
        count=0
        for i in range(y.shape[0]):
            if y[i]==1:
                count +=1  
        p_y = [count/ N, 1-(count/N)]
        self.p_y = p_y
        
        #Calculate each propability using equation 6
        #Note : D is feature column, self.num_classes is class
        for i in range(2):
            for j in range(D):
                #Calculate mean and std
                feature_d = X[i==y,j]
                mean = feature_d.mean(0)
                std = np.std(feature_d, axis=0)
                p_xy[j,i] = [mean, std] 
        self.p_xy = p_xy
                

    def predict(self, X):
        N, D = X.shape
        p_xy = self.p_xy
        p_y = self.p_y
        y_pred = np.zeros(N)
        
        for i in range(N):
            prob = p_y.copy() # initialize with the p(y) terms
            for j in range(D):
                X_ij = X[i,j]
                #Negative
                mean0, std0 = p_xy[j, 0]
                #Positive
                mean1, std1 = p_xy[j, 1]
                
                naive_0 =  -(1/2 * np.square((X_ij-mean0)/std0) + np.log(std0 * math.sqrt(math.pi*2)))
                naive_1 =  -(1/2 * np.square((X_ij-mean1)/std1) + np.log(std1 * math.sqrt(math.pi*2)))
                
                prob[0] += naive_0
                prob[1] += naive_1
                
            y_pred[i] = np.argmax(prob)

        return y_pred
