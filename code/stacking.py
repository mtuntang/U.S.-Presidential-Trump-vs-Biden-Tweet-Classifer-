import numpy as np
import utils
from random_forest import RandomForest
from knn import KNN
from naive_bayes import NaiveBayes
from random_forest import DecisionTree
from random_forest import DecisionStumpErrorRate

class Stacking():

    def __init__(self):
        self.knn = KNN(k=3)
        self.naive_bayes = NaiveBayes()
        self.random_forest = RandomForest(num_trees=15)
        self.decision_tree = DecisionTree(max_depth=np.inf, stump_class=DecisionStumpErrorRate)

    def fit(self, X, y):
        N = X.shape[0]
        self.knn.fit(X,y)
        self.naive_bayes.fit(X,y)
        self.random_forest.fit(X,y)
    
        y_pred = np.zeros((N, 3))
        y_knn = self.knn.predict(X).astype(int)
        y_naive_bayes = self.naive_bayes.predict(X)
        y_random_forest = self.random_forest.predict(X)
        
        y_pred[:, 0] = y_knn
        y_pred[:, 1] = y_naive_bayes
        y_pred[:, 2] = y_random_forest
        
        self.decision_tree.fit(y_pred, y.astype(int))

    def predict(self, X):
        N = X.shape[0]
        y_pred = np.empty((N, 3))
        
        y_knn = self.knn.predict(X)
        y_naive_bayes = self.naive_bayes.predict(X)
        y_random_forest = self.random_forest.predict(X)
        
        y_pred[:, 0] = y_knn
        y_pred[:, 1] = y_naive_bayes
        y_pred[:, 2] = y_random_forest
        
        return self.decision_tree.predict(y_pred)
