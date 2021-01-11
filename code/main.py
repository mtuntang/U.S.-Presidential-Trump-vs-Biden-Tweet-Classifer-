# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd


# our code
import utils

from knn import KNN
from naive_bayes import NaiveBayes
from kmeans import Kmeans
from random_forest import RandomForest
from stacking import Stacking

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    
    #training set
    wordvec_train = pd.read_csv("..\data\wordvec_train.csv").to_numpy()
    n, d = wordvec_train.shape
    #print(n,d)
    X_train = wordvec_train[:, 0:d-1].astype(np.float)
    #print("X_train shape", X_train.shape)
    y_train = wordvec_train[:, d-1].astype(np.int)
    #print("y_train shape", y_train.shape)
   # print(y_train)
        
    #test set
    wordvec_test = pd.read_csv("..\data\wordvec_test.csv").to_numpy()
    N,D = wordvec_test.shape
    #print(N,D)
    X_test = wordvec_test[:, 0:D-1].astype(np.float)
    #print("X_test shape", X_test.shape)
    y_test = wordvec_test[:, D-1].astype(np.int)
    #print("y_test shape", y_test.shape)

    if question == "1":
        #KNN
        model = KNN(3)
        utils.evaluate_model(model, X_train, y_train, X_test, y_test)
    
    elif question == "2":
        #Naive Bayes
        model = NaiveBayes()
        utils.evaluate_model(model, X_train, y_train, X_test, y_test)
        
    elif question == "3":
        #Random Forest
        model = RandomForest(15, np.inf)
        utils.evaluate_model(model, X_train, y_train, X_test, y_test)
    
    elif question == "4":
        #Stacking
        model = Stacking()
        utils.evaluate_model(model, X_train, y_train, X_test, y_test)
        