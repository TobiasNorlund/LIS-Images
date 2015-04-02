# For reading and writing csv files
import csv
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sys

import numpy as np


MAX_TRAIN_SAMPLES = None # 10427

def get_features(row):
    return np.array([row]).astype(np.float)

def read_data(inpath, get_features_fun):

    print('Started loading: ' + inpath + '\n')

    X = None
    num_lines = sum(1 for line in open(inpath))
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        i = 0
        for row in reader:
            if(MAX_TRAIN_SAMPLES is not None and i > MAX_TRAIN_SAMPLES):
                break
            
            features = get_features_fun(row)
            
            if(X is None):
                X = np.empty((num_lines,features.shape[1]))
            
            X[i,:] = features

            if i % 100 == 0:
                print("\rProgress {:2.1%}".format(float(i)/float(num_lines))),

            i = i+1
    return np.atleast_2d(X)

# Define score function
def score(gtruth, pred):
    score = np.sum(gtruth != pred)/(2*gtruth.shape[0])

    print('score: ', score)
    return score
scorefun = skmet.make_scorer(score, greater_is_better=False)

def load(get_features_fun = get_features, load_val=True, load_test=True):

    X = read_data('project_data/train.csv', get_features_fun)
    X_val = read_data('project_data/validate.csv', get_features_fun) if load_val else 0
    X_test = read_data('project_data/test.csv', get_features_fun) if load_test else 0
    
    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
    Y = Y[0:MAX_TRAIN_SAMPLES].astype(np.int)

    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

    # Normalize Data
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Find one hot columns
    for i in np.arange(X.shape[1]):
        if(set(np.unique(X[:,i])) == set((0,1))):
            means[i] = 0
            stds[i] = 0

    stds[stds == 0] = 1
    
    X_norm = (X-means)/stds
    X_val_norm = (X_val-means)/stds
    X_test_norm = (X_test - means)/stds

    print('Data loaded sucessfully')
    
    return (X_norm, Y, X_val_norm, X_test_norm)
