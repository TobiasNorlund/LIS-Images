# For reading and writing csv files
import csv
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn
import sys
import h5py

import numpy as np

def get_features(row):
    return np.array([row]).astype(np.float)

# Define score function
def score(gtruth, pred):
    score = np.sum(gtruth != pred)/(2*gtruth.shape[0])

    print('score: ', score)
    return score
scorefun = skmet.make_scorer(score, greater_is_better=False)

def load(n_samples = None, load_val=False, load_test=False):

    X = h5py.File('project_data/train.h5','r')["data"][0:n_samples,]
    X_val = h5py.File('project_data/validate.h5', 'r')["data"] if load_val else 0
    X_test = read_data('project_data/test.csv', get_features_fun) if load_test else 0
    
    Y = sklearn.utils.column_or_1d(h5py.File('project_data/train.h5', 'r')["label"][0:n_samples,])    

    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

    # Normalize Data
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Find one hot columns
#     for i in np.arange(X.shape[1]):
#         if(set(np.unique(X[:,i])) == set((0,1))):
#             means[i] = 0
#             stds[i] = 0

    stds[stds == 0] = 1
    
    X_norm = (X-means)/stds
    X_val_norm = (X_val-means)/stds
    X_test_norm = (X_test - means)/stds

    print('Data loaded sucessfully')
    
    return (X_norm, Y, X_val_norm, X_test_norm)
