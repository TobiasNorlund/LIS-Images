
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.decomposition import PCA

print("Start")

f = h5py.File('project_data/train.h5','r')

# 
# X = f["data"][0:100,]
# 
# pca = PCA(n_components=2)
# pca.fit_transform(X)
# 
# colors = numpy.random.rand(3,10)
# color = [colors[:,i] for i in f["label"][0:100,0]]
# 
# 
# plt.scatter(X[0,], X[1,], c=color)
# plt.show()
# 
# print(pca.explained_variance_ratio_) 

# for i in range(100):
#     print f["data"][:,0].shape
#plt.hist(f["data"][:,0], 100)
#plt.show()
#np.set_printoptions(threshold=np.nan)
def probability_normalize(X, resolution=500):
    
    X_prob = np.zeros(X.shape)
    
    for i in range(X.shape[1]):
        hist, bins = np.histogram(X[:,i], resolution)
        inds = np.digitize(X[:,i], bins, right=True)
        inds[inds==resolution] = resolution -1
        X_prob[:,i] = hist[inds] / float(max(hist))
        #print(X_prob[:,i])
        print(str(float(i)/float(X.shape[1])) + "%")
 
    return X_prob


X_prob = probability_normalize(f["data"])

print(X_prob)

f2 = h5py.File('project_data/train_p.h5','w')
f2["data"] = X_prob
f2["label"] = f["label"][:]
#f2.flush()

print("Done")