
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy
import matplotlib as mpl
from sklearn.decomposition import PCA

print("Start")

f = h5py.File('project_data/train.h5','r')

X = f["data"][0:100,]

pca = PCA(n_components=2)
pca.fit_transform(X)

colors = numpy.random.rand(3,10)
color = [colors[:,i] for i in f["label"][0:100,0]]


plt.scatter(X[0,], X[1,], c=color)
plt.show()

print(pca.explained_variance_ratio_) 



# for i in range(100):
#     print f["data"][:,0].shape
#     plt.hist(f["data"][:,i], 100)
#     plt.show()
 
print("Done")