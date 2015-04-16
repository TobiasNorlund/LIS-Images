
import classification_base

(X, Y, X_val, X_test) = classification_base.load(5000, load_val=True)

# -----

import sklearn.linear_model as sklin
from sklearn import cross_validation
import numpy as np
from sklearn.decomposition import PCA

# Step 1: PCA

pca = PCA(n_components=1000)
X = pca.fit_transform(X)

print("Finished PCA. Shape of X: " + str(X.shape))

# Step 2: Logistic Regression classifier

classifier = sklin.LogisticRegression()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

classifier.fit(X_train, y_train)

print("Score: " + str(classifier.score(X_test, y_test)))

# Predict validation set
X_val = pca.transform(X_val)
Y_val = classifier.predict(X_val)
#Y_val2 = best2.predict(X_val)
np.savetxt('result_validation.txt', Y_val, fmt='%i')

print("Done")

