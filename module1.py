
import h5py
import matplotlib.pyplot as plt

f = h5py.File('project_data/train.h5','r')

print(f)

img = f["data"][0].reshape((32, 64))

plot = plt.imshow(img)
plt.show()

print("hej")