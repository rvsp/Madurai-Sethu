# import datasets from sklearn
from sklearn import datasets
#import matplotlib
import matplotlib.pyplot as plt

# load digits dataset
digits = datasets.load_digits()

# print dictionary keys of digits dataset
print(digits.keys())

# print the dimensions of images digits dataset
print(digits.images.shape)
# print the dimensions of digits dataset
print(digits.data.shape)

# plot the image
plt.imshow(digits.images[12], cmap=plt.cm.gray_r, interpolation='nearest')
# show the plot
plt.show()

plt.imshow(digits.images[12], cmap=plt.cm.get_cmap('Blues'), interpolation='nearest')
# show the plot
plt.show()