#import numpy
import numpy as np
# import datasets from sklearn
from sklearn import datasets
# import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
#import train_test_split
from sklearn.model_selection import train_test_split
#import matplotlib
import matplotlib.pyplot as plt

# load digits dataset
digits = datasets.load_digits()

# load data from digits dataset
X = digits.data
# load the target data
y = digits.target

# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

knn_1 = KNeighborsClassifier()

knn_1.fit(X_train, y_train)

print('knn_1 score',knn_1.score(X_test, y_test))


neighbors = np.arange(1, 9)
print('neighbors',neighbors)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# loop
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
