#import sklearn
from sklearn import datasets
#import sklearn.neighbours
from sklearn.neighbors import KNeighborsClassifier

"""
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/classification.py
"""
"""
Algorithm - Euclidean Distance.
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

Predict Function.
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
"""
# load iris dataset from sklearn
iris = datasets.load_iris()

# declare an instance of the KNN classifier class with the value of K
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model with training data and target values
knn.fit(iris['data'], iris['target'])

# provide data whose class labels are to be predicted
X = [
    [5.9, 1.0, 5.1, 1.8],
    [3.4, 2.0, 1.1, 4.8],
]

# prints the data provided
print(X)

# store predicted class labels of X
prediction = knn.predict(X)

# print the predicted class labels of X
print(prediction)
