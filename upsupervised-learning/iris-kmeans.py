from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading dataset
iris_df = datasets.load_iris()

# target names
print(iris_df.target_names)
# Dataset Slicing
x_axis = iris_df.data[:, 0]		# sepal length
y_axis = iris_df.data[:, 2]		# sepal width

# Declaring model
model = KMeans(n_clusters=3)

# Fitting model
model.fit(iris_df.data)

# prediction on the entire data
all_predictions = model.predict(iris_df.data)

# plot the predictions of the model
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.show()