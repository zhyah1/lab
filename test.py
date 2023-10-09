import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn import metrics

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images (convert 28x28 images to 1D arrays)
x_train_flattened = x_train.reshape(len(x_train), -1)
x_test_flattened = x_test.reshape(len(x_test), -1)

# Choose the number of clusters (K) based on the number of classes in MNIST
num_clusters = len(np.unique(y_train))

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(x_train_flattened)

# Get the cluster labels
predicted_labels = kmeans.labels_

# Map cluster labels to actual labels using a mapping dictionary
cluster_to_label_map = {}
for cluster_idx in range(num_clusters):
    mask = (predicted_labels == cluster_idx)
    mapped_label = np.argmax(np.bincount(y_train[mask]))
    cluster_to_label_map[cluster_idx] = mapped_label

# Assign labels to the test set using the mapping dictionary
predicted_test_labels = [cluster_to_label_map[label] for label in kmeans.predict(x_test_flattened)]

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, predicted_test_labels)
print("Classification accuracy of K-means on MNIST dataset:", accuracy)
