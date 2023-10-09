

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


mnist = fetch_openml('mnist_784')
print (mnist)
X = mnist.data
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train_pca)


y_train_pred = kmeans.labels_


accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", accuracy)
