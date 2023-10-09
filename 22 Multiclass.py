# Download any multiclass numerical dataset from UCI repostiory and do the classification with  a.SVM with majority voting


import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# Download the dataset from UCI repository
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#filename = 'iris.csv'
#urllib.request.urlretrieve(url, filename)

# Load the dataset
data = pd.read_csv('iris.csv', header=None)
 
# Prepare the dataset
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target variable (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM classifiers with different kernels
svm_model_1 = SVC(kernel='linear')
svm_model_2 = SVC(kernel='rbf')
svm_model_3 = SVC(kernel='poly')

# Create the Voting Classifier with SVM models
voting_model = VotingClassifier(estimators=[('svm1', svm_model_1), ('svm2', svm_model_2), ('svm3', svm_model_3)],
                                voting='hard')

# Fit the Voting Classifier on the training data
voting_model.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = voting_model.predict(X_test)

# Calculate the accuracy of the majority voting classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)
