




# 5.)Construct a Bayesian classifier using Titanic survival prediction dataset. Calculatethe accuracy, precision,and recall for the dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pd.read_csv('Data\Titanic_dataset.csv')


data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
data = data.dropna()
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})


X = data.drop('Survived', axis=1)
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
