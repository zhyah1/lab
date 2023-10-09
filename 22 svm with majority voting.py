
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_classifier1 = SVC(kernel='linear', C=1, probability=True)
svm_classifier2 = SVC(kernel='rbf', C=1, probability=True)
svm_classifier3 = SVC(kernel='poly', C=1, probability=True)


voting_classifier = VotingClassifier(estimators=[
    ('linear', svm_classifier1),
    ('rbf', svm_classifier2),
    ('poly', svm_classifier3)
], voting='soft')


voting_classifier.fit(X_train, y_train)


y_pred = voting_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
