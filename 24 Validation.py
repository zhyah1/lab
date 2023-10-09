

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.datasets import load_wine


data = load_wine()
X, y = data.data, data.target


def svm_with_cross_val(X, y):
    svm = SVC(kernel='linear')
    f1_scores = cross_val_score(svm, X, y, cv=10, scoring='f1_weighted')
    return f1_scores.mean()


def svm_with_train_test_split(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return f1_score(y_test, y_pred, average='weighted')


split_ratios = [0.8, 0.7, 0.6]
f1_scores_split = []
for ratio in split_ratios:
    f1_score_split = svm_with_train_test_split(X, y, test_size=1 - ratio)
    f1_scores_split.append(f1_score_split)


f1_score_cross_val = svm_with_cross_val(X, y)


results = {
    "Method": ["10-fold Cross-Validation"] + [f"{int(ratio*100)}:{int((1-ratio)*100)}" for ratio in split_ratios],
    "F1 Score": [f1_score_cross_val] + f1_scores_split
}

comparison_table = pd.DataFrame(results)
print(comparison_table)
