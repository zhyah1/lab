



import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv(r"Data\1000_features.csv")


X = data.drop(data.columns[0], axis=1)
y = data[data.columns[0]]

pca_dims = [300, 400, 500]


results = []
for dim in pca_dims:

    pca = PCA(n_components=dim)

    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)


    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')


    results.append({'Dimension': dim, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})


results_df = pd.DataFrame(results)
print(results_df)

