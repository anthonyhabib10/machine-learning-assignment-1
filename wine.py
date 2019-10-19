import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

balance_data = pd.read_csv("wine.data",sep= ',', header=None)
print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)
print(balance_data.head())
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]
clf_gini = DecisionTreeClassifier(random_state = 69, max_depth=4)
clf_gini.fit(X, Y)
clf_gini.predict([[4,4,3,3]])
y_pred = clf_gini.predict(X)
print ("Accuracy is ", accuracy_score(X, y_pred)*100)
