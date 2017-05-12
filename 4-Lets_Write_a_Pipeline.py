# Pipeline
## Source
# - https://www.youtube.com/watch?v=84gqSbLcBFE

import numpy as np
from sklearn import datasets

## Import dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

## Split Training / Testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

## Set Classifier
### Using decision tree classifier
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

### Using nearest neighbors classifiers classifier
from sklearn import neighbors
my_classifier = neighbors.KNeighborsClassifier()

## Train and Predict with the Classifier
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

## Measure the Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy Score: %s" % accuracy_score(y_test, predictions))
