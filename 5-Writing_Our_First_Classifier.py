# Write Our First Classifier
## https://www.youtube.com/watch?v=AoeEHqVSNOw

### 1 - Comment out imports
### 2 - Implement a class
### 3 - Understand interface
### 4 - Get pipeline working
### 5 - Intro to k-NN
### 6 - Measure distance
### 7 - implement the algorithm
### 8 - Run pipeline

## Pros
### - Relatively simple

## Cons
### - Computationally intensive
### - Hard to represent the relations between features

import random
from scipy.spatial import distance

def euc(a, b):
  return distance.euclidean(a, b)

class ScrappyKNN():
  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

  def predict(self, X_test):
    predictions = []
    for row in X_test:
      # label = random.choice(self.y_train)
      label = self.closet(row)
      predictions.append(label)

    return predictions

  def closet(self, row):
    best_dist = euc(row, self.X_train[0])
    best_index = 0
    for index in range(1, len(self.X_train)):
      dist = euc(row, self.X_train[index])
      if best_dist > dist:
        best_dist = dist
        best_index = index
    return self.y_train[best_index]

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
# from sklearn import neighbors
# my_classifier = neighbors.KNeighborsClassifier()

my_classifier = ScrappyKNN();

## Train and Predict with the Classifier
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

## Measure the Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy Score: %s" % accuracy_score(y_test, predictions))
