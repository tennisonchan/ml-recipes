# https://www.youtube.com/watch?v=tNa99PG8hR8

# Goals
## - Import dataset
## - Train a classifier
## - Predict label for new flowers
## - Visualizing the decision tree

## Wiki
### https://en.wikipedia.org/wiki/Iris_flower_data_set

# Import Dataset
from sklearn.datasets import load_iris
iris = load_iris()

print('Feature Names: %s' % iris.feature_names)
print('Target Names: %s' % iris.target_names)
print('First Dataset: %s' % iris.data[0])
print('First Target: %s' % iris.target[0])

# Simple Test Dataset
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
test_idx = [0, 50, 100]

### training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

### testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier().fit(train_data, train_target)

print(clf.predict(test_data))
print(test_target)


# Viz code
### http://scikit-learn.org/stable/modules/tree.html
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("#2-iris_visualized_decision_tree.pdf")

import subprocess
subprocess.Popen("open '%s'" % "#2-iris_visualized_decision_tree.pdf", shell=True)

os.startfile("2-iris_visualized_decision_tree.pdf")
