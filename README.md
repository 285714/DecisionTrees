
# Global Evaluation For Decision Tree Learning

The purpose of this project is to experiment with different splitting strategies
for decision tree learning. Most commonly, algorithms for decision tree learning
split a node such that the two newly formed groups of data points in the node
have minimum distance to the ground truth labeling, with respect to some pre
defined distance measure. Instead, one could split a node such that the
partitioning induced by the resulting tree has minimum distance to the ground
truth labeling of the whole data set. The code here is to evaluate both
approaches under different distance measures.

## Implementation

The main implementation (and documentation) can be found in [dt.py](dt.py) and
integrates into [https://scikit-learn.org/stable/](Sklearn).

A sample invocation can be found below.

```
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score

# create artificial data
iris = load_iris()
# X, y = make_blobs(n_samples=2000, n_features=3, cluster_std=[2.3,3.5,2.4], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

clf = GlobalInformationGainDecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(f"accuracy={accuracy_score(y_test, y_predict)}")

# we can even use any sklearn function on classifiers
score = cross_validate(clf, iris.data, iris.target, scoring=['accuracy'], cv = 5)
print(f"accuracy(cv)={score['test_accuracy']}")
```

Besides the `GlobalInformationGainDecisionTreeClassifier`, the following classes
using different splitting criteria are available.

Similarity/Distance Measure | Local Evaluation | Global Evaluation |
| -- | -- | -- |
| Information Gain | `LocalInformationGainDecisionTreeClassifier` | `GlobalInformationGainDecisionTreeClassifier`
| Gain Ratio | `LocalGainRatioDecisionTreeClassifier` | `GlobalGainRatioDecisionTreeClassifier`
| normalized Variation of Information | `LocalNVIDecisionTreeClassifier` | `GlobalNVIDecisionTreeClassifier`
| Gini Impuriy | `LocalGiniImpurityDecisionTreeClassifier` | `GlobalGiniImpurityDecisionTreeClassifier`
| Jaccard | `LocalJaccardDecisionTreeClassifier` | `GlobalJaccardDecisionTreeClassifier`
| Accuracy | `LocalAccuracyDecisionTreeClassifier` | `GlobalAccuracyDecisionTreeClassifier`

## Experiments

Figures for evaluation purposes can be created by the scripts found
in the [Figures/](Figures folder). Datsets used for evaluation are loaded
in [datasets.py](datasets.py) and plots created through [plots.py](plots.py)


