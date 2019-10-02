from dt import *
from joblib import Parallel, delayed
from plots import *


def rnd_tree(size, n_features=2, n_classes=2):
    if size == 1:
        label = np.random.randint(n_classes)
        return Node(None, label, is_leaf=True)

    else:
        size_left = 2 * np.random.randint(size // 2) + 1
        size_right = size - 1 - size_left
        left = rnd_tree(size_left, n_features=n_features, n_classes=n_classes)
        right = rnd_tree(size_right, n_features=n_features, n_classes=n_classes)

        feature = np.random.randint(n_features)
        threshold = np.random.rand()
        return Node(None, 0, is_leaf=False, feature=feature, threshold=threshold, left=left, right=right)

def rnd_sample(t, n_samples, n_features):
    X = np.random.rand(n_samples, n_features)
    y = t.predict(X)
    return X, y


n_samples = 500
n_features = 10
n_classes = 4
n_trees = 10

clfs = [LocalInformationGainDecisionTreeClassifier(), LocalGainRatioDecisionTreeClassifier(),
        LocalNVIDecisionTreeClassifier(), LocalGiniImpurityDecisionTreeClassifier(),
        LocalJaccardDecisionTreeClassifier(), LocalAccuracyDecisionTreeClassifier()]

t = rnd_tree(15, n_features=n_features, n_classes=n_classes)

clf_ = None
for clf in clfs:
    X, y = rnd_sample(t, n_samples, n_features)
    if clf_ is None:
        clf_ = clf
        clf_.fit(X, y)
    first_row = np.unique(y, return_counts=True)[1]
    confusion_matrix = np.zeros((1, n_classes))
    confusion_matrix[0,0:len(first_row)] = first_row
    d = clf.dist_function(clf_.tree_, confusion_matrix)
    print(d)
