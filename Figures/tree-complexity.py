from dt import *
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.utils.random import sample_without_replacement
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
from plots import *
from sklearn.linear_model import LinearRegression



def sample(X, y, sample_size=0.5):
    n = X.shape[0]
    if sample_size <= 1: sample_size = int(sample_size * n)
    s = np.random.choice(n, sample_size)
    return X[s], y[s]


def train(clf_, X, y, test_size=0.1, sample_size=0.9):
    X_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=test_size)
    X_sample, y_sample = sample(X_train, y_train, sample_size=sample_size)
    clf = clone(clf_)
    clf.fit(X_sample, y_sample)
    return (type(clf_).__name__, clf.tree_.get_n_nodes(), clf.score(X_test, y_test))


def print_matrix(xs, ys, cells, agg_):
    def agg(cells):
        cells = list(cells)
        return "" if len(cells) == 0 else agg_(cells)

    rows = ["" for _ in xs] + [""]*2
    def add_row(new_rows):
        nonlocal rows
        l = max(map(len, new_rows))
        rows = [ rows[i] + "  " + r + " "*(l - len(r)) for i, r in enumerate(new_rows) ]

    new_rows = [""] + xs + [""]
    add_row(new_rows)

    for y in ys:
        new_rows = [str(y)]
        for x in xs:
            new_rows.append(agg(list(filter(lambda c: c[0] == x and c[1] == y, cells))))
        new_rows.append(agg(filter(lambda c: c[1] == y, cells)))
        add_row(new_rows)

    new_rows = [""]
    for x in xs:
        new_rows.append(agg((filter(lambda c: c[0] == x, cells))))
    new_rows.append(agg(cells))
    add_row(new_rows)

    print("\n".join(rows))


def show_line(xs, ys, cells, agg_):
    width = 0.35

    def agg(cells):
        cells = list(cells)
        return (0, [None, None]) if len(cells) == 0 else agg_(cells)

    liness = []
    for i, x in enumerate(xs):
        lines = []
        boxes = []
        for y in ys:
            l, b = agg(list(filter(lambda c: c[0] == x and c[1] == y, cells)))
            lines.append([y, l])
            if b[0] is not None:
                boxes.append([y] + b)
        liness.append(lines)

        boxes = np.array(boxes)
        plt.bar(boxes[:,0] + (2*i-1) * width/2, boxes[:,1], width, yerr=boxes[:,2], label=x)

    for lines in liness:
        lines = np.array(lines)
        plt.plot(lines[:,0], lines[:,1])
        plt.plot(lines[:,0], lines[:,1], linewidth=0.5, color="black")

    sizes = list(map(lambda c: c[1], cells))
    test = np.array([min(sizes), max(sizes)])
    for x in xs:
        points = np.array(list(map(lambda c: (c[1], c[2]), filter(lambda c: c[0] == x, cells))))
        reg = LinearRegression().fit(points[:,0].reshape(-1,1), points[:,1])
        plt.plot(test, reg.predict(test.reshape(-1,1)), linewidth=0.5, label=x)
    points = np.array(list(map(lambda c: (c[1], c[2]), cells)))
    reg = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])
    plt.plot(test, reg.predict(test.reshape(-1, 1)), color="black", linewidth=0.5)

    plt.legend(loc='center right')

# d, dataset_name = iris(), "iris"
# d, dataset_name = wine(), "wine"
d, dataset_name = any(), "artificial (5 features, 400 samples)"
# d, dataset_name = monks3(), "monks3"




sample_size = 0.9
n_classifiers = 10000
for i, test_size in enumerate([0.1, 0.5, 0.8, 0.9]):
    plt.figure()

    clfs = [LocalGiniImpurityDecisionTreeClassifier(), GlobalGiniImpurityDecisionTreeClassifier()]
#          [LocalGainRatioDecisionTreeClassifier(), GlobalGainRatioDecisionTreeClassifier()]
#          [LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier()]
    infos = []
    for clf in clfs:
        infos.extend(Parallel(n_jobs=-2, verbose=10)(
            delayed(train)(clf, d.data, d.target, test_size=test_size, sample_size=sample_size) for i in range(n_classifiers)))
#       print(".", end="")

#   sizes = [1,3,5,7,9,11,13,15,17,19,21,23,25]
    sizes = list(np.unique(list(map(lambda info: info[1], infos))))
    sizes.sort()
    clfs = list(map(lambda clf: type(clf).__name__, clfs))

    print("test_size = %g\nsample_size = %g (fixed)\n(mixed training of %d classifiers)" % (test_size, sample_size, n_classifiers))
    plt.title("%s\ntest\\_size = %g\nsample\\_size = %g (fixed)\n(mixed training of %d classifiers)" % (dataset_name, test_size, sample_size, n_classifiers))

    print_matrix(clfs, sizes, infos, lambda infos: "%5d / %.3f" % (len(infos), np.mean(list(map(lambda i: i[2], infos)))))
    show_line(clfs, sizes, infos, lambda infos: (len(infos) / n_classifiers, [np.mean(list(map(lambda i: i[2], infos))), np.std(list(map(lambda i: i[2], infos)))]))

    save()

