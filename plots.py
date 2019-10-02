import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


from distances import clustering_jaccard_dist

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from itertools import count

from dt import *

import os
import sys

import time
from datetime import datetime

from joblib import Parallel, delayed
import itertools


matplotlib.rcParams['axes.linewidth'] = 0.1

# plt.tick_params(labelbottom=False, labelleft=False, width=0.1, length=1)

matplotlib.rc('text', usetex=True)

# matplotlib.rc('font', family='serif')





def decision_surface(clf, dot_width=3):
    if clf.data.n_features != 2:
        raise Exception("2-dimensional data required")

    n_classes = clf.data.n_classes
    X = clf.data.X
    y = clf.data.y

    plot_colors = np.array(_color_brew(n_classes)) / 256
    cmap = ListedColormap(plot_colors)
    plot_step = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = np.array(clf.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap)

#   plt.xlabel("xxx")
#   plt.ylabel("yyy")

    for i, color in zip(range(clf.data.n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=[color], #label="ttt",
                    cmap=cmap, edgecolor='black', s=dot_width, marker="o", edgecolors="none", linewidth=0.1)

    # plt.suptitle("Decision surface of a decision tree using paired features")
    # plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    # plt.axis("off")


def fitting_evolution(clf, d, f, height=2, width=3):
    def watch(self, i):
        if i <= height * width:
            print(i)
            # ax = plt.subplot(height, width, i, aspect='equal')
            x = (i-1) % width
            y = (i-1) // width
            ax = plt.subplot(height, width, i, position=[x,-y,0.9,0.9], aspect="equal")
            ax.tick_params(labelbottom=False, labelleft=False, width=0.1, length=1)
            ax.set_xticks(np.arange(30))
            ax.set_yticks(np.arange(30))
            f(clf)

    plt.figure(figsize=(width/2, height/2))
    clf.watch = watch
    clf.fit(d.data, d.target)


def save(content = None, extension = None):
    filename = os.path.splitext(sys.argv[0])[0]
    now = datetime.now()
    rnd = now.strftime("%d-%m-%Y--%H-%M-%S")
    if content is None:
        plt.savefig(filename + '.pdf', bbox_inches = "tight", pad_inches = 0)
        plt.savefig(filename + '_backup_' + rnd + '.pdf', bbox_inches = "tight", pad_inches = 0)
    else:
        f1 = open(filename + '.' + extension, "w")
        f2 = open(filename + '_backup_' + rnd + '.' + extension, "w")
        f1.write(content)
        f2.write(content)
        f1.close()
        f2.close()
    print('plot "%s" saved (backup %s)' % (filename, rnd))


class TreeWrapper(BaseDecisionTreeClassifier):
    def __init__(self, clf_class, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, random_state = None,
                 max_leaf_nodes = None, min_dist = 0., verbose = False, watch = None, args = {}):
        self.data = None
        self.tree = None
        self.clf = None
        self.clf_class = clf_class
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_dist = min_dist
        self.verbose = verbose
        self.watch = watch
        self.args = args

    def set_classifier(self, i):
        self.clf =  self.clf_class(min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf,
                                   random_state = 0, max_leaf_nodes = i, **self.args)

    def fit(self, X, y):
        self.data = Sorted(X, y)
        self.tree = Node(self.data, None)
        self.tree.get_n_nodes = lambda: 1 if self.clf is None else len(self.clf.tree_.children_left)
        self.tree.get_depth = lambda: 0 if self.clf is None else self.clf.tree_.max_depth

        n_nodes = 0
        for i in count() if self.max_leaf_nodes is None else range(0, self.max_leaf_nodes):
            if self.watch is not None: self.watch(self, i+1)

            self.set_classifier(i+2)
            self.clf.fit(X, y)

            if n_nodes == self.tree.get_n_nodes():
                break
            else:
                n_nodes = self.tree.get_n_nodes()
        return self

    def predict(self, X):
        if self.clf is None:
            return np.array([self.data.y_original[0]] * len(X))
        else:
            return self.clf.predict(X)


def stats(X_test = None, y_test = None):
    stats = []

    def collect_stats(self, i):
        column = []
        for X, y in [(self.data_.X, self.data_.y_original)] + ([(X_test, y_test)] if X_test is not None else []):
            y_ = self.predict(X)
            accuracy = accuracy_score(y, y_)
            jaccard = clustering_jaccard_dist(y, y_)
            precision = precision_score(y, y_, average='macro')
            recall = recall_score(y, y_, average='macro')
            f1 = f1_score(y, y_, average='macro')
            n_nodes = self.tree_.get_n_nodes()
            depth = self.tree_.get_depth()
            column.extend([i, n_nodes, depth, accuracy, jaccard, precision, recall, f1])
        stats.append(column)

    def retrieve_stats():
        return np.array(stats), ["..."] # TODO descriptions

    return collect_stats, retrieve_stats


def timing(clf, X, y):
    now = time.process_time()
    clf.fit(X, y)
    return time.process_time() - now



def read_csv(filename):
    def strToNum(s):
        try:
            return float(s)
        except:
            s = s.strip()
            return None if s == "-" else s

    dataset = np.loadtxt(filename, delimiter=',', dtype=str)
    data = []
    headline = None

    for line in dataset:
        if headline is None:
            headline = list(map(lambda s: s.strip(), list(line)))
        else:
            data.append(dict(zip(headline, map(strToNum, line))))
    return data


def write_csv(filename, data):
    def NumToStr(n):
        if n is None:
            return "-"
        return str(n)

    headline = None
    lengths = None
    out = ""

    for d in data:
        if headline is None:
            headline = list(d.keys())
            lengths = list(map(len, headline))
        lengths = [ max(lengths[i], len(s)) for i,s in enumerate(map(NumToStr, d.values())) ]

    last = len(lengths) - 1
    for i, s in enumerate(data[0].keys()):
        out += s + ("" if i == last else "," + " " * (1 + lengths[i] - len(s)))

    for d in data:
        out += "\n"
        for i, n in enumerate(d.values()):
            s = NumToStr(n)
            out += s + ("" if i == last else "," + " " * (1 + lengths[i] - len(s)))

    with open(filename, "w") as f:
        f.write(out)
    return out


def rnd_sample(X, y, sample_size):
    n = X.shape[0]
    sample = np.random.choice(n, int(np.floor(n * sample_size)))
    return X[sample], y[sample]


def parallel_array(f, xss, matrix=True):
    xss_ = list(filter(lambda xs: not(xs is None), xss))
    print(" [%d tasks ahead] " % np.product(list(map(len, xss_))))
    raw = Parallel(n_jobs=-2, verbose=10)(delayed(f)(*ys) for ys in itertools.product(*xss_) )
    shape = list(map(lambda xs: -1 if xs is None else len(xs), xss))
    if matrix:
        return np.reshape(raw, shape)
    else:
        return np.array(raw, dtype=object).reshape(shape)


def train_stats(X_test, y_test):
    stats = []

    def collect_stats(self, i):
        train_accuracy = self.score(self.data_.X, self.data_.y_original)
        test_accuracy = self.score(X_test, y_test)
        n_nodes = self.tree_.get_n_nodes()
        depth = self.tree_.get_depth()
        stats.append([i, n_nodes, depth, train_accuracy, test_accuracy])

    def retrieve_stats():
        return np.array(stats)

    return collect_stats, retrieve_stats


