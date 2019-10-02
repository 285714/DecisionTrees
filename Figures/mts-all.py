from dt import *
from plots import *
import matplotlib.lines as mlines
from sklearn.base import clone


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


n_samples = 500 # 500
n_features = 6 # 6
n_classes = 4 # 4

tree_sizes = np.arange(3,25,2)
trees_per_size = 300 # 100

clfs = [
        LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier(),
        LocalGainRatioDecisionTreeClassifier(), GlobalGainRatioDecisionTreeClassifier(),
        LocalNVIDecisionTreeClassifier(), GlobalNVIDecisionTreeClassifier(),
        LocalGiniImpurityDecisionTreeClassifier(), GlobalGiniImpurityDecisionTreeClassifier(),
        LocalJaccardDecisionTreeClassifier(), GlobalJaccardDecisionTreeClassifier(),
        LocalAccuracyDecisionTreeClassifier(), GlobalAccuracyDecisionTreeClassifier(),
       ]

def train(size, _):
    clf_sizes = np.zeros(len(clfs))
    while True:
        t = rnd_tree(size, n_features=n_features, n_classes=n_classes)
        X, y = rnd_sample(t, n_samples=n_samples, n_features=n_features)
        for i_clf, clf_ in enumerate(clfs):
            clf = clone(clf_)
            clf.max_leaf_nodes = 60
            clf.fit(X, y)
            s = clf.tree_.get_n_nodes()
            if s < size:
                break
            clf_sizes[i_clf] = s
        else:
            break
    return clf_sizes

s = parallel_array(train, [tree_sizes, range(trees_per_size), None])

# ...?

def plot(ax, i_clf, clf, step):
    p = clf.plotting()
    s_clf = s[:,:,i_clf].reshape((len(tree_sizes), -1))
    tree_means = np.mean(s_clf, axis=1)
    tree_stds = np.std(s_clf, axis=1)
    upper = tree_means - tree_stds / 2
    lower = tree_means + tree_stds / 2
    if step == 0:
        ax.fill_between(tree_sizes, upper, lower, color=p["snd_color"], alpha=0.3)
    elif step == 1:
        ax.plot(tree_sizes, tree_sizes, color="black")
        ax.plot(tree_sizes, tree_means, label=p["label"], color=p["snd_color"])


w = max(2, int(np.ceil(len(clfs) / 4)))
h = 2
fig, axes = plt.subplots(h, w, sharex='col', sharey='row', figsize=(30, 10))

for step in [0,1]:
    for i_clf, clf in enumerate(clfs):
        i = i_clf // h
        col = i % w
        row = i // w
        if step == 0:
            plot(axes[row,col], i_clf, clf, step)
            if col == 0 and i_clf % 2 == 0:
                axes[row,col].set_ylabel('Tree Size')
                axes[row,col].set_ylim([0, 60])
            elif col > 0:
                axes[row,col].get_yaxis().set_visible(False)
            if row == 0:
                axes[row,col].get_xaxis().set_visible(False)
        elif step == 1:
            plot(axes[row,col], i_clf, clf, step)
            p_local = clfs[i_clf - 1].plotting()
            p_global = clf.plotting()
            local_line = mlines.Line2D([], [], ls="-", color=p_local["snd_color"], label=p_local["label"])
            global_line = mlines.Line2D([], [], ls="-", color=p_global["snd_color"], label=p_global["label"])
            axes[row,col].legend(loc='lower right', handles=[local_line, global_line])



fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle("trees (n\\_features=%d, n\\_classes=%d) | training (n\\_samples=%d, n\\_trees\\_per\\_size=%d)" % (n_features, n_classes, n_samples, trees_per_size), y=0.9)
save()


