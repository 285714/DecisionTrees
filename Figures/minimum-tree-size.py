from dt import *
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
n_features = 6
n_classes = 4
n_trees = 10

sizes = np.arange(3,25,2)
clfs = [LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier(),
        LocalGainRatioDecisionTreeClassifier(), # GlobalGainRatioDecisionTreeClassifier(),
        LocalNVIDecisionTreeClassifier(), GlobalNVIDecisionTreeClassifier(),
        LocalGiniImpurityDecisionTreeClassifier(), GlobalGiniImpurityDecisionTreeClassifier(),
        # LocalJaccardDecisionTreeClassifier(),
        GlobalJaccardDecisionTreeClassifier(),
        # LocalAccuracyDecisionTreeClassifier(), GlobalAccuracyDecisionTreeClassifier()
       ]

def train_for_size(size):
    all_sizes = np.zeros((len(clfs), n_trees))
    for i_tree in range(n_trees):
        while True:
            t = rnd_tree(size, n_features=n_features, n_classes=n_classes)
            X, y = rnd_sample(t, n_samples=n_samples, n_features=n_features)
            for i_clf, clf in enumerate(clfs):
                clf.fit(X, y)
                s = clf.tree_.get_n_nodes()
                if s < size:
                    break
                all_sizes[i_clf, i_tree] = s
            else:
                break
    return all_sizes
all_sizes = np.moveaxis(np.array(Parallel(n_jobs=-2, verbose=10)(delayed(train_for_size)(size) for size in sizes)), 0, 1)


plt.plot(sizes, sizes, color="black")
for i_clf, clf in enumerate(clfs):
    p = clf.plotting()
    tree_means = np.mean(all_sizes[i_clf, :, :], axis=1)
    plt.plot(sizes, tree_means, p["fmt"], label=p["label"], color=p["color"])

plt.legend(loc='lower right')
save()

