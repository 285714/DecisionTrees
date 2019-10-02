from dt import *
from joblib import Parallel, delayed
from plots import *
from datasets import *
from sklearn.model_selection import train_test_split


dsets = [("Monks 1", monks()), ("Monks 2", monks2()), ("Monks 3", monks3()), ("Iris", iris()),#("MFeat", mfeat()),
         ("Wine", wine()), ("Cardiotocography", cardiotocography()), ("Artificial SKLearn 1", artificial_sklearn_1())]
sample_size = 0.9
n_repeat = 10

clfs = [LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier()]


def train(d, clf, _):
    X_sample, y_sample = rnd_sample(d[1].data, d[1].target, sample_size)
    clf.fit(X_sample, y_sample)
    return clf.tree_.get_n_nodes()
s = parallel_array(train, [dsets, clfs, range(n_repeat)])

width = 0.4
for i_clf, clf in enumerate(clfs):
    p = clf.plotting()
    size_means = np.mean(s[:, i_clf, :], axis=1)
    size_stds = np.std(s[:, i_clf, :], axis=1)
    plt.bar(np.arange(len(dsets)) + (2 * i_clf - 1) * width / 2, size_means, width, yerr=size_stds,
            color=p["snd_color"], label=p["label"], tick_label=list(map(lambda d: d[0], dsets)))

# plt.title(title + " (n\\_repeat=%d)" % (test_size, n_repeat))
plt.legend(loc='upper right')
save()

# plt.show()

