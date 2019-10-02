from dt import *
from joblib import Parallel, delayed
from plots import *
from datasets import *
from sklearn.model_selection import train_test_split


def rnd_sample(X, y, sample_size):
    n = X.shape[0]
    sample = np.random.choice(n, int(n * sample_size))
    return X[sample], y[sample]

def parallel_array(f, xss):
    raw = Parallel(n_jobs=-2, verbose=10)( delayed(f)(*ys) for ys in itertools.product(*xss) )
    return np.reshape(raw, list(map(len, xss)))

d, title = cardiotocography(), "cardiotocography"
X, y = d.data, d.target
sample_sizes = np.arange(0.01, 0.7, 0.03)
n_repeat = 30

clfs = [LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier(),
        LocalGainRatioDecisionTreeClassifier(), # GlobalGainRatioDecisionTreeClassifier(),
        LocalNVIDecisionTreeClassifier(), GlobalNVIDecisionTreeClassifier(),
        LocalGiniImpurityDecisionTreeClassifier(), GlobalGiniImpurityDecisionTreeClassifier(),
        LocalJaccardDecisionTreeClassifier(), GlobalJaccardDecisionTreeClassifier(),
        LocalAccuracyDecisionTreeClassifier(), GlobalAccuracyDecisionTreeClassifier()]


def train(clf, sample_size, _):
    X_sample, y_sample = rnd_sample(X, y, sample_size)
    clf.fit(X_sample, y_sample)
    return clf.tree_.get_n_nodes()
s = parallel_array(train, [clfs, sample_sizes, range(n_repeat)])

width = 0.4
for i_clf, clf in enumerate(clfs):
    p = clf.plotting()
    size_means = np.mean(s[i_clf, :, :], axis=1)
    plt.plot(sample_sizes, size_means, p["fmt"], label=p["label"], color=p["color"])

plt.title(title + " (n\\_repeat=%d)" % n_repeat)
plt.legend(loc='upper right')
save()

# plt.show()

