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



# sample_sizes = np.arange(0.1, 1.0, 0.1)
sample_sizes = np.arange(0.1, 1.1, 0.1)
n_repeat = 10
test_size = 0.9 # <-

d, title = artificial_sklearn_1(), "artificial sklearn 1"
X, y = d.data, d.target

clfs = [LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier(),
        LocalGainRatioDecisionTreeClassifier(), GlobalGainRatioDecisionTreeClassifier(),
        LocalNVIDecisionTreeClassifier(), GlobalNVIDecisionTreeClassifier(),
        LocalGiniImpurityDecisionTreeClassifier(), GlobalGiniImpurityDecisionTreeClassifier(),
        LocalJaccardDecisionTreeClassifier(), GlobalJaccardDecisionTreeClassifier(),
        LocalAccuracyDecisionTreeClassifier(), GlobalAccuracyDecisionTreeClassifier()]

def train(_, sample_size, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_sample, y_sample = rnd_sample(X_train, y_train, sample_size)
    clf.fit(X_sample, y_sample)
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-sample_size)
#   clf.fit(X_train, y_train)
    return clf.score(X_train, y_train)
s = parallel_array(train, [range(n_repeat), sample_sizes, clfs]) # (n_repeat, #sample_sizes, #clfs)

for i_clf, clf in enumerate(clfs):
    p = clf.plotting()
    means_accuracy = np.mean(s[:, :, i_clf], axis=0)
    plt.plot(sample_sizes, means_accuracy, p["fmt"], label=p["label"], color=p["color"])

plt.title(title + " (test\\_size=%g, n\\_repeat=%d)" % (test_size, n_repeat))
plt.legend(loc='lower right')
save()

