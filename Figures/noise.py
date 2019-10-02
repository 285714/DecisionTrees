from dt import *
from datasets import *
from plots import *

from sklearn.metrics import *
from sklearn.model_selection import KFold


def add_noise(y, p):
    values = np.unique(y)
    for i in range(len(y)):
        if np.random.rand() < p:
            y[i] = np.random.choice(y,1)[0]
    return y


def eval(clf, p):
    kf = KFold(n_splits=5)
    acc_train = []
    acc_test = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = add_noise(y_train, p)

        clf.fit(X_train, y_train)
        y_train_ = clf.predict(X_train)
        y_test_ = clf.predict(X_test)

        acc_train.append(accuracy_score(y_train, y_train_))
        acc_test.append(accuracy_score(y_test, y_test_))

    return np.mean(acc_train), np.mean(acc_test)


d = wine()
X = d.data
y = d.target


# clfs = [LocalGiniImpurityDecisionTreeClassifier(max_depth=5), GlobalJaccardDecisionTreeClassifier(max_depth=5)]
clfs = [LocalInformationGainDecisionTreeClassifier(),
        GlobalJaccardDecisionTreeClassifier(global_options={"improve_only": True})]
ps = np.arange(0, 1, 0.1)
accs_train = np.zeros((len(clfs), len(ps)))
accs_test = np.zeros((len(clfs), len(ps)))

for j, p in enumerate(ps):
    for i, clf in enumerate(clfs):
        acc_train, acc_test = eval(clf, p)
        accs_train[i,j] = acc_train
        accs_test[i,j] = acc_test
        print(j, i)


colors = ["red", "blue"]
desc = ["Local GiniImpurity", "Global Jaccard"]
plt.tick_params(labelbottom=False)
for i in range(len(clfs)):
    plt.plot(ps, accs_train[i,:], "--", label="Train " + desc[i], color=colors[i])
    plt.plot(ps, accs_test[i,:], label="Test " + desc[i], color=colors[i])
plt.ylabel('accuracy')
leg = plt.legend(loc='lower right')
leg.get_frame().set_linewidth(0.0)

save()



