from dt import *
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.utils.random import sample_without_replacement
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
import plots



def accuracy_variance(clf_, d):
    X_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.5)
    scores = []

    for _ in range(100):
        clf = clone(clf_)
        X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.9)

        clf.fit(X_sample, y_sample)
        scores.append(clf.score(X_test, y_test))

        print(clf.tree_.get_depth())

#       print("mean:%f\tvar: %f" % (np.mean(scores), np.var(scores)))

    return np.mean(scores), np.var(scores)


def model_variance(clf_, d):
    X, y = d.data, d.target
    trees = []

    for _ in range(1000):
        clf = clone(clf_)
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.95)
        clf.fit(X_sample, y_sample)
        t_ = clf.tree_

        for i, (t,c) in enumerate(trees):
            if t.equals(t_):
                trees[i] = (t, c+1)
                break
        else:
            trees.append((t_, 1))

    cs = list(map(lambda x: x[1], trees))
    cs.sort(reverse=True)
    return cs


parallel = None
def bias_variance(clf_, data_gen, p_error=0, train_size=100, test_size=10000, n_classifiers=500, res=100):
    global parallel
    if parallel is None: parallel = Parallel(n_jobs=-1, verbose=10)

    def create_data(size):
        xs = []
        ys = []
        for i in range(size):
            x, y = data_gen()
            if np.random.rand() < p_error:
                y = 1 - y
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def create_clf():
        clf = clone(clf_)
        X, y = create_data(train_size)
        clf.fit(X, y)
        return clf
    clfs = parallel(delayed(create_clf)() for i in range(n_classifiers))

#   X, y = create_data(test_size)

    xs = list(itertools.product(np.arange(0.5/res, 1, 1/res), repeat=2))
    X = np.array(xs)
    y = np.array(list(map(lambda x: data_gen(x)[1], xs)))

    ys = np.array(list(map(lambda clf: clf.predict(X), clfs))).transpose()
    bias = np.power(np.mean(ys, axis=1) - y, 2)
    var = np.mean(np.power(ys, 2), axis=1) - np.power(np.mean(ys, axis=1), 2)

    dim = (res,res,1)
#   cmap = np.dstack([y.transpose().reshape(dim), bias.reshape(dim), 1 - y.reshape(dim), 1 - var.reshape(dim)])
#   cmap = np.dstack([bias.reshape(dim), np.zeros(dim), np.zeros(dim), 1 - var.reshape(dim)])
    cmap = np.dstack([bias.reshape(dim), 0.1*np.ones(dim), var.reshape(dim)])
#   plt.imshow(cmap, interpolation="nearest")

    depth, size = np.array(list(map(lambda clf: [clf.tree_.get_depth(), clf.tree_.get_n_nodes()], clfs))).transpose()
    return cmap, "bias=%f, var=%f, depth=%g, size=%g" % (np.mean(bias), np.mean(var), np.mean(depth), np.mean(size))

#   return np.mean(bias), np.mean(var)


clf1 = LocalGiniImpurityDecisionTreeClassifier()
clf2 = GlobalGiniImpurityDecisionTreeClassifier()
d = wine()

# trees = model_variance(clf1, d)
# print(trees)

def circle_data(x = None):
    [x1,x2] = np.random.rand(2) if x is None else x
    r = np.sqrt(0.5/np.pi)
    return [x1,x2], int( (x1-0.5)**2 + (x2-0.5)**2 < r**2 )

def swirl_data(x = None):
    [x1,x2] = np.random.rand(2) if x is None else x
    s1 = int(x1 * 5)
    s2 = int(x2 * 4)
    v = 1 if s1 == 0 else (0 if s1 == 4 else (1 if s2 == 0 else (0 if s2 == 3 else (1 if s1 == 3 else (0 if s1 == 1 else (1 if s2 == 2 else 0))))))
    return [x1,x2], v

def checkerboard_data(x = None):
    [x1,x2] = np.random.rand(2) if x is None else x
    return [x1,x2], (int(x1*6) + int(x2*6)) % 2

def swirl2_data(x = None):
    [x1,x2] = np.random.rand(2) if x is None else x
    s1 = int(x1 * 5)
    s2 = int(x2 * 5)
    if s1 == 2 and 2 == s2:
        _, v = swirl2_data([5 * x1 - 2, 5 * x2 - 2])
    else:
        v = int(s1 <= 1 and s2 <= 2 or s1 >=3 and s2 >= 2)
    return [x1,x2], v

def test_data(x = None):
    [x1,x2] = np.random.rand(2) if x is None else x
    s1 = int(x1 * 2)
    s2 = int(x2 * 5)
    v = int(s1 < 1 and s2 < 3 or s1 > 0 and s2 < 2)
    return [x1,x2], v


# plt.subplot(1,2,1)
# bias_variance(clf1, circle_data)
# plt.subplot(1,2,2)
# bias_variance(clf2, circle_data)
# plt.show()


hres = 100
for i, train_size in enumerate([5, 10, 20, 50]):
    for j, p in enumerate([0, 0.1, 0.3]):
        cmap1, str1 = bias_variance(clf1, swirl_data, train_size=train_size, res=2*hres, p_error=p)
        cmap2, str2 = bias_variance(clf2, swirl_data, train_size=train_size, res=2*hres, p_error=p)
    #   cmap = np.hstack((cmap1[0:hres,0:hres,:], cmap2[0:hres,hres:2*hres,:]))
        cmap = np.hstack((cmap1[:,0:hres,:], cmap2[:,hres:2*hres,:]))

        plt.subplot(4, 3, 3*i+j+1)
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.imshow(cmap, interpolation="nearest")
        plt.title("train\\_size=%d | %s\np\\_error=%g | %s" %
                  (train_size, str1, p, str2))
plt.show()

# bias, var = bias_variance(clf1, circle_data)
# print("bias: %f, var: %f" % (bias, var))
#
# bias, var = bias_variance(clf2, circle_data)
# print("bias: %f, var: %f" % (bias, var))

# m1, v1 = accuracy_variance(clf1, d)
# m2, v2 = accuracy_variance(clf2, d)
#
# print(m1, v1)
# print(m2, v2)
#
#
# plt.bar([0,1], [m1, m2], yerr=[v1, v2])
# plt.xticks([0,1], ["LocalGini", "GlobalGini"])
#
# plt.show()

