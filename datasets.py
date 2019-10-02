import numpy as np
from sklearn.datasets import fetch_openml, base, load_iris, load_digits, load_wine, make_classification,\
                             make_multilabel_classification, make_blobs



def artificial1():
    X = np.array([[0,0], [0,1], [1,0], [1,1], [1,2], [2,2], [2,0]])
    y = [0,0,0,1,1,0,1]
    return base.Bunch(data=X, target=y)

def artificial2():
    X = np.array([[0,0], [1,0], [1,1], [1,2], [2,2], [2,0]])
    y = [0,0,1,1,0,1]
    return base.Bunch(data=X, target=y)

def monks():
    return fetch_openml(name='monks-problems-1', version=1)

def monks2():
    return fetch_openml(name='monks-problems-2', version=1)

def monks3():
    return fetch_openml(name='monks-problems-3', version=1)

def iris():
    return load_iris()

def mfeat():
    return fetch_openml(name='mfeat-factors', version=1)

def mfeat_fourier():
    return fetch_openml(name='mfeat-fourier', version=1)

def cardiotocography():
    return fetch_openml(name='cardiotocography', version=1)

def square():
    X = [ [x,y] for x in range(10) for y in range(10) ]
    y = [ x < 3 and y < 3 for [x,y] in X ]
    return base.Bunch(data=np.array(X), target=np.array(y))

def circle():
    X = [ [x,y] for x in range(30) for y in range(30) ]
    y = [ (x-10)**2 + (y-10)**2 < 200 for [x,y] in X ]
    return base.Bunch(data=np.array(X), target=np.array(y))

def digits():
    return load_digits()

def wine():
    return load_wine()

def diabetes():
    return fetch_openml(name='diabetes', version=1)

def any():
    X, y = make_classification(n_samples=400, n_features=5,
                               n_informative=2, n_redundant=1,
                               random_state=0, shuffle=True,
                               class_sep=2)
    return base.Bunch(data=X, target=y)

def artificial_sklearn_1():
    X, y = make_blobs(n_samples=2000, n_features=3, cluster_std=[2.3,3.5,2.4], random_state=0)
    return base.Bunch(data=X, target=y)
