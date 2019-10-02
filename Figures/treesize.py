import numpy as np
from dt import *
from datasets import *
from sklearn.model_selection import cross_validate
from plots import read_csv, write_csv



def eval(algorithm, dataset):
    clf = {"LocalInformationGain": LocalInformationGainDecisionTreeClassifier(),
           "GlobalInformationGain": GlobalInformationGainDecisionTreeClassifier(),
           "LocalGainRatio": LocalGainRatioDecisionTreeClassifier(),
           "GlobalGainRatio": GlobalGainRatioDecisionTreeClassifier(),
           "LocalNVI": LocalNVIDecisionTreeClassifier(),
           "GlobalNVI": GlobalNVIDecisionTreeClassifier(),
           "LocalGiniImpurity": LocalGiniImpurityDecisionTreeClassifier(),
           "GlobalGiniImpurity": GlobalGiniImpurityDecisionTreeClassifier(),
           "LocalJaccard": LocalJaccardDecisionTreeClassifier(),
           "GlobalJaccard": GlobalJaccardDecisionTreeClassifier(),
           "LocalAccuracy": LocalAccuracyDecisionTreeClassifier(),
           "GlobalAccuracy": GlobalAccuracyDecisionTreeClassifier()}[algorithm]
    d = {"monks": monks,
         "iris": iris,
         "mfeat": mfeat,
         "square": square,
         "circle": circle,
         "digits": digits,
         "wine": wine,
         "diabetes": diabetes}[dataset]()

    clf.fit(d.data, d.target)
    return clf.tree.get_n_nodes()



filename = "treesize.csv"
data = read_csv(filename)
for d in data:
    algorithm = d["algorithm"]
    for dataset, v in d.items():
        if v is None:
            d[dataset] = eval(algorithm, dataset)
            print(algorithm, dataset)
            write_csv(filename, data)

