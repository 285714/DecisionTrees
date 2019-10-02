import numpy as np
from dt import *
from datasets import *
from sklearn.model_selection import cross_validate
from plots import read_csv, write_csv




# LocalInformationGainDecisionTreeClassifier
# GlobalInformationGainDecisionTreeClassifier
# LocalGainRatioDecisionTreeClassifier
# GlobalGainRatioDecisionTreeClassifier
# LocalNVIDecisionTreeClassifier
# GlobalNVIDecisionTreeClassifier
# LocalGiniImpurityDecisionTreeClassifier
# GlobalGiniImpurityDecisionTreeClassifier
# LocalJaccardDecisionTreeClassifier
# GlobalJaccardDecisionTreeClassifier
# LocalAccuracyDecisionTreeClassifier
# GlobalAccuracyDecisionTreeClassifier

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
#   clf = {"LocalGini": tree.DecisionTreeClassifier(),
#          "LocalEntropy": tree.DecisionTreeClassifier(criterion="entropy"),
#          "GlobalJaccard": GlobalJaccardDecisionTreeClassifier(),
#          "LocalJaccard": LocalJaccardDecisionTreeClassifier(),
#          "GlobalJaccardEntropy": GlobalGenericJaccardDecisionTreeClassifier(f="entropy", max_leaf_nodes=100),
#          "GlobalJaccardGini": GlobalGenericJaccardDecisionTreeClassifier(f="gini", max_leaf_nodes=100)}[algorithm]
    d = {"monks": monks,
         "iris": iris,
         "mfeat": mfeat,
         "square": square,
         "circle": circle,
         "digits": digits,
         "wine": wine,
         "diabetes": diabetes}[dataset]()

    score = cross_validate(clf, d.data, d.target, scoring=['precision_macro','recall_macro'], cv = 5,
                           return_train_score = True)
    score = ({ k: round(np.average(v), 5) for k,v in score.items() })
    return score["test_precision_macro"]
      # score["fit_time"], score["score_time"], score["test_precision_macro"], score["train_precision_macro"], score["test_recall_macro"], score["train_recall_macro"]



filename = "eval.csv"
data = read_csv(filename)
for d in data:
    algorithm = d["algorithm"]
    for dataset, v in d.items():
        if v is None:
            d[dataset] = eval(algorithm, dataset)
            print(algorithm, dataset)
            write_csv(filename, data)

