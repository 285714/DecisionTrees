from dt import *
from datasets import *
from plots import *
from sklearn.datasets import base


# d = circle()

X = [ [x,y] for x in range(10) for y in range(10) if x > 5 or y > 7 ]
y = [ (x-10)**2 + (y-10)**2 < 5 for [x,y] in X ]
d = base.Bunch(data=np.array(X), target=np.array(y))

clf = GlobalGiniImpurityDecisionTreeClassifier_Legacy()
fitting_evolution(clf, d, decision_surface, height=4, width=4)
save()

