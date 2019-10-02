from dt import *
from datasets import *
from plots import *


d = circle()
clf = GlobalJaccardDecisionTreeClassifier()
fitting_evolution(clf, d, decision_surface, height=4, width=4)
save()

