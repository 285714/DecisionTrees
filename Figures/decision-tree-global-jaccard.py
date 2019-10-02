from dt import *
from datasets import *
from plots import *


clf = GlobalJaccardDecisionTreeClassifier()

d = monks()
clf.fit(d.data, d.target)

graph = clf.export_graphviz()
save(graph, "dot")

