from dt import *
from datasets import *
from plots import *


clf = LocalInformationGainDecisionTreeClassifier()

d = monks()
clf.fit(d.data, d.target)

graph = clf.export_graphviz()
save(graph, "dot")

