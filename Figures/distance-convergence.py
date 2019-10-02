from dt import *
from datasets import *
from plots import *

from sklearn import tree


ds = [monks(), iris(), mfeat(), circle()]

for i, d in enumerate(ds):
    collect_stats1, retrieve_stats1 = stats()
    collect_stats2, retrieve_stats2 = stats()

    clf1 = TreeWrapper(tree.DecisionTreeClassifier, watch=collect_stats1, max_leaf_nodes=20)
    clf1.fit(d.data, d.target)

    clf2 = GlobalJaccardDecisionTreeClassifier(watch=collect_stats2, max_leaf_nodes=20)
    clf2.fit(d.data, d.target)

    stats1, _ = retrieve_stats1()
    stats2, _ = retrieve_stats2()

    ax = plt.subplot(2, 2, i+1)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.plot(stats1[:,0], 1-stats1[:,3], label="local entropy")
    plt.plot(stats2[:,0], 1-stats2[:,3], label="global jaccard")
    plt.ylabel('1-accuracy')
    plt.legend(loc='upper right')

save()
