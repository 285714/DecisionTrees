from dt import *
from datasets import *
from plots import *

from sklearn import tree
from sklearn.model_selection import train_test_split


d = digits()
X_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.3)

collect_stats1, retrieve_stats1 = stats(X_test, y_test)
collect_stats2, retrieve_stats2 = stats(X_test, y_test)

clf1 = TreeWrapper(tree.DecisionTreeClassifier, watch=collect_stats1)
clf1.fit(X_train, y_train)

clf2 = GlobalJaccardDecisionTreeClassifier(watch=collect_stats2, min_dist=0.1, max_leaf_nodes=100)
# clf2 = GlobalGenericDecisionTreeClassifier(watch=collect_stats2, max_leaf_nodes=60)
clf2.fit(X_train, y_train)

stats1, _ = retrieve_stats1()
stats2, _ = retrieve_stats2()

max_x = math.ceil(max(max(stats1[:,0]), max(stats2[:,0])))
plt.gca().set_xticks(range(1, max_x+1))
plt.tick_params(labelbottom=False)
plt.plot(stats1[:,0], stats1[:,4], '--', label="train local entropy", color="red")
plt.plot(stats1[:,0], stats1[:,12], label="test local entropy", color="red")
plt.plot(stats2[:,0], stats2[:,4], '--', label="train global jaccard", color="blue")
plt.plot(stats2[:,0], stats2[:,12], label="test global jaccard", color="blue")
plt.ylabel('accuracy')
leg = plt.legend(loc='upper right')
leg.get_frame().set_linewidth(0.0)

print("entropy", clf1.score(X_test, y_test))
print("jaccard", clf2.score(X_test, y_test))

save()
