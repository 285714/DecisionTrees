from dt import *
from datasets import *
from plots import *

from sklearn import tree
from sklearn.model_selection import cross_validate


d = mfeat()
# d = diabetes() # <- has overfitting
# d = digits() # <- std. example

def collect_scores_avg(i, clf):
    score = cross_validate(clf, d.data, d.target, scoring=['precision_macro','recall_macro'], cv = 5,
                           return_train_score = True)
    return list(map(np.average, [i, score["fit_time"], score["score_time"], score["test_precision_macro"],
                                 score["train_precision_macro"], score["test_recall_macro"],
                                 score["train_recall_macro"]]))


scores1 = []
scores2 = []
for max_leaf_nodes in range(2, 200):
    clf1 = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    scores1.append(collect_scores_avg(max_leaf_nodes, clf1))

#   clf2 = GlobalJaccardDecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
#   scores2.append(collect_scores_avg(max_leaf_nodes, clf2))

scores1 = np.array(scores1)
# scores1 += 0.005 * np.random.random(scores1.shape)
scores2 = np.array(scores2)
# scores2 += 0.005 * np.random.random(scores2.shape)


# plt.plot(scores1[:,0], scores1[:,1], label="local entropy fit_time")
# plt.plot(scores1[:,0], scores1[:,2], label="local entropy score_time")
plt.plot(scores1[:,0], scores1[:,3], label="local entropy test_precision_macro", color="red")
plt.plot(scores1[:,0], scores1[:,4], "--", label="local entropy train_precision_macro", color="red")
# plt.plot(scores1[:,0], scores1[:,5], label="local entropy test_recall_macro")
# plt.plot(scores1[:,0], scores1[:,6], label="local entropy train_recall_macro")

# plt.plot(scores2[:,0], scores2[:,1], label="global jaccard fit_time")
# plt.plot(scores2[:,0], scores2[:,2], label="global jaccard score_time")
#  plt.plot(scores2[:,0], scores2[:,3], label="global jaccard test_precision_macro", color="blue")
#  plt.plot(scores2[:,0], scores2[:,4], "--", label="global jaccard train_precision_macro", color="blue")
# plt.plot(scores2[:,0], scores2[:,5], label="global jaccard test_recall_macro")
# plt.plot(scores2[:,0], scores2[:,6], label="global jaccard train_recall_macro")

leg = plt.legend(loc='lower right')
leg.get_frame().set_linewidth(0.0)


plt.show()
# save()
