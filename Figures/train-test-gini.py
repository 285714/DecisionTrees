from dt import *
from datasets import *
from plots import *

from sklearn import tree
from sklearn.model_selection import train_test_split


d = wine() # diabetes()
X_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.2)

collect_stats1, retrieve_stats1 = stats(X_test, y_test)
collect_stats2, retrieve_stats2 = stats(X_test, y_test)

clf1 = LocalGiniImpurityDecisionTreeClassifier(watch=collect_stats1)
clf1.fit(X_train, y_train)

clf2 = GlobalGiniImpurityDecisionTreeClassifier(global_options={"improve_only": True}, verbose=True, watch=collect_stats2)
clf2.fit(X_train, y_train)

stats1, _ = retrieve_stats1()
stats2, _ = retrieve_stats2()

# max_x = math.ceil(max(max(stats1[:,0]), max(stats2[:,0])))
# plt.gca().set_xticks(range(1, max_x+1))
# plt.tick_params(labelbottom=False)
plt.plot(stats1[:,0], stats1[:,3], '--', label="Train Local GiniImpurity", color="red")
plt.plot(stats1[:,0], stats1[:,11], label="Test Local GiniImpurity", color="red")
plt.plot(stats2[:,0], stats2[:,3], '--', label="Train Global GiniImpurity", color="blue")
plt.plot(stats2[:,0], stats2[:,11], label="Test Global GiniImpurity", color="blue")
plt.ylabel('accuracy')
leg = plt.legend(loc='lower right')
# leg.get_frame().set_linewidth(0.0)

print("local", clf1.score(X_test, y_test))
print("global", clf2.score(X_test, y_test))

save()
