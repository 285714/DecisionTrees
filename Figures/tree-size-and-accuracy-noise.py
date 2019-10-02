from dt import *
from plots import *
from datasets import *
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines


train_size = 0.8
sample_sizes = np.arange(0.05, 1, 0.05)
#   sample_sizes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.15, 0.175, 0.2, 0.225, 0.25,
#                   0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
n_repeat = 10 # 50

clfs = [
    LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier(improve_only=True),
    LocalGiniImpurityDecisionTreeClassifier(), GlobalGiniImpurityDecisionTreeClassifier(improve_only=True),
    LocalJaccardDecisionTreeClassifier(improve_only=True), GlobalJaccardDecisionTreeClassifier(improve_only=True),
    LocalNVIDecisionTreeClassifier(), GlobalNVIDecisionTreeClassifier(improve_only=True),
]

d, title = iris(), "iris"
X, y = d.data, d.target


def add_noise(ys, p):
    universe = np.unique(ys)
    ys_ = [ np.random.choice(universe) if np.random.rand() < p else y for y in list(ys)]
    return np.array(ys_)

def train(clf, sample_size, _):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size)
    X_sample, y_sample = rnd_sample(X_train, y_train, sample_size)
    y_sample_noise = add_noise(y_sample, 0.5)
    clf.fit(X_sample, y_sample_noise)
    return [clf.tree_.get_n_nodes(), clf.score(X_test, y_test)]
s = parallel_array(train, [clfs, sample_sizes, range(n_repeat), None])


def plot(ax, i_clf, clf, step):
    p = clf.plotting()
    tree_means = np.mean(s[i_clf, :, :, 0], axis=1)
    tree_stds = np.std(s[i_clf, :, :, 0], axis=1)
    upper = tree_means - tree_stds / 2
    lower = tree_means + tree_stds / 2
    if step == 0:
        ax.fill_between(sample_sizes, upper, lower, color=p["snd_color"], alpha=0.3)
    elif step == 1:
        ax.plot(sample_sizes, tree_means, "-.", label="Tree Size " + p["label"], color=p["snd_color"])
        ax.tick_params(axis='y')
    elif step == 2:
        accuracy_means= np.mean(s[i_clf, :, :, 1], axis=1)
        ax.plot(sample_sizes, accuracy_means, label="Accuracy " + p["label"], color=p["snd_color"])
        ax.tick_params(axis='y')


fig, axes = plt.subplots(1, len(clfs) // 2, sharex='col', sharey='row', figsize=(30, 10))
secondary_ax = []

add_legend = True

for step in [0,1,2]:
    for i_clf, clf in enumerate(clfs):
        col = i_clf // 2
        if step == 0:
            plot(axes[col], i_clf, clf, step)
            if i_clf % 2 == 0:
                secondary_ax.append(axes[col].twinx())
                if i_clf > 0:
                    secondary_ax[0].get_shared_y_axes().join(secondary_ax[0], secondary_ax[col])
            if i_clf == 0:
                axes[col].set_ylabel('Tree Size')
            elif col > 0:
                axes[col].get_yaxis().set_visible(False)
            if i_clf == len(clfs) - 1:
                secondary_ax[col].set_ylabel('Accuracy')
            elif col < (len(clfs) - 1) // 2:
                secondary_ax[col].get_yaxis().set_visible(False)
        elif step == 1:
            plot(axes[col], i_clf, clf, step)
        elif step == 2:
            plot(secondary_ax[col], i_clf, clf, step)
            if add_legend and i_clf % 2 == 1:
                p_local = clfs[i_clf - 1].plotting()
                p_global = clf.plotting()
                local_size = mlines.Line2D([], [], ls="-.", color=p_local["snd_color"], label='Tree Size ' + p_local["label"])
                local_accuracy = mlines.Line2D([], [], ls="-", color=p_local["snd_color"], label='Accuracy ' + p_local["label"])
                global_size = mlines.Line2D([], [], ls="-.", color=p_global["snd_color"], label='Tree Size ' + p_global["label"])
                global_accuracy = mlines.Line2D([], [], ls="-", color=p_global["snd_color"], label='Accuracy ' + p_global["label"])
                axes[col].legend(loc='lower center', handles=[local_size, global_size, local_accuracy, global_accuracy])

fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle(title + " (train\\_size=%g, n\\_repeat=%d)" % (train_size, n_repeat), y=0.9)
# fig.tight_layout()


"""
for i in [0, 1, 2]:
    if i == 0:
        ax1.set_xlabel('sample size')
        ax1.set_ylabel('tree size')
    elif i == 2:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('accuracy')  # we already handled the x-label with ax1

    for i_clf, clf in enumerate(clfs):
        p = clf.plotting()
        tree_means = np.mean(s[i_clf, :, :, 0], axis=1)
        tree_stds = np.std(s[i_clf, :, :, 0], axis=1)
        upper = tree_means - tree_stds / 2
        lower = tree_means + tree_stds / 2
        if i == 0:
            ax1.fill_between(sample_sizes, upper, lower, color=p["snd_color"], alpha=0.3)
        elif i == 1:
            ax1.plot(sample_sizes, tree_means, "-.", label="Tree Size " + p["label"], color=p["snd_color"])
            ax1.tick_params(axis='y')
        elif i == 2:
            accuracy_means= np.mean(s[i_clf, :, :, 1], axis=1)
            ax2.plot(sample_sizes, accuracy_means, label="Accuracy " + p["label"], color=p["snd_color"])
            ax2.tick_params(axis='y')

plt.title(title + " (train\\_size=%g, n\\_repeat=%d)" % (train_size, n_repeat))
plt.legend(loc='lower center')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
"""

save()
# plt.show()
