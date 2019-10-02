# from dt import *
from datasets import *
from plots import *
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from sklearn.base import clone
from scipy.stats import mode
import itertools


def find_contiguous_colors(colors):
  # finds the continuous segments of colors and returns those segments
  segs = []
  curr_seg = []
  prev_color = ''
  for c in colors:
    if c == prev_color or prev_color == '':
      curr_seg.append(c)
    else:
      segs.append(curr_seg)
      curr_seg = []
      curr_seg.append(c)
    prev_color = c
  segs.append(curr_seg) # the final one
  return segs

def plot_multicolored_lines(x,y,colors):
  segments = find_contiguous_colors(colors)
  plt.figure()
  start= 0
  for seg in segments:
    end = start + len(seg)
    l, = plt.gca().plot(x[start:end],y[start:end],lw=2,c=seg[0])
    start = end


def add_noise(ys, p):
    universe = np.unique(ys)
    ys_ = [ np.random.choice(universe) if np.random.rand() < p else y for y in list(ys)]
    return np.array(ys_)



train_size = 0.8
noise_p = 0.5
n_repeat = 1000



clfs = [
#   LocalNVIDecisionTreeClassifier(),
#   GlobalNVIDecisionTreeClassifier(),
#   GlobalNVIDecisionTreeClassifier(glocal=True),

#   LocalInformationGainDecisionTreeClassifier(),
    GlobalInformationGainDecisionTreeClassifier(),
    LocalInformationGainDecisionTreeClassifier(max_leaf_nodes=5),
    GlobalInformationGainDecisionTreeClassifier(max_leaf_nodes=5),

#   GlobalJaccardDecisionTreeClassifier(),
#   LocalJaccardDecisionTreeClassifier(),
#   GlobalJaccardDecisionTreeClassifier(glocal=True),

#   LocalAccuracyDecisionTreeClassifier(),
#   GlobalAccuracyDecisionTreeClassifier(),

#   LocalGiniImpurityDecisionTreeClassifier(improve_only=True),
#   GlobalGiniImpurityDecisionTreeClassifier(improve_only=True),

#   LocalGiniImpurityDecisionTreeClassifier(),
#   GlobalGiniImpurityDecisionTreeClassifier(),

#   LocalGainRatioDecisionTreeClassifier(),
#   GlobalGainRatioDecisionTreeClassifier(),

#   LocalNVIDecisionTreeClassifier(),
#   GlobalNVIDecisionTreeClassifier(),

#   LocalNVIDecisionTreeClassifier(min_samples_split=30),
#   GlobalNVIDecisionTreeClassifier(min_samples_split=30),

#   LocalJaccardDecisionTreeClassifier(),
#   GlobalJaccardDecisionTreeClassifier(),

#   LocalJaccardDecisionTreeClassifier(improve_only=True),
#   GlobalJaccardDecisionTreeClassifier(improve_only=True),
]
#   [
#       GlobalNVIDecisionTreeClassifier(),
#       GlobalNVIDecisionTreeClassifier(improve_only=True),
#   ]
#   LocalInformationGainDecisionTreeClassifier(), GlobalInformationGainDecisionTreeClassifier(improve_only=True),
#   LocalGiniImpurityDecisionTreeClassifier(), GlobalGiniImpurityDecisionTreeClassifier(improve_only=True),
#   GlobalJaccardDecisionTreeClassifier(), LocalNVIDecisionTreeClassifier(),
#   GlobalNVIDecisionTreeClassifier(improve_only=True),

d, title = iris(), "iris"
X, y = d.data, d.target


def train(clf_, _):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size)
    y_train_noise = add_noise(y_train, noise_p)

    clf = clone(clf_)
    collect_stats, retrieve_stats = train_stats(X_test, y_test)
    clf.watch = collect_stats
    clf.fit(X_train, y_train_noise)

    return retrieve_stats()
s = parallel_array(train, [clfs, range(n_repeat)], matrix=False)


def gen_line(i_clf, i):
    width = int(max([ max(s[i_clf, i_repeat][:,0]) for i_repeat in range(n_repeat) ]))
    yss = np.zeros((n_repeat, width))
    lens = []
    density = np.zeros(width)
    for i_repeat in range(n_repeat):
        ys = s[i_clf, i_repeat][:, i+1]
        l = len(ys)
        lens.append(l-1)
        density[0:l] += 1
        ys = np.pad(ys, (0, width - l), "edge")
        yss[i_repeat, :] = ys
    return (1 + np.arange(width),
            np.mean(yss, axis=0),
            density / max(density),
            mode(lens)[0][0])

#   yss = []
#   lens = []
#   for i_repeat in range(n_repeat):
#       xs, *zss = s[i_clf][i_repeat].transpose()
#       lens.append(int(xs[-1]))
#       for x, y in zip(xs, zss[i]):
#           x = int(x)
#           yss = yss + [[]] * (x+1 - len(yss))
#           yss[x].append(y)
#   return (np.arange(len(yss)),
#           np.array(list(map(lambda ys: np.mean(ys) if len(ys) > 0 else 0, yss))),
#           np.array(list(map(len, yss))) / n_repeat,
#           mode(lens)[0][0])

def plot_multicolor(xs, ys, colors, cmap="Blues", ax=plt, linestyle="solid", mode=None):
    from matplotlib.collections import LineCollection
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if ax is None:
        _, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    norm = plt.Normalize(colors.min() - (0.5 if colors.min() == colors.max() else 0), colors.max() + 0.3)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles=linestyle)
    lc.set_array(colors)
    lc.set_linewidth(1.0)
    ax.add_collection(lc)
    ax.plot(xs, ys, linewidth=1, alpha=0)
    if mode is not None:
        ax.scatter(m+1, ys[m], c=[matplotlib.cm.get_cmap(cmap)(0.8)], marker="|", linewidths=8, label="Mode")

def plot_all(i_clf, i, cmap="Blues", ax=plt):
    color = matplotlib.cm.get_cmap(cmap)(0.8)
    for i_repeat in range(n_repeat):
        xs, *zss = s[i_clf][i_repeat].transpose()
        ax.plot(xs, zss[i], color=color, alpha=0.1, linewidth=0.5)


"""
fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
for i_clf, (clf, cmap) in enumerate(zip(clfs, ["Blues", "Reds", "Greens"])):
    p = clf.plotting()
    xs, ys, ds, m = gen_line(i_clf, 2)
    plot_multicolor(xs, ys, ds, mode=m, cmap=cmap, linestyle="dashed", ax=ax)

    xs, ys, ds, m = gen_line(i_clf, 3)
    plot_multicolor(xs, ys, ds, mode=m, cmap=cmap, ax=ax)
ax.legend()
"""




fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(5, 4))
ax.set_ylim(0.3, 1)

legend_handles = []
for i_clf, (clf, cmap) in enumerate(zip(clfs, ["Blues", "Reds", "Greens", "Purples"])):
    color = matplotlib.cm.get_cmap(cmap)(0.8)
    p = clf.plotting()
    xs, ys, ds, m = gen_line(i_clf, 2)
    plot_multicolor(xs, ys, ds, mode=m, cmap=cmap, linestyle="dashed", ax=ax)
    xs, ys, ds, m = gen_line(i_clf, 3)
    plot_multicolor(xs, ys, ds, mode=m, cmap=cmap, ax=ax)

    legend_handles.append(mlines.Line2D([], [], ls="-", color=color, label='Test Accuracy %s%s' % (p["label"].replace("InformationGain", "IG"), " (" + p["params"] + ")" if p["params"] else ""), marker="|"))
    legend_handles.append(mlines.Line2D([], [], ls="--", color=color, label='Train Accuracy %s%s' % (p["label"].replace("InformationGain", "IG"), " (" + p["params"] + ")" if p["params"] else ""), marker="|"))

#   legend_handles.append(mlines.Line2D([], [], ls="-", color=color, label='Test Accuracy %s' % p["label"], marker="|"))
#   legend_handles.append(mlines.Line2D([], [], ls="--", color=color, label='Train Accuracy %s' % p["label"], marker="|"))

#   legend_handles.append(mlines.Line2D([], [], ls="-", color=color, label='Test (no noise) Accuracy %s (%s)' % (p["label"], p["params"]), marker="|"))
#   legend_handles.append(mlines.Line2D([], [], ls="--", color=color, label='Train Accuracy %s (%s)' % (p["label"], p["params"]), marker="|"))

ax.legend(loc='lower right', handles=legend_handles)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Training Step')

# fig.suptitle(title + " (train\\_size=%g, noise\\_p=%g, n\\_repeat=%d)" % (train_size, noise_p, n_repeat), y=0.91)
0



# plt.show()
save()

