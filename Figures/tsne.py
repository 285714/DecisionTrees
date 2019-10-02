from dt import *
from datasets import *
from plots import *
from sklearn.metrics import *
from sklearn.manifold import TSNE
import networkx as nx




clfs = [LocalGiniImpurityDecisionTreeClassifier(),
        LocalJaccardDecisionTreeClassifier(),
        GlobalGiniImpurityDecisionTreeClassifier(),
        GlobalJaccardDecisionTreeClassifier()]


d = diabetes()
X = d.data
y = d.target

yss_ = [ [] for _ in clfs ]
for i, clf in enumerate(clfs):
    clf.watch = lambda clf, _: yss_[i].append(clf.predict(X))
    clf.fit(X, y)


def translate(y_):
    t = dict(list(zip(np.unique(y), itertools.count())))
    return list(map(lambda l: t[l], y_))


def drawTSNE(yss_, df, perplexity=20):
#   G = nx.Graph()
#   for name, c in cs.items():
#       G.add_node(name, label=name)
#   for name1, c1 in cs.items():
#       for name2, c2 in cs.items():
#           if name1 != name2 and not G.has_edge(name2, name1):
#               d = df(c1, c2)
#               G.add_edge(name1, name2, weight=d, dist=round(d, 3))

    G = nx.DiGraph()
    all_ys_ = []

    colors = ["lightcoral", "red", "dodgerblue", "blue"]

    for i, ys_ in enumerate(yss_):
        for j, y_ in enumerate(ys_):
            y_ = translate(y_)
            n = len(all_ys_)
            all_ys_.append(y_)
            G.add_node(n,
                       label = ("" if j > 0 else "s") if j < len(ys_)-1 else "e",
                       color=colors[i])
            if j > 0: G.add_edge(n-1, n)

#   print(list(map(lambda i: accuracy_score(yss_[0][i], yss_[1][i]), range(min(len(yss_[0]), len(yss_[1]))))))

#   C = np.array(all_ys_)
    for p in [perplexity]: # i, p in enumerate([0.1, 1, 4, 8, 10, 11, 12, 13, 14, 16, 20, 25]):
#       print(p)

        pos = TSNE(n_components=2, metric=df, method="exact", perplexity=p).fit_transform(all_ys_)
        diam = np.max(pos) - np.min(pos)
        pos = 10 * pos / diam

#       fig = plt.figure(1, figsize=(20, 20))
#       ax = fig.add_subplot(1, 1, 1)
#       ax.set_aspect(1. / ax.get_data_ratio())

#       plt.subplot(3, 4, i+1)

        nx.draw(G,
            pos,
            labels = nx.get_node_attributes(G, 'label'),
            node_size = 5,
            node_color = list(nx.get_node_attributes(G, 'color').values()),
            arrowsize = 3,
            width = 0.5)

#       plt.title(str(p))

#   i = 0
#   pos_with_name = {}
#   for name, c in cs.items():
#       pos_with_name[name] = pos[i]
#       i += 1

#   edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
#   weights_inv = list(map(lambda w: 3 - w, weights))
#   nx.draw(G, pos_with_name, node_shape='.', node_size=10, alpha=0.9, labels=nx.get_node_attributes(G, 'label'),
#           edgelist=edges, edge_color=weights_inv, width=weights_inv, edge_cmap=plt.cm.Blues, font_color="red")
#   labels = nx.get_edge_attributes(G, 'dist')
#   # nx.draw_networkx_edge_labels(G, pos_with_name, edge_labels=labels)
#
#   nx.draw_networkx_edges(G, pos_with_name, edge_color="white", style="dashed", width=5,
#                          edgelist=list(map(lambda ed: ed[0], filter(lambda ed: ed[1] == 0,
#                                                                     nx.get_edge_attributes(G, 'weight').items()))))
#
#   # return
#   i = 0
#   for name, c in cs.items():
#       x, y = pos[i]
#       size = 25
#       axin = inset_axes(ax, width="100%", height="100%",
#                         bbox_to_anchor=(x - size / 2, y - size / 2, size, size),
#                         bbox_transform=ax.transData)
#       axin.scatter(X, Y, c=c)
#       plt.title(str(i) + ": " + name)
#       plt.xticks(())
#       plt.yticks(())
#       i += 1

drawTSNE(yss_, lambda x, y: 1 - accuracy_score(x,y), perplexity=30)
save()

