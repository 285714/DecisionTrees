import numpy as np
import itertools
import math
import sklearn.metrics as metrics

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment

from heapq import heappush, heappop, nlargest



# =============================================================================
# Helper Classes
# =============================================================================


class Sorted:
    """A container for samples sorted along each features

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        The input samples

    y: array of shape = [n_samples], optional (default=None)
        The classes of each sample in X

    indices: array, optional (default=None)
        Indicates the original sample position in array X

    top: Sorted, optional (default=None)
        Reference to the topmost instance of Sorted
    """
    def __init__(self, X, y = None, indices = None, top = None):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.sorted_lists = [ None for _j in range(self.n_features) ]
        if y is None:
            self.top = top
            self.n_classes = top.n_classes
            self.indices = indices
        else:
            self.top = self
            self.y_original = y
            self.retranslate_y = np.unique(y)
            self.translate_y = dict(zip(self.retranslate_y, itertools.count()))
            self.y = np.array([ self.translate_y[c] for c in y ])
            self.n_classes = len(self.retranslate_y)
            self.indices = np.arange(self.n_samples)

    """Sort samples according to each feature
    """
    def sort(self):
        for j in range(self.n_features):
            self.sorted_lists[j] = np.argsort(self.X[:, j])

    """Traverse through all features
    
    Yield
    -----
    i : int
        feature indices
    """
    def traverse_features(self):
        return range(self.n_features)


    """Traverse through all samples

    The traverse order follows the j-th feature, samples are traversed in batches of equal feature values

    Parameters
    ----------
    j : int
        The feature to traverse samples along
    
    Yield
    -----
    i : int
        feature indices

    ls : array of int
        indices of samples being traversed over since the last step

    count : array of int
        class count of samples in ls
    """
    def traverse_vectors(self, j):
        x_ = None
        ls = []
        for i in range(self.n_samples):
            l = self.sorted_lists[j][i]
            x = self.X[l,:]
            if x_ is None:
                x_ = x[j]
            if x[j] != x_:
                threshold = round((x[j] + x_) / 2, math.ceil(-math.log10(x[j] - x_)) + 1)
                yield i, ls, threshold
                ls = []
                x_ = x[j]
            ls.append(self.indices[l])

    """Split samples
    
    Samples are split into two along the i-th sample as ordered by the j-th feature
    
    Parameters
    ----------
    j : int
        feature used as order

    i : int
        position to split along

    Returns
    -------
    left, right : Sorted
        Sorded instances for the previous (left) and following (right) samples
    """
    def split(self, j, i):
        ls    = self.sorted_lists[j][0:i]
        rs    = self.sorted_lists[j][i:]
        left  = Sorted(self.X[ls], indices = self.indices[ls], top = self.top)
        right = Sorted(self.X[rs], indices = self.indices[rs], top = self.top)
        left.sort()
        right.sort()
        return left, right

    """Utility to count class occurrences

    Parameters
    ----------
    ls : array of int or None
        If given, only the samples with index in ls are counted

    Returns
    -------
    count : array of int
        class counts
    """
    def count(self, ls = None):
        if ls is None:
            ls = self.indices
        _, count = np.unique(np.hstack((range(self.top.n_classes), self.top.y[ls])), return_counts=True)
        return count - 1


class Node:
    """A single node in a decision tree with optional children

    Parameters
    ----------
    data : Sorted
        The samples that are sorted into this node

    label : int
        Class label of the node

    depth : int, optional (default=0)
        depth of this node in a tree

    threshold : float or None, optional (default=None)
        if None, the node is a leaf; otherwise does a sample belong into the left subtree if the feature (as given)
        is less than or equal to the threshold and to the right if it is greater than.

    is_leaf : bool, optional (default=True)
        indicates whether the node is currently a leaf

    left : Node or None, optional (default=None)
        left subtree

    right : Node or None, optional (default=None)
        right subtree
    """
    def __init__(self,
                 data,
                 label,
                 depth     = 0,
                 threshold = None,
                 feature   = None,
                 is_leaf   = True,
                 left      = None,
                 right     = None):
        self.is_leaf   = is_leaf
        self.left      = left
        self.right     = right
        self.depth     = depth
        self.feature   = feature
        self.threshold = threshold
        self.data      = data
        self.splittype = "global"
        self.label     = np.argmax(self.data.count()) if label is None else label

    """Depth of the tree rooted in this node

    Returns
    -------
    depth : int
        tree depth
    """
    def get_depth(self):
        return 0 if self.is_leaf else 1 + max(self.left.get_depth(), self.right.get_depth())

    """Number of nodes in the tree rooted in this node
    
    Returns
    -------
    n_nodes : int
        number of nodes
    """
    def get_n_nodes(self):
        return 1 + (0 if self.is_leaf else self.left.get_n_nodes() + self.right.get_n_nodes())

    """Are the samples in this node all of the same class

    Returns
    -------
    pure : bool
        true if the leaf is pure
    """
    def is_pure(self):
        return sum(self.data.count() > 0) <= 1

    """Predict classes following the tree from this node

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        The samples to predict

    Returns
    -------
    y_ : array of shape = [n_samples]
        The predicted class labels
    """
    def predict(self, X):
        if self.is_leaf:
            return np.array([self.label] * len(X))
        else:
            cs        = X[:,self.feature] > self.threshold
            ls        = np.where(cs == 0)[0]
            rs        = np.where(cs == 1)[0]
            left_X   = X[ls]
            right_X  = X[rs]
            left_y_  = self.left.predict(left_X)
            right_y_ = self.right.predict(right_X)
            y_       = np.empty(len(X), dtype=np.int32)
            y_[ls]   = left_y_
            y_[rs]   = right_y_
            return y_

    """Split this node

    Parameters
    ----------
    j : int
        the splitting feature

    i : int
        the position to split samples at

    label_left : int
        class label of the left child node

    label_right : int
        class label of the right child node

    threshold : float
        threshold to split at, corresponds to i
    """
    def split(self, j, i, label_left, label_right, threshold):
        self.is_leaf = False
        self.feature = j
        self.threshold = threshold
        left_data, right_data = self.data.split(j, i)
        self.left  = Node(left_data,  label_left,  self.depth + 1)
        self.right = Node(right_data, label_right, self.depth + 1)
        return self.left, self.right

    """Undo a past split
    """
    def undo_split(self):
        self.is_leaf = True
        self.feature = self.threshold = self.left = self.right = None

    """Statistics on the samples in this node
    
    Used for plotting the decision tree
    """
    def stats(self):
        count = self.data.count()
        return {"gini":    round(1 - sum(np.square(count / self.data.n_samples)), 3),
                "entropy": round(entropy(count), 3),
                "samples": self.data.n_samples,
                "value":   list(count),
                "note":    None,
                "highlight": self.splittype == "entropy"}

    """Collect all nodes in the tree rooted in this node
    
    Returns
    -------
    nodes : array of Node
        all descendant nodes and this node
    """
    def get_all_nodes(self):
        if self.is_leaf:
            return [self]
        else:
            return [self] + self.left.get_all_nodes() + self.right.get_all_nodes()


    """Generates a name for this node
    
    for debugging only
    """
    def get_name(self, origin, target=None, chain=[]):
        if target is None:
            return origin.get_name(None, target=self)
        else:
            if target is self:
                return "[" + ("L" if self.is_leaf else "B") + " " + " ∧ ".join(chain) + "]"
            elif self.is_leaf:
                return ""
            else:
                left, right = map(lambda s: "x%d %s %g" % (self.feature, s, self.threshold), ["⩽", ">"])
                l = self.left.get_name(None, target=target, chain=chain+[left])
                r = self.right.get_name(None, target=target, chain=chain+[right])
                return l + r

    """Equality check with another tree

    for debugging only
    
    Parameters
    ----------
    t : Node
        the tree rooted in t to compare with

    Returns
    -------
    is_equal : bool
        true, if both nodes and descendants are equal
    """
    def equals(self, t):
        if self.is_leaf or t.is_leaf:
            return self.is_leaf == t.is_leaf
        else:
            return self.feature == t.feature and self.threshold == t.threshold and \
                   self.left.equals(t.left) and self.right.equals(t.right)


"""Utility functions to generate colors
    (https://github.com/scikit-learn/scikit-learn/blob/7b136e92acf49d46251479b75c88cba632de1937/sklearn/tree/export.py#L25)
"""
def _color_brew(n):
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list

def get_color(colors, value):
    color = list(colors['rgb'][np.argmax(value)])
    sorted_values = sorted(value, reverse=True)
    if len(sorted_values) == 1:
        alpha = 0
    else:
        alpha = int(np.round(255 * (sorted_values[0] -
                                    sorted_values[1]) /
                             (1 - sorted_values[1]), 0))

    # Return html color code in #RRGGBBAA format
    color.append(alpha)
    hex_codes = [str(i) for i in range(10)]
    hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
    color = [hex_codes[c // 16] + hex_codes[c % 16] for c in color]

    return '#' + ''.join(color)



# =============================================================================
# Decision Tree Classifiers
# =============================================================================


class BaseDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Base class for all decision tree learners

    Parameters
    ----------
    max_depth : int or None, optional (default=None)
        if not None, limits the depth of the fitted tree

    min_samples_split : int, optional (default=2)
        the minimum number of samples required to perform a split

    min_samples_leaf : int, optional (default=1)
        the minimum number of samples allowed per leaf

    max_leaf_nodes : int or None, optional (default=None)
        if not None, limits the number of nodes in the fitted tree

    min_dist : float or None, optional (default=None)
        if not none, the tree is grown until it reaches this minimum distance to the ground truth

    improve_only : bool, optional (default=False)
        if true, splits will only be performed if that decreases the distance

    glocal : bool, optional (default=False)
        (global evaluation only) if true, the optimizer will perform local evaluations once every split induces a global
        increase in distance to the ground truth

    f : function or None, optional (default=None)
        (generic Jaccard only) the submodular function used in the global generic Jaccard optimizer

    verbose : bool, optional (default=False)
        prints debugging output if true

    watch : function or None, optional (default=None)
        (debugging only) watch function called on every split
    """
    def __init__(self, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_leaf_nodes = None,
                 min_dist = None, improve_only = False, glocal = False, verbose = False, watch = None, f = None):
        assert not(glocal) or (glocal and not(improve_only))

        self.set_params(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                        max_leaf_nodes=max_leaf_nodes, verbose=verbose, watch=watch, f=f, min_dist=min_dist,
                        improve_only=improve_only, glocal=glocal)

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_leaf_nodes": self.max_leaf_nodes,
                "min_dist": self.min_dist, "improve_only": self.improve_only, "glocal": self.glocal}

    def set_params(self, **params):
        self.__dict__.update(params)

    """Utility function to iterate through splits as long as they are allowed
    
    ... according to max_leaf_nodes

    Parameters
    ----------
    i : int, optional (default=0)
        number of nodes to start with
    
    Yields
    -------
    n_splits : int
        number of split performed
    """
    def check_iteration(self, i=0):
        while self.max_leaf_nodes is None or i < self.max_leaf_nodes:
            yield i
            i += 1

    """Utility function to ensure asplit is allowed

    ... according to max_depth, min_samples_split, and min_samples_leaf
    
    Parameters
    ----------
    a : Node
        node that is being split

    left_sum : array of int
        class counts of the samples in the left leaf

    right_sum : array of int
        class counts of the samples in the right leaf
    """
    def check_split(self, a, left_sum, right_sum):
        return (self.max_depth is None or a.depth < self.max_depth) and \
               left_sum + right_sum >= self.min_samples_split and \
               left_sum >= self.min_samples_leaf and right_sum >= self.min_samples_leaf

    """Start learning
    
    to be implemented by derived classes

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        The input samples

    y: array of shape = [n_samples], optional (default=None)
        The classes of each sample in X
    """
    def fit(self, X, y):
        return self

    """Predict class labels according to the fitted three

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        The samples to predict

    Returns
    -------
    y_ : array of shape = [n_samples]
        The predicted class labels
    """
    def predict(self, X):
        return [ self.data_.retranslate_y[c] for c in list(self.tree_.predict(X)) ]

    """Export the tree for plotting in grahpviz syntax

    Parameters
    ----------
    feature_names : array of string or None, optional (default=None)
        if not none, used as feature names instead of indices

    class_names : array of string or None, optional (default=None)
        if not none, used as class names instead of indices

    Returns
    -------
    graphviz : string
        decision tree in graphviz syntax
    """
    def export_graphviz(self,
                        feature_names = None,
                        class_names = None):
        if feature_names is None:
            feature_names = [ "X<SUB>%d</SUB>" % (f+1) for f in range(self.data_.n_features) ]
        if class_names is None:
            class_names = { c: "class%s" % c for c in self.data_.top.translate_y }

        colors = {'rgb': _color_brew(self.data_.top.n_classes)}
        lines = ['node [shape=box, style="filled", color="black"]']
        q = [(self.tree_, 0)]
        i_max = 0
        while len(q) > 0:
            n, i = q.pop()
            info = n.stats()
            color = get_color(colors, np.array(info["value"]) / info["samples"])

            lines.append(str(i) +
                         ' [label=<' + ('' if n.is_leaf else '{} &le; {}<br/>'.
                                        format(feature_names[n.feature], n.threshold)) +
                         'gini = {}<br/>entropy = {}<br/>samples = {}<br/>value = {}{}>, fillcolor="{}"{}]'.
                         format(info["gini"], info["entropy"], info["samples"], info["value"],
                                '' if info["note"] is None else '<br/>{}'.format(info["note"]), color,
                                ', shape="note"' if info["highlight"] else ''))

            if not n.is_leaf:
                left_i  = i_max + 1
                right_i = i_max + 2
                i_max += 2
                lines.append('{} -> {}'.format(i, left_i) +
                             (' [labeldistance=2.5, labelangle=45, headlabel="True"]' if left_i == 1 else ''))
                lines.append('{} -> {}'.format(i, right_i) +
                             (' [labeldistance=2.5, labelangle=-45, headlabel="False"]' if right_i == 2 else ''))
                q.insert(0, (n.left,  left_i))
                q.insert(0, (n.right, right_i))
                i_max += 2

        return 'digraph Tree {\n' + str.join(' ;\n', lines) + ' ;\n}'

    """Utility function to have matching colors for estimators when plotting results
    """
    def plotting(self):
        name = type(self).__name__.split("DecisionTreeClassifier")[0]
        is_local = "Local" in name
        color = {"GiniImpurity": "limegreen", "InformationGain": "blue", "Jaccard": "chocolate", "GainRatio": "cornflowerblue",
                 "Accuracy": "lightcoral", "NVI": "darkviolet", "AreaSep": "cyan"}[name.split("al")[1]]
        snd_color = "red"
        self_params = self.get_params()
        default_params = self.__class__().get_params()
        params = ", ".join([ k if type(v) is bool else "%s=%s" % (k,v) for k,v in self_params.items() if default_params[k] != v ])
        params = str.replace(params, "_", "\\_")
        return {"fmt": ("-" if is_local else "-."), "label": name, "is_local": is_local, "is_global": not is_local,
                "color": color, "snd_color": color if is_local else snd_color, "params":params}


class LocalDecisionTreeClassifier(BaseDecisionTreeClassifier):
    """Base class for decision tree classifiers performing local evalutions

    Parameters: see BaseDecisionTreeClassifier
    """
    def fit(self, X, y):
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)

        heap = None
        for iter in self.check_iteration():
            childs = []
            if heap is None:
                childs = [self.tree_]
                heap = []
            else:
                if self.watch is not None: self.watch(self, iter)
                if len(heap) == 0:
                    break
                _, _, split_function = heappop(heap)
                childs = split_function()
            for a in childs:
                if a.is_pure():
                    continue
                s = self.find_best_split(a)
                if s is not None:
                    heappush(heap, s)

        return self

    """Distance function (to be evaluated locally)
    
    Parameters
    ----------
    a : Node
        current leaf being considered for splitting

    count : array of array of int of shape=[2, n_classes]
        confusion matrix of a split vs. the ground truth

    Returns
    -------
    distance : float
        distance to the ground truth
    """
    def dist_function(self, a, count):
        return None

    """Find the split for a leaf that has minimum distance to the ground truth

    Parameters
    ----------
    a : Node
        current leaf
    """
    def find_best_split(self, a):
        if a.is_pure():
            return None

        count_all = a.data.count()
        value_min = None
        split_function = None

        for j in a.data.traverse_features():
            count_right = np.copy(count_all)
            count_left = np.zeros_like(count_right)

            for i, ls, threshold in a.data.traverse_vectors(j):
                count = a.data.count(ls)
                count_right = count_right - count
                count_left = count_left + count

                if not self.check_split(a, sum(count_left), sum(count_right)):
                    continue

                value = self.dist_function(a, np.array([count_left, count_right]).transpose() / a.data.n_samples)

                if value_min is None or value < value_min:
                    value_min = value
                    a_, j_, i_, threshold_ = a, j, i, threshold
                    l_right_, l_left_ = np.argmax(count_right), np.argmax(count_left)
                    split_function = lambda: a.split(j_, i_, l_left_, l_right_, threshold_)

        if split_function is None or (self.improve_only and value_min < 0):
            return None
        else:
            return value_min, np.random.rand(), split_function


class LocalInformationGainDecisionTreeClassifier(LocalDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of local evaluations of the information gain

    Parameters: see LocalDecisionTreeClassifier
    """
    def dist_function(self, a, confusion_matrix):
        marginal0 = np.sum(confusion_matrix, axis=0)
        marginal1 = np.sum(confusion_matrix, axis=1)
        return -(entropy(marginal0) + entropy(marginal1) - entropy(confusion_matrix.flatten()))


class LocalGainRatioDecisionTreeClassifier(LocalDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of local evaluations of the gain ratio

    Parameters: see LocalDecisionTreeClassifier
    """
    def dist_function(self, a, confusion_matrix):
        marginal0 = np.sum(confusion_matrix, axis=0)
        gain = LocalInformationGainDecisionTreeClassifier.dist_function(self, a, confusion_matrix)
        return gain / (entropy(marginal0) or np.inf)


class LocalNVIDecisionTreeClassifier(LocalDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of local evaluations of the normalized variation of
    information

    Parameters: see LocalDecisionTreeClassifier
    """
    def dist_function(self, a, confusion_matrix):
        gain = LocalInformationGainDecisionTreeClassifier.dist_function(self, a, confusion_matrix)
        return 1 + np.nan_to_num(gain / entropy(confusion_matrix.flatten()))


class LocalGiniImpurityDecisionTreeClassifier(LocalDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of local evaluations of the Gini impurity

    Parameters: see LocalDecisionTreeClassifier
    """
    def dist_function(self, a, confusion_matrix):
        marginal0 = np.sum(confusion_matrix, axis=0)
        squared = np.sum(np.square(confusion_matrix), axis=0)
        return 1 - np.inner(np.divide(1, marginal0, out=np.zeros_like(marginal0), where=marginal0 != 0), squared)


class LocalJaccardDecisionTreeClassifier(LocalDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of local evaluations of the extended Jaccard distance

    Parameters: see LocalDecisionTreeClassifier
    """
    def dist_function(self, a, confusion_matrix):
        k = a.data.n_classes
        confusion_matrix = np.pad(confusion_matrix, (0,k), "constant")[0:k,0:k]
        marginal0 = np.sum(confusion_matrix, axis=0)
        marginal1 = np.sum(confusion_matrix, axis=1)
        union = np.add.outer(marginal1, marginal0) - confusion_matrix
        jaccards = np.divide(confusion_matrix, union, out=np.zeros_like(union), where=union != 0)
        row_ind, col_ind = linear_sum_assignment(-jaccards)
        return k - sum([jaccards[i, col_ind[i]] for i in row_ind])


class LocalAccuracyDecisionTreeClassifier(LocalDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of local evaluations of the accuracy

    Parameters: see LocalDecisionTreeClassifier
    """
    def dist_function(self, a, confusion_matrix):
        k = a.data.n_classes
        confusion_matrix = np.pad(confusion_matrix, (0,k), "constant")[0:k,0:k]
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        return 1 - sum([confusion_matrix[i, col_ind[i]] for i in row_ind])


class GlobalJaccardDecisionTreeClassifier(BaseDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of global evaluations of the extended Jaccard distance

    Also a base class for other global evaluation decision tree classifiers

    Parameters: see BaseDecisionTreeClassifier
    """

    """Finding the best split based on local evaluations
    
    Parameters
    ----------
    leaves : array of Node
        nodes of the current tree

    Returns
    -------
    split_function : function
        carries out the split and returns the new leaves
    """
    local_optimize_max_entropy = 1.0
    def local_optimize(self, leaves):
        candidates = [ (LocalInformationGainDecisionTreeClassifier.find_best_split(self, a), a)
                       for a in leaves if entropy(a.data.count()) < self.local_optimize_max_entropy ]
        candidates = list(filter(lambda x: x[0] is not None, candidates))

        if len(candidates) > 0:
            (gain, _, split_function), a = max(candidates)
            leaves.remove(a)
            a.splittype = "entropy"
            if self.verbose: print("(local) entropy step gaining %3f" % gain)
            return split_function
        else:
            return None

    """Distance function for glocal optimization
    """
    def dist_function(self, a, count):  # in case jaccard optimization is flat
        #       return EntropyDecisionTreeClassifier.dist_function(self, a, count)
        return LocalGiniImpurityDecisionTreeClassifier.dist_function(self, a, count)  #

    """Maximize values from the list on differnt positions

    Finds indices i1, i2 such that i1 != i2 and xs[i1] + xs[i2] is maximized. Used to determine class labels

    Parameters
    ----------
    xs, ys : list of int

    Returns
    -------
    a1, b1 : int
        list positions
    """
    def best2(self, xs, ys):
        if len(xs) == 1:
            return 0, 0
        a1, a2 = nlargest(2, range(len(xs)), xs.take)
        b1, b2 = nlargest(2, range(len(ys)), ys.take)
        if a1 == b1:
            if xs[a1] + ys[b2] > xs[a2] + ys[b1]:
                return a1, b2
            else:
                return a2, b1
        else:
            return a1, b1

    def fit(self, X, y):
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)

        def jaccards(union, symdif):
            return symdif / (union + (union == 0))

        union = self.data_.count()
        symdif = np.copy(union)
        symdif[0] = self.data_.n_samples - union[0]
        union[0] = self.data_.n_samples

        global_dist = sum(jaccards(union, symdif))
        leaves = [self.tree_]

        for iter in self.check_iteration(1):
            if self.min_dist is not None and global_dist < self.min_dist:
                break

            if self.watch is not None: self.watch(self, iter)

            split_function = None
            improve_max = None

            for a in leaves:
                count_all = a.data.count()
                n = a.data.n_samples
                js_all = jaccards(union, symdif)

                for j in a.data.traverse_features():
                    union_tmp_right = np.copy(union)
                    symdif_tmp_right = np.copy(symdif)

                    union_tmp_right += n - count_all
                    union_tmp_right[a.label] = union[a.label]
                    symdif_tmp_right += n - 2*count_all
                    symdif_tmp_right[a.label] = symdif[a.label]

                    union_tmp_left = np.copy(union)
                    symdif_tmp_left = np.copy(symdif)
                    union_tmp_left[a.label] -= n - count_all[a.label]
                    symdif_tmp_left[a.label] -= n - 2*count_all[a.label]

                    union_remove = union_tmp_left[a.label]
                    symdif_remove = symdif_tmp_left[a.label]
                    j_remove = jaccards(union_tmp_left, symdif_tmp_left)[a.label]

                    total_sum = 0
                    for i, ls, threshold in a.data.traverse_vectors(j):
                        count = a.data.count(ls)
                        s = sum(count)
                        total_sum += s

                        if not self.check_split(a, total_sum, n - total_sum):
                            continue

                        # adjust confusion matrix
                        union_tmp_right -= s - count
                        symdif_tmp_right -= s - 2*count

                        union_tmp_left += s - count
                        symdif_tmp_left += s - 2*count

                        js_right = jaccards(union_tmp_right, symdif_tmp_right)
                        js_left = jaccards(union_tmp_left, symdif_tmp_left)

                        improve_right = js_all - js_right
                        improve_left = js_all - js_left

                        xs = improve_right + js_all[a.label] - j_remove
                        xs[a.label] -= js_all[a.label] - j_remove
                        ys = improve_left
                        ys[a.label] -= js_all[a.label] - j_remove
                        l_right, l_left = self.best2(xs, ys)
                        improve = xs[l_right] + ys[l_left]

                        if improve_max is None or improve > improve_max:
                            improve_max = improve
                            a_, j_, i_, l_right_, l_left_, threshold_ = a, j, i, l_right, l_left, threshold
                            improve_ = improve

                            if l_right == a.label:
                                union_remove_, symdif_remove_ = union_tmp_right[a.label], symdif_tmp_right[a.label]
                            elif l_left == a.label:
                                union_remove_, symdif_remove_ = union_tmp_left[a.label], symdif_tmp_left[a.label]
                            else:
                                union_remove_, symdif_remove_ = union_remove, symdif_remove
                            union_right_, symdif_right_ = union_tmp_right[l_right], symdif_tmp_right[l_right]
                            union_left_, symdif_left_ = union_tmp_left[l_left], symdif_tmp_left[l_left]

                            def split_function():
                                nonlocal global_dist
                                leaves.remove(a_)
                                global_dist -= improve_
                                union[a_.label] = union_remove_
                                symdif[a_.label] = symdif_remove_
                                union[l_right_] = union_right_
                                symdif[l_right_] = symdif_right_
                                union[l_left_] = union_left_
                                symdif[l_left_] = symdif_left_
                                return a_.split(j_, i_, l_left_, l_right_, threshold_)

            if improve_max is None:
                break
            elif self.improve_only and improve_max < 0:
                break
            elif improve_max <= 0 and self.glocal:
                if self.verbose: print("deterioration")
                split_function_ = self.local_optimize(leaves)
                if split_function_ is not None:
                    split_function = split_function_

            childs = split_function()
            leaves += [ c for c in childs if not c.is_pure() ]
            if self.verbose: print("%3d %f" % (iter, global_dist))

        return self


class GlobalGenericJaccardDecisionTreeClassifier(GlobalJaccardDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of global evaluations of the generic extended Jaccard
    distance

    ... using the submodular, nonnegative, and monotonous function f

    Parameters: see GlobalJaccardDecisionTreeClassifier
    """
    local_optimize_max_entropy = 0.5

    def fit(self, X, y):
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)

        if self.f is None:
            self.f = "cardinality"
        if isinstance(self.f, str):
            self.f = {"entropy": lambda x: entropy(abs(x)) if sum(x) > 0 else 0,
                      "gini": lambda x: (1 - 2*gini(abs(x))) if sum(x) > 0 else 0,
                      "cardinality": sum}[self.f]

        count_all = self.data_.count()

        def jaccards(m):
            unions = m - np.diag(np.diag(m)) + np.diag(count_all)
            symdifs = unions - np.diag(np.diag(m))
            denom = np.apply_along_axis(self.f, 1, unions)
            return np.apply_along_axis(self.f, 1, symdifs) / (denom + (denom == 0))

        leaves = [self.tree_]
        confusion_matrix = np.array([count_all] + [[0] * self.data_.n_classes] * (self.data_.n_classes - 1))
        global_dist = sum(jaccards(confusion_matrix))

        for iter in self.check_iteration(1):
            if global_dist < self.min_dist:
                break

            if self.watch is not None: self.watch(self, iter)

            split_function = None
            improve_max = None

            js_all = jaccards(confusion_matrix)
            for a in leaves:
                count_leaf = a.data.count()
                range_except = [ c for c in range(self.data_.n_classes) if c != a.label ]
                confusion_matrix_empty = np.copy(confusion_matrix)
                confusion_matrix_empty[a.label] -= count_leaf
                j_remove = jaccards(confusion_matrix_empty)[a.label]

                for j in a.data.traverse_features():
                    confusion_matrix_right = np.copy(confusion_matrix_empty)
                    confusion_matrix_right += count_leaf
                    confusion_matrix_left = np.copy(confusion_matrix_empty)

                    for i, ls, threshold in a.data.traverse_vectors(j):
                        count = a.data.count(ls)
                        confusion_matrix_left += count
                        confusion_matrix_right -= count

                        js_right = jaccards(confusion_matrix_right)
                        js_left = jaccards(confusion_matrix_left)

                        improve_right = js_all - js_right
                        improve_left = js_all - js_left

                        xs = improve_right
                        xs[range_except] += js_all[a.label] - j_remove
                        ys = improve_left
                        ys[a.label] -= js_all[a.label] - j_remove
                        l_right, l_left = self.best2(xs, ys)
                        improve = xs[l_right] + ys[l_left]

                        if improve_max is None or improve > improve_max:
                            improve_max = improve
                            a_, j_, i_, l_right_, l_left_, threshold_ = a, j, i, l_right, l_left, threshold
                            improve_ = improve
                            confusion_matrix_ = np.copy(confusion_matrix_empty)
                            confusion_matrix_[l_right] = confusion_matrix_right[l_right]
                            confusion_matrix_[l_left] = confusion_matrix_left[l_left]

                            def split_function():
                                nonlocal global_dist, confusion_matrix
                                leaves.remove(a_)
                                global_dist -= improve_
                                confusion_matrix = confusion_matrix_
                                return a_.split(j_, i_, l_left_, l_right_, threshold_)

            if improve_max is None:
                break
            elif improve_max <= 0:
                split_function_ = self.local_optimize(leaves)
                if split_function_ is not None:
                    def split_function():
                        nonlocal confusion_matrix
                        ret = split_function_()
                        y_ = self.predict(X)
                        confusion_matrix = np.array(metrics.confusion_matrix(y_, y))
                        return ret

            childs = split_function()
            leaves += [ c for c in childs if not c.is_pure() ]

        return self


class GlobalGenericConfusionMatrixDecisionTreeClassifier(GlobalJaccardDecisionTreeClassifier):
    """Base class for decision tree classifiers performing global evalutions of a confusion matrix

    Parameters: see GlobalJaccardDecisionTreeClassifier
    """

    """Select the best split

    ... using lexicographic order

    Parameters
    ----------
    global_dist : float
        distance of the whole tree (global) to the ground truth
        
    local_dist : float
        distance of the current leaf (local) to the ground truth
        
    rand : float
        random value to break ties

    min_global_dist : float
        distance of the whole tree (global) to the ground truth of the current best split
        
    local_dist : float
        distance of the current leaf (local) to the ground truth of the current best split
        
    rand : float
        random value to break ties of the current best split
    """
    def select(self, global_dist, local_dist, rand, min_global_dist, min_local_dist, min_rand):
        return (global_dist, local_dist, rand) < (min_global_dist, min_local_dist, min_rand)

    def fit(self, X, y):
        if self.verbose: print(" --#--  ---  --#--")
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)

        count_all = self.data_.count()

        leaves = [self.tree_]
        confusion_matrix = np.copy(np.array([count_all] + [[0] * self.data_.n_classes] * (self.data_.n_classes - 1)))
        previous_global_dists = []

        for iter in self.check_iteration(1):
            if self.watch is not None: self.watch(self, iter)

            split_function = None
            min_dists = None
            current_global_dist = self.distance(confusion_matrix.transpose())
            previous_global_dists.append(current_global_dist)
            if len(previous_global_dists) > 3:
                previous_global_dists.pop(0)
                d1 = previous_global_dists[0] - previous_global_dists[1]
                d2 = previous_global_dists[1] - previous_global_dists[2]
                if d1 - d2 == 0:
                    pass
                if self.verbose: print(d1 - d2)

            for a in leaves:
                count_leaf = a.data.count()
                confusion_matrix_empty = np.copy(confusion_matrix)
                confusion_matrix_empty[a.label] -= count_leaf

                for j in a.data.traverse_features():
                    count_right = np.copy(count_leaf)
                    count_left = np.zeros_like(count_leaf)

                    for i, ls, threshold in a.data.traverse_vectors(j):
                        count = a.data.count(ls)
                        count_right -= count
                        count_left += count

                        l_right = np.argmax(count_right)
                        l_left = np.argmax(count_left)

                        confusion_matrix_empty[l_right] += count_right
                        confusion_matrix_empty[l_left] += count_left

                        global_dist = self.distance(confusion_matrix_empty.transpose())
                        local_dist = self.distance(np.array([count_right, count_left]).transpose())
                        dists = (global_dist, local_dist, np.random.rand())
                        assert not(np.isnan(global_dist) or np.isnan(local_dist))

                        if self.improve_only and global_dist > current_global_dist or \
                                not self.check_split(a, sum(count_left), sum(count_right)):
                            continue

                        if min_dists is None or self.select(*dists, *min_dists):
                            min_dists = dists
                            a_, j_, i_, l_right_, l_left_, threshold_ = a, j, i, l_right, l_left, threshold
                            confusion_matrix_ = np.copy(confusion_matrix_empty)

                            def split_function():
                                nonlocal confusion_matrix
                                leaves.remove(a_)
                                confusion_matrix = confusion_matrix_
                                return a_.split(j_, i_, l_left_, l_right_, threshold_)

                        confusion_matrix_empty[l_right] -= count_right
                        confusion_matrix_empty[l_left] -= count_left

            if min_dists is None:
                break

            childs = split_function()
            leaves += [ c for c in childs if not c.is_pure() ]

        return self


class GlobalInformationGainDecisionTreeClassifier(GlobalGenericConfusionMatrixDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of global evaluations of the information gain

    Parameters: see GlobalGenericConfusionMatrixDecisionTreeClassifier
    """
    def distance(self, m):
        return LocalInformationGainDecisionTreeClassifier.dist_function(self, self.tree_, m / np.sum(m))


class GlobalGainRatioDecisionTreeClassifier(GlobalGenericConfusionMatrixDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of global evaluations of the gain ratio

    Parameters: see GlobalGenericConfusionMatrixDecisionTreeClassifier
    """
    def distance(self, m):
        return LocalGainRatioDecisionTreeClassifier.dist_function(self, self.tree_, m / np.sum(m))


class GlobalNVIDecisionTreeClassifier(GlobalGenericConfusionMatrixDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of global evaluations of the normalized variation of
    information

    Parameters: see GlobalGenericConfusionMatrixDecisionTreeClassifier
    """
    def distance(self, m):
        return LocalNVIDecisionTreeClassifier.dist_function(self, self.tree_, m / np.sum(m))


class GlobalGiniImpurityDecisionTreeClassifier(GlobalGenericConfusionMatrixDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of global evaluations of the Gini impurity

    Parameters: see GlobalGenericConfusionMatrixDecisionTreeClassifier
    """
    def distance(self, m):
        return LocalGiniImpurityDecisionTreeClassifier.dist_function(self, self.tree_, m / np.sum(m))


class GlobalAccuracyDecisionTreeClassifier(GlobalGenericConfusionMatrixDecisionTreeClassifier):
    """Decision tree classifier performing splits on the basis of global evaluations of the accuracy

    Parameters: see GlobalGenericConfusionMatrixDecisionTreeClassifier
    """
    def distance(self, m):
        return LocalAccuracyDecisionTreeClassifier.dist_function(self, self.tree_, m / np.sum(m))


class GlobalGenericDecisionTreeClassifier(GlobalJaccardDecisionTreeClassifier):
    """Base class for decision tree classifiers performing internal global and local evalutions

    Splits are evaluated with the function called `d`

    Parameters: see GlobalJaccardDecisionTreeClassifier
    """
    def fit(self, X, y):
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)

        def select(cs):
            return min([ (c[0][0], c) for c in cs if c[0][2] ])[1]

        leaves = [self.tree_]
        for iter in self.check_iteration(1):
            if self.watch is not None: self.watch(self, iter)

            candidates =[ (self.simulate_split(a, j, i, l_left, l_right, threshold),
                           np.random.rand(), a, j, i, threshold, l_left, l_right)
                          for a in leaves
                          for j in a.data.traverse_features()
                          for i, _, threshold in a.data.traverse_vectors(j)
                          for l_right in range(self.data_.n_classes)
                          for l_left in range(self.data_.n_classes) ]

            if len(candidates) == 0:
                break

            (global_dist, local_dist, _), _, a, j, i, threshold, l_left, l_right = select(candidates)

            leaves.remove(a)
            childs = a.split(j, i, l_left, l_right, threshold)
            leaves += [ c for c in childs if not c.is_pure() ]

            if self.min_dist is not None and self.min_dist >= global_dist:
                break

            if self.verbose: print(iter, global_dist, local_dist)

        return self

    """A split is simulated and evaluated locally and globally.
    
    Parameters
    ----------
    a : Node
        the node to split

    j : int
        the splitting feature

    i : int
        the position to split samples at

    l_left : int
        class label of the left child node

    l_right : int
        class label of the right child node

    threshold : float
        threshold to split at, corresponds to i
    """
    def simulate_split(self, a, j, i, l_left, l_right, threshold):
        a.split(j, i, l_left, l_right, threshold)

        # global
        y_ = self.predict(self.data_.X)
        global_dist = self.d(self.data_.y_original, y_)

        #local
        def is_mode(ls, l):
            bins = np.bincount(ls)
            return max(bins) == 0 if l >= len(bins) else bins[l] == max(bins)
        y_ = a.predict(a.data.X)
        local_dist = self.d(self.data_.y[a.data.indices], y_)
        y_left = self.data_.y[a.left.data.indices]
        y_right = self.data_.y[a.right.data.indices]
        is_differently_labeled = l_left != l_right

        a.undo_split()
        assert not(np.isnan(global_dist) or np.isnan(local_dist))
        return (global_dist, local_dist, is_differently_labeled)



# =============================================================================
# Experimental Decision Tree Classifiers
# =============================================================================


class ExperimentalDecisionTreeClassifier(BaseDecisionTreeClassifier):
    def fit(self, X, y):
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)

        old_dist = None
        mode = 0 # 0: add split, 1: move/remove split

        while True:
            nodes = self.tree_.get_all_nodes()
            ops = []

            for a in nodes:
                if mode == 0 and a.is_leaf:
                    for j in a.data.traverse_features():
                        for i, _, threshold in a.data.traverse_vectors(j):
                            ops.append(((lambda a, j, i, threshold: lambda: self.new_split(a, j, i, threshold))(a, j, i, threshold),
                                        "new(" + a.get_name(self.tree_) + ", " + str(j) + ", " + str(threshold) + ")"))
                elif mode == 1 and not a.is_leaf:
                    ops.append(((lambda a: lambda: self.remove_split(a))(a),
                                 "remove(" + a.get_name(self.tree_) + ")"))
                    for _, _, threshold in a.data.traverse_vectors(a.feature):
                        ops.append(((lambda a, threshold: lambda: self.move_split(a, threshold))(a, threshold),
                                    "move(" + a.get_name(self.tree_) + ", " + str(threshold) + ")"))

            min_dist = None
            min_op = None

            for op in ops:
                undo = op[0]()
                y_ = self.predict(self.data_.X)
                dist = self.distance(self.data_.y_original, y_)

                undo()

                if min_dist is None or min_dist > dist:
                    min_dist = dist
                    min_op = op

            mode_ = mode
            mode = 1
            if not old_dist is None and old_dist <= min_dist:
                if mode_ == 1:
                    mode = 0
                else:
                    break
            old_dist = min_dist

            min_op[0]()

            y_ = self.predict(self.data_.X)
            dist = self.distance(self.data_.y_original, y_)
            print("  ~~~~  ", min_op[1], "->", dist, "\n")

    def distance(self, y_true, y_pred):
        return GlobalInformationGainDecisionTreeClassifier.distance(self,self._tree,
                                                                    metrics.confusion_matrix(y_true, y_pred) / np.sum(y_true))

    def move_split(self, a, threshold):
        threshold_ = a.threshold
        a.threshold = threshold

        def undo():
            a.threshold = threshold_
        return undo

    def new_split(self, a, j, i, threshold):
        ls = a.data.indices[a.data.sorted_lists[j][0:i]]
        rs = a.data.indices[a.data.sorted_lists[j][i:]]
        l_left  = np.argmax(a.data.count(ls))
        l_right = np.argmax(a.data.count(rs))
        a.split(j, i, l_left, l_right, threshold)

        def undo():
            a.undo_split()
        return undo

    def remove_split(self, a):
        a.is_leaf = True

        def undo():
            a.is_leaf = False
        return undo


class LocalGenericDecisionTreeClassifier(BaseDecisionTreeClassifier):
    def fit(self, X, y):
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)
        leaves = [self.tree_]

        for iter in self.check_iteration():
            if self.verbose: print("        ~ [ accuracy: %f, #nodes: %d, depth: %d ]" % (self.score(self.data_.X, self.data_.y_original), self.tree_.get_n_nodes(), self.tree_.get_depth()))

            max_gain = None
            op = None

            for a in leaves:
                y = self.data_.y
                for j in a.data.traverse_features():
                    y_ = np.zeros_like(y)
                    base_distance = self.distance(y[a.data.indices], y_[a.data.indices], a)

                    for i, ls, threshold in a.data.traverse_vectors(j):
                        y_[ls] = 1

                        distance = self.distance(y[a.data.indices], y_[a.data.indices], a)
                        gain = base_distance - distance
                        if max_gain is None or max_gain < gain:
                            a_, j_, i_, threshold_ = a, j, i, threshold
                            max_gain = gain
                            def op():
                                nonlocal leaves
                                leaves.remove(a_)
                                childs = a_.split(j_, i_, None, None, threshold_)
                                leaves += [ c for c in childs if not c.is_pure() ]
                                return "split(%s, %d, %g)" % (a_.get_name(self.tree_), j_, threshold_)

            if max_gain is None:
                break
            res = op()
            if self.verbose: print(res)


class LocalClustQualDecisionTreeClassifier(LocalGenericDecisionTreeClassifier):
    def distance(self, y, y_, a):
        qs = []
        for c in np.unique(y_):
            xs = np.where(y_ == c)
            qs.append(self.quality(a.data.X[xs], y[xs]))
        return np.sum(qs)

    def quality(self, X, y):
        X_norm = X - np.min(X, axis=0)
        X_norm = X / np.max(X_norm, axis=0)
        inter_min = np.sqrt(X.shape[1])
        intra_max = 0
        for i1, y1 in enumerate(y):
            for i2, y2 in enumerate(y):
                d = np.linalg.norm(X[i1] - X[i2])
                if y1 == y2:
                    intra_max = max(intra_max, d)
                else:
                    inter_min = min(inter_min, d)
        return inter_min - intra_max


class LocalRndBoxDecisionTreeClassifier(LocalClustQualDecisionTreeClassifier):
    def quality(self, X, y):
        n, n_dim = X.shape
        ps = []
        for i in range(1000):
            xs = np.random.choice(range(n), 2)
            box = np.sort(X[xs], axis=0)
            content = filter(lambda i: all(box[0] <= X[i]) and all(X[i] <= box[1]), range(n))
            ps.append(self.purity(y[list(content)]))
        return - np.sum(ps)

    def purity(self, y):
        n = len(y)
        if n == 0:
            return 1
        count = np.unique(y, return_counts=True)[1] / n
        return np.sum(np.square(count))


class MHDecisionTreeClassifier(BaseDecisionTreeClassifier):
    def fit(self, X, y):
        self.data_ = Sorted(X, y)
        self.data_.sort()
        self.tree_ = Node(self.data_, 0)

        most_probable = None
        p_y = 0

        while True:
            if self.verbose: print("        ~ [ accuracy: %f, #nodes: %d, depth: %d, probability: %f ]" % (self.score(self.data_.X, self.data_.y_original), self.tree_.get_n_nodes(), self.tree_.get_depth(), p_y))

            ops = self.get_current_ops()
            op_y, desc = ops[np.random.randint(len(ops))]
            p_x = self.get_current_prob()
            q_y_x = 1 / len(ops)
            undo = op_y()
            p_y = self.get_current_prob()
            q_x_y = 1 / len(self.get_current_ops())
            a = min(1, p_x * q_x_y / p_x / q_y_x)
            if np.random.random() > a:
                undo()
                if self.verbose: print("rejected:", desc)
            else:
                if self.verbose: print("     acc:", desc)

                if most_probable is None or most_probable[0] < p_y:
                    most_probable = (self.score(self.data_.X, self.data_.y_original), self.tree_.get_n_nodes(), self.tree_.get_depth(), p_y)

            if most_probable is not None:
                if self.verbose: print("        T [ accuracy: %f, #nodes: %d, depth: %d, probability: %g ]" % most_probable)

    def get_current_ops(self):
        ops = []
        nodes = self.tree_.get_all_nodes()

        for a in nodes:
            if a.is_leaf:
                for j in a.data.traverse_features():
                    for i, _, threshold in a.data.traverse_vectors(j):
                        ops.append(((lambda a, j, i, threshold: lambda: self.new_split(a, j, i, threshold))(a, j, i, threshold),
                                    "new(" + a.get_name(self.tree_) + ", " + str(j) + ", " + str(threshold) + ")"))
            else:
                ops.append(((lambda a: lambda: self.remove_split(a))(a),
                            "remove(" + a.get_name(self.tree_) + ")"))
                for _, _, threshold in a.data.traverse_vectors(a.feature):
                    ops.append(((lambda a, threshold: lambda: self.move_split(a, threshold))(a, threshold),
                                "move(" + a.get_name(self.tree_) + ", " + str(threshold) + ")"))

        return ops + [(lambda: lambda: None, "nothing()")]

    def get_current_prob(self):
        p1 = self.prob(self.data_.y_original, self.predict(self.data_.X))
        p2 = max(0.1, 1 - 0.01 * self.tree_.get_n_nodes())
        return p1 * p2

    def prob(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    def move_split(self, a, threshold):
        threshold_ = a.threshold
        a.threshold = threshold

        def undo():
            a.threshold = threshold_
        return undo

    def new_split(self, a, j, i, threshold):
        a.split(j, i, None, None, threshold)
        return a.undo_split

    def remove_split(self, a):
        a.is_leaf = True

        def undo():
            a.is_leaf = False
        return undo


class LocalAreaSepDecisionTreeClassifier(LocalClustQualDecisionTreeClassifier):
    def quality(self, X, y):
        n, n_dim = X.shape
        sep_sum = 0
        for dim in range(n_dim):
            x = X[:,dim]
            y_sorted = y[np.argsort(x)]
            current_class = None
            for c in y_sorted:
                if current_class is None:
                    current_class = c
                else:
                    if current_class != c:
                        sep_sum += 1
                        current_class = c
        return (2 * sep_sum + 1) / n_dim


def test_rnd_box(y):
    ps = []
    n = len(y)
    for l in range(0, n):
        for w in range(l, n):
            xs = range(l, w+1)
            p = LocalRndBoxDecisionTreeClassifier.purity(None, np.array(y)[xs])
            ps.append(p)
    return sum(ps) / ( n * (n+1) / 2 )
