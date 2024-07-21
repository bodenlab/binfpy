import math
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import binfpy.stats as stats
from binfpy.heap import *


class PairArray:
    """Storage for all unordered pairs of indices (a, b) == (b, a) and (a != b), where both a and b are members in the set 0..N.
    Space requirement is N * N / 2 - N / 2, e.g. N=8 gives 28 cells"""

    def __init__(self, N):
        self._N = N
        self._store = [None for _ in range((N * N) // 2 - (N // 2))]

    def hash(self, idx1, idx2):
        if idx1 == idx2:
            return None
        if idx1 < idx2:
            return idx1 * self._N + idx2 - ((idx1 + 1) ** 2 // 2 + (idx1 + 2) // 2)
        else:
            return idx2 * self._N + idx1 - ((idx2 + 1) ** 2 // 2 + (idx2 + 2) // 2)

    def __setitem__(self, idxtuple, value):
        self._store[self.hash(idxtuple[0], idxtuple[1])] = value

    def __getitem__(self, idxtuple):
        return self._store[self.hash(idxtuple[0], idxtuple[1])]


class DNode:
    """A class for a node in a dendrogram for agglomerative clustering, assuming ultrametricity."""

    def __init__(self, idx=None, children=[], dist=None):
        """
        Initialise a node by the children it joins (if any).
        If linked, children already need to have been created;
        their reciprocal linkages will be created/ensured.
        Use dist to indicate the distance between each child and this node.
        :param idx: index/label of node
        :param children: the list of DNode instances making up the children (default empty list)
        :param dist: distance from children to this node
        """
        self.parent = None
        self.idx = idx
        self.children = children
        self.dist = dist
        if children:
            self.leaves = 0
            self.refs = []
            for child in children:  # link each child to this parent
                child.parent = self
                self.leaves += child.leaves
                self.refs.extend(child.refs)
        else:
            self.leaves = 1
            self.refs = [idx]

    def nChildren(self):
        """
        Check how many direct children this node has
        :return: number of children
        """
        if self.children:
            return len(self.children)
        else:
            return 0

    def getLeaves(self):
        """
        Retrieve all leaves under this node (recursively)
        :return: all nodes that do not have children themselves under this node
        """
        if self.children:
            leaves = []
            for c in self.children:
                leaves.extend(c.getLeaves())
            return leaves
        else:
            return [self]

    def getDistances(self, sort=False):
        if self.children:
            d = [self.dist]
            for child in self.children:
                d.extend(
                    child.getDistances()
                    if not sort
                    else sorted(child.getDistances(), reverse=True)
                )
            return d
        else:
            return []

    def findNode(self, idx):
        if self.leaves == 1:
            return self if idx in self.refs else None
        else:
            if idx == self.idx:
                return self
            for child in self.children:
                catch = child.findNode(idx)
                if catch:
                    return catch
            return None

    def getParents(self):
        if self.parent == None:
            return []
        else:
            pp = self.parent.getParents()
            if pp:
                pp.append(self.parent)
                return pp
            else:
                return [self.parent]

    def getNodes(self, mindist=0, internal=False):
        if self.leaves == 1:
            return [] if internal else [self]
        else:
            nodes = []
            if self.dist < mindist:
                return self
            for child in self.children:
                if child.leaves == 1 and not internal:
                    nodes.extend(child.getNodes(mindist))
                elif child.leaves > 1:
                    if child.dist >= mindist:
                        nodes.extend(child.getNodes(mindist))
                    else:
                        nodes.append(child)
            nodes.append(self)
            return nodes

    def reIndex(self, childcount, parentcount):
        if self.children:  # this is a parent node
            for c in self.children:
                (childcount, parentcount) = c.reIndex(childcount, parentcount)
            self.idx = parentcount
            for c in self.children:
                c.parent = self
            parentcount += 1
        else:  # this is a leaf node
            self.idx = childcount
            childcount += 1
        return (childcount, parentcount)

    def clone(self):
        n = DNode(self.idx, [c.clone() for c in self.children], self.dist)
        n.refs = self.refs
        return n

    def getNodesBarCondition(self, mindist=0, idxs=[]):
        if self.leaves == 1:
            return [self]
        elif self.dist < mindist or self.idx in idxs:
            return [self]
        else:
            children = []
            for c in self.children:
                mynodes = c.getNodesBarCondition(mindist, idxs)
                children.extend(mynodes)
            children.append(self)
            return children

    def cloneBarCondition(self, mindist=0, idxs=[], leakytrace=[]):
        if self.leaves == 1:
            n = DNode(self.idx)
        elif (self.dist < mindist or self.idx in idxs) and not self.idx in leakytrace:
            n = DNode(self.idx)
        else:
            children = []
            for c in self.children:
                myclone = c.cloneBarCondition(mindist, idxs, leakytrace)
                children.append(myclone)
            if len(children) > 1:
                n = DNode(self.idx, children, self.dist)
            else:
                n = DNode(self.idx)
        n.refs = self.refs
        return n

    def __repr__(self):
        return (
            "<"
            + str(self.idx)
            + "("
            + str(self.leaves)
            + "|"
            + str(len(self.refs))
            + "):"
            + str(self.dist)
            + ">"
            if self.dist
            else "<" + str(self.idx) + ">"
        )

    def __str__(self):
        """Returns string with node (incl descendants) in a Newick style."""
        stubs = ["" for _ in range(self.nChildren())]
        dist = ""
        for i in range(self.nChildren()):
            stubs[i] = str(self.children[i])
        if self.dist or self.dist == 0.0:
            dist = ":" + str(self.dist)
        label = str(self.idx)
        if self.nChildren() == 0:
            return label + dist
        else:
            stubstr = "("
            for i in range(len(stubs) - 1):
                stubstr += stubs[i] + ","
            return stubstr + stubs[-1] + ")" + label + dist

    def newick(self, labels):
        """Returns string with node (incl descendants) in a Newick style."""
        stubs = ["" for _ in range(self.nChildren())]
        label = dist = ""
        for i in range(self.nChildren()):
            stubs[i] = self.children[i].newick(labels)
        if self.dist or self.dist == 0.0:
            dist = ":" + str(self.dist)
        if self.idx != None and self.idx < len(labels):
            label = labels[self.idx]
        if self.nChildren() == 0:
            return label + dist
        else:
            stubstr = "("
            for i in range(len(stubs) - 1):
                stubstr += stubs[i] + ","
            return stubstr + stubs[-1] + ")" + label + dist

    """ ----------------------------------------------------------------------------------------
        Methods for processing files of trees on the Newick format
        ----------------------------------------------------------------------------------------"""


def parse(string, idx2idx={}):
    """
    Parse a Newick string to create a (root) node (including descendant nodes)
    :param string: Newick formatted string
    :param idx2idx: dictionary to translate indices/labels (default empty)
    :return: root node of tree described by input string
    """
    first = string.find("(")
    last = string[::-1].find(")")  # look from the back
    if first == -1 and last == -1:  # we are at leaf
        y = string.split(":")
        if y[0] in idx2idx:
            node = DNode(idx=idx2idx[y[0]])
        else:
            node = DNode(idx=y[0])
        if len(y) >= 2:
            node.dist = float(y[1])
        return node
    elif first >= 0 and last >= 0:
        # remove parentheses
        last = (
            len(string) - last - 1
        )  # correct index to refer from start instead of end of string
        embed = string[first + 1 : last]
        tail = string[last + 1 :]
        # find where corresp comma is
        commas = _findComma(embed)
        if len(commas) < 1:
            raise RuntimeError(
                'Invalid format: invalid placement of "," in sub-string "' + embed + '"'
            )
        prev_comma = 0
        child_tokens = []
        for comma in commas:
            child_tokens.append(embed[prev_comma:comma].strip())
            prev_comma = comma + 1
        child_tokens.append(embed[prev_comma:].strip())
        y = tail.split(":")
        if len(y[0]) == 0:
            node = DNode()
        elif y[0] in idx2idx:
            node = DNode(idx=idx2idx[y[0]])
        else:
            node = DNode(idx=y[0])
        if len(y) >= 2:
            node.dist = float(y[1])
        node.children = []
        node.leaves = 0
        node.refs = []
        for tok in child_tokens:
            child = parse(tok, idx2idx)
            child.parent = node
            node.leaves += child.leaves
            node.refs.extend(child.refs)
            node.children.append(child)
        return node
    else:
        raise RuntimeError(
            'Invalid format: unbalanced parentheses in sub-string "' + string + '"'
        )


def _findComma(string, level=0):
    """Find all commas at specified level of embedding"""
    mylevel = 0
    commas = []
    for i in range(len(string)):
        if string[i] == "(":
            mylevel += 1
        elif string[i] == ")":
            mylevel -= 1
        elif string[i] == "," and mylevel == level:
            commas.append(i)
    return commas


class _RegistryIterator:
    def __init__(self, registry):
        assert len(registry._inuse) > 1, "Must have at least two indices to iterate"
        self._registry = registry
        self._n = len(registry._inuse)
        self._inuse = list(registry._inuse)
        self._curi = 0
        self._curj = 1

    def __iter__(self):
        return self

    def __next__(self):
        for i in range(self._curi, self._n - 1):
            self._curj = max(self._curi + 1, self._curj)
            for j in range(self._curj, self._n):
                address = self._registry.encode(self._inuse[i], self._inuse[j])
                y = self._registry._elements[address]
                if y:
                    if j < self._n - 1:
                        self._curj = j + 1
                    else:
                        self._curi = i + 1
                        self._curj = self._curi + 1
                    return y
        else:
            raise StopIteration


class Registry:

    def __init__(self, N):
        self.N = N  # number of leaves
        self.NwP = 2 * N - 1  # number of leaves and parents
        self.NR = self.NwP**2 // 2 - self.NwP // 2  # number of records to keep
        self._elements = np.array([None for i in range(self.NR)])  # self.NR records
        self._inuse = set(
            [i for i in range(N)]
        )  # define what indices that are currently in use
        self.k = N

    def __iter__(self):
        return _RegistryIterator(self)

    def useIndex(self, idx):
        self._inuse.add(idx)

    def removeIndex(self, idx):
        self._inuse.remove(idx)

    def encode(self, idx1, idx2):
        if idx1 == idx2:
            return None
        if idx1 < idx2:
            return (
                idx1 * (2 * self.N - 1)
                + idx2
                - ((idx1 + 1) ** 2 // 2 + (idx1 + 2) // 2)
            )
        else:
            return (
                idx2 * (2 * self.N - 1)
                + idx1
                - ((idx2 + 1) ** 2 // 2 + (idx2 + 2) // 2)
            )

    def putDNode(self, dnode):
        if dnode.children[0].idx == None:
            dnode.children[0].idx = self.k
            self.k += 1
        if dnode.children[1].idx == None:
            dnode.children[1].idx = self.k
            self.k += 1
        idx1 = dnode.children[0].idx
        idx2 = dnode.children[1].idx
        address = self.encode(idx1, idx2)
        self._elements[address] = dnode

    def get(self, address):
        return self._elements[address]

    def set(self, address, value):
        self._elements[address] = value

    def getDNode(self, idx1, idx2):
        address = self.encode(idx1, idx2)
        if self._elements[address]:
            return self._elements[address]
        else:
            return None

    def decode(self, address):
        # address = idx1 * (2 * self.N - 1) + idx2 - ((idx1+1)**2 // 2 + (idx1+2) // 2)
        dnode = self._elements[address]
        if dnode:
            return [dnode.children[0].idx, dnode.children[1].idx]
        # idx1 = self._elements[address][0]
        # idx2 = address - idx1 * (2 * self.N - 1) + ((idx1+1)**2 // 2 + (idx1+2) // 2)
        return None

    def getClosest(self):
        closest = None
        try:
            for node in self:
                if closest == None or node.dist < closest.dist:
                    closest = node
        except:
            pass
        return closest


class Dendrogram:
    """A class for a dendrogram for agglomerative clustering, assuming ultrametricity"""

    def __init__(self):
        self.Labels = []  # Labels, if available
        self.Data = []  # Data, if available
        self.BaseTree = None  # Root of base tree/dendrogram, if available
        self.defaultview = "original"  # current, default view is 'original'
        self.views = {self.defaultview: None}  # root of each view
        self.means = {}  # dict frozenset/refs : list/v

    def loadBaseTree(self, filename):
        """Read file on Newick format and set as base tree in this dendrogram."""
        f = open(filename)
        string = "".join(f)
        self.BaseTree = self.parseNewick(string)
        self.N = k = len(self.Labels)
        # re-indexed, viewable version
        self.BaseTree.reIndex(0, self.N)
        self.views[self.defaultview] = self.BaseTree
        self.setView(self.defaultview)

    def parseNewick(self, string):
        """Main method for parsing a Newick string into a tree.
        Handles labels (on both leaves and internal nodes), and includes distances (if provided).
        Returns the root node, which defines the tree."""
        if string.find(";") != -1:
            string = string[: string.find(";")]
        return parse(string)

    def saveBaseTree(self, filename):
        """
        Save the base tree that defines the dendrogram; not that distances are in the tree but not in the form of a data matrix
        :param filename: name of file, under which the dendrogram base tree is saved
        """
        assert self.BaseTree and self.Labels != None, "Invalid tree: cannot save"
        with open(filename, "w") as fh:
            print(self.BaseTree.__str__(self.Labels), end="", file=fh)

    def loadData(self, filename, headers=False):
        """
        Load the data associated with this dendrogram.
        :param filename: the name of the tab-separated value file
        :param headers: if headers are present in the file, default is False
        """
        reindex = False
        if (
            self.Labels and self.BaseTree
        ):  # labels are already available for base tree, so data need to be re-indexed
            reindex = True
            self.Data = [None for _ in range(len(self.Labels))]
        with open(filename) as csvfile:
            r = csv.reader(csvfile, delimiter="\t")
            headers = [] if headers else None
            for row in r:
                if headers == []:
                    headers = row[1:]
                elif reindex:
                    self.Data[self.indexLabel(row[0])] = [float(y) for y in row[1:]]
                else:
                    self.Data.append([float(y) for y in row[1:]])
                    self.Labels.append(row[0])

    def setData(self, data, labels):
        """
        Assign data vectors to labels in this dendrogram; will match with pre-existing tree and overwrite current values if any
        :param data: data vectors
        :param labels: labels (same index as data)
        """
        assert len(data) == len(
            labels
        ), "Number of data rows and labels must be the same"
        reindex = False
        if (
            self.Labels and self.BaseTree
        ):  # labels are already available for base tree, so data need to be re-indexed
            reindex = True
            self.Data = [None for _ in range(len(self.Labels))]
        for i in range(len(data)):
            if reindex:
                idx = self.indexLabel(labels[i])
                self.Data[idx] = data[idx]
            else:
                self.Data.append(data[i])
                self.Labels.append(labels[i])

    def normData(self, mode=None):
        """
        Normalise data
        :param mode: normalisation mode, default is None (unit length)
        """
        for i in range(len(self.Data)):
            self.Data[i] = normalise(self.Data[i])

    def createBaseTree(self, d, labels=None):
        "d is the distance matrix, labels is the list of labels"
        self.N = len(d)
        self.Labels = labels or ["X" + str(i + 1) for i in range(self.N)]
        registry = Registry(self.N)
        " For each node-pair, assign the distance between them. "
        dnodes = [DNode(idx=i) for i in range(self.N)]
        for i in range(self.N):
            registry.useIndex(i)
            for j in range(i + 1, self.N):
                registry.putDNode(DNode(children=[dnodes[i], dnodes[j]], dist=d[i, j]))
        # Treat each node as a cluster, until there is only one cluster left, find the closest pair
        # of clusters, and merge that pair into a new cluster (to replace the two that merged).
        # In each case, the new cluster is represented by the node that is formed.
        closest = registry.getClosest()
        while closest:
            # So we know the closest, now we need to merge...
            x = closest.children[0]
            y = closest.children[1]
            z = closest  # use this node for new cluster z
            Nx = x.leaves  # find number of sequences in x
            Ny = y.leaves  # find number of sequences in y
            for widx in registry._inuse:  # for each node w ...
                if widx != x.idx and widx != y.idx:
                    # we will merge x and y into a new cluster z, so need to consider w (which is not x or y)
                    dxw_address = registry.encode(x.idx, widx)
                    dyw_address = registry.encode(y.idx, widx)
                    dxw = registry.get(
                        dxw_address
                    )  # retrieve and remove distance from D: x to w
                    dyw = registry.get(
                        dyw_address
                    )  # retrieve and remove distance from D: y to w
                    w = (
                        dxw.children[0]
                        if dxw.children[0].idx == widx
                        else dxw.children[1]
                    )
                    dzw = DNode(
                        children=[z, w],
                        dist=(Nx * dxw.dist + Ny * dyw.dist) / (Nx + Ny),
                    )  # distance: z to w
                    registry.set(dxw_address, None)
                    registry.set(dyw_address, None)
                    registry.putDNode(dzw)
            registry.useIndex(z.idx)
            # remove x and y from registry
            registry.removeIndex(x.idx)
            registry.removeIndex(y.idx)
            closest = registry.getClosest()
        self.BaseTree = z
        self.BaseTree.reIndex(0, self.N)
        # re-indexed, viewable version
        self.views[self.defaultview] = self.BaseTree
        self.setView(self.defaultview)

    def getRoot(self, view=None):
        if not view:
            view = self.defaultview
        return self.views[view]

    def getIndex(self, label):
        for i in range(len(self.Labels)):
            if self.Labels[i] == label:
                return i
        return -1

    def getNodeByIndex(self, idx, view=None):
        if idx != -1:
            return self.getRoot(view).findNode(idx)
        else:
            return None

    def getNodeByName(self, label, view=None):
        idx = self.getIndex(label)
        if idx != -1:
            return self.getNodeByIndex(idx, view)
        else:
            return None

    def _intersect(self, L1, L2):
        return [y for y in L1 if y in L2]

    def getTraceByMembers(self, members=[], view=None):
        return self.getRootByMembers(members, view, format="trace")

    def getRootByMembers(self, members=[], view=None, format="root"):
        """Retrieve the root of the smallest sub-tree that contains all the members."""
        prevtrace = None
        for member in members:
            if not isinstance(member, DNode):
                name = member
                member = self.getNodeByName(name, view)
                if member == None:
                    print("Could not find any node with label", name)
            trace = member.getParents()
            if prevtrace:
                prevtrace = self._intersect(prevtrace, trace)
            else:
                prevtrace = trace
        if format == "root" and len(prevtrace) >= 1:
            return prevtrace[-1]
        elif format == "trace":
            return prevtrace
        else:
            return None

    def setView(self, name):
        self.defaultview = name

    def createView(self, name, newroot=None, mindist=0, terminate_nodes=[]):
        idxs = [n.idx for n in terminate_nodes]
        newroot = newroot or self.getRoot()
        newtree = newroot.cloneBarCondition(mindist, idxs)
        newtree.reIndex(0, len(newtree.getLeaves()))
        self.views[name] = newtree
        self.setView(name)

    def createLeakyView(self, name, trace=[], mindist=0):
        """
        Create view from specified minimum distance but all a trace of parent nodes for a subtree to leak through
        :param name: name of view
        :param trace: list of nodes that all lead to a subtree
        :param mindist: the minimum distance for terminating nodes
        :return:
        """
        idxs = [n.idx for n in trace]
        newtree = self.getRoot().cloneBarCondition(mindist, leakytrace=idxs)
        newtree.reIndex(0, len(newtree.getLeaves()))
        self.views[name] = newtree
        self.setView(name)

    def getNodesByEnrichment(
        self, positiveLabels=[], view=None, pval=0.05, format="nodes only"
    ):
        """
        Get all nodes in the view which are positively enriched for the specified labels.
        Assumes that the tree is rooted.
        :param positiveLabels: labels
        :param view: view
        :param pval: maximum p-value for the test to return node
        :param format: "nodes only" means a sorted list with nodes, "with p" means a sorted list with tuples (node, p)
        :return: a list of nodes or (node, p) tuples, sorted in order of ascending p-value (most significant first), truncated at max p-value
        """
        root = self.getRoot(view)
        # set-up reference sets
        pos = set()
        neg = set()
        for idx in root.refs:
            label = self.Labels[idx]
            if label in positiveLabels:
                pos.add(idx)
            else:
                neg.add(idx)
        if len(pos) == 0:
            return []
        # gather then try all nodes in the view
        nodes = root.getNodes()
        res = dict()
        for n in nodes:
            a1 = a2 = 0
            for idx in n.refs:
                if idx in pos:
                    a1 += 1
                elif idx in neg:
                    a2 += 1
            b1 = len(pos) - a1
            b2 = len(neg) - a2
            p = stats.getFETpval(a1, a2, b1, b2, False)
            if p < pval:
                res[n] = p
        sorted_tuples = sorted(res.items(), key=lambda kv: kv[1])
        sorted_list = [y[0] for y in sorted_tuples]
        return sorted_list if format == "nodes only" else sorted_tuples

    def getLinkage(self, nodes):
        links = []
        for n in nodes:
            if n.leaves > 1:
                links.append([n.children[0].idx, n.children[1].idx, n.dist, n.leaves])
        return links

    black = "000000"
    red2yellow = [
        "200000",
        "400000",
        "550000",
        "5f0000",
        "690000",
        "730000",
        "7d0000",
        "870000",
        "910000",
        "9b0000",
        "a50000",
        "af0000",
        "b90000",
        "c30000",
        "cd0000",
        "d70000",
        "e10000",
        "eb0000",
        "f50000",
        "FF0000",
        "FF0a00",
        "FF1400",
        "FF1e00",
        "FF2800",
        "FF3200",
        "FF3c00",
        "FF4600",
        "FF5000",
        "FF5a00",
        "FF6400",
        "FF6e00",
        "FF7800",
        "FF8200",
        "FF8c00",
        "FF9600",
        "FFa000",
        "FFaa00",
        "FFb400",
        "FFbe00",
        "FFc800",
        "FFd200",
        "FFdc00",
        "FFe600",
        "FFf000",
        "FFfa00",
    ]

    def getHighlight(self, nodes, scored_nodes):
        mymax = None
        mymin = None
        for node, score in scored_nodes:
            if not mymax or not mymin:
                mymax = mymin = score
            else:
                mymax = max(mymax, score)
                mymin = min(mymin, score)
        rate = (mymax - mymin) / len(self.red2yellow)
        print("Min", mymin, "Max", mymax, "Rate", rate)
        node2score = dict()
        for node, score in scored_nodes:
            node2score[node] = score
        color_link = {}
        for n in nodes:  # a node
            if n.leaves > 1:  # only internal nodes can be colored
                try:
                    score = node2score[n]
                    step = int((score - mymin) / rate)
                    color_link[n.idx] = "#" + self.red2yellow[step]
                    print(step, end=" ")
                except:
                    color_link[n.idx] = "#" + self.black
        print()
        return color_link

    def getRGB(self, nodes, colored_nodes=[], defaultcol="B0B0B0"):
        assert len(colored_nodes) == 3, "Must have three lists of nodes to use RGB"
        allcolored = set([])
        for color in colored_nodes:
            allcolored.update(color)
        color_link = {}
        for n in nodes:  # a node
            if n.leaves > 1:  # only internal nodes can be colored
                color_link[n.idx] = "#"
                if n in allcolored:
                    inall = True
                    for color in colored_nodes:  # lists for Red, Green then Blue
                        if n in color:
                            color_link[n.idx] += "E0"
                        else:
                            color_link[n.idx] += "00"
                            inall = False
                    if inall:
                        color_link[n.idx] = "#101010"
                else:
                    color_link[n.idx] += defaultcol
        return color_link

    def getRefs(self, nodes):
        """
        Get the group of indices for each node
        :param nodes: the nodes of interest
        :return: list of reference index lists
        """
        groups = []
        for n in nodes:
            if n.leaves == 1:
                groups.append(n.refs)
        return groups

    def getMeanRefs(self, refs):
        """
        Determine the average vector for a group of reference indices
        :param refs: the index group
        :return: the average vector
        """
        if self.Data:
            key = frozenset(refs)
            if key in self.means:
                return self.means[key]
            m = None
            for r in refs:
                if not m:
                    m = [y for y in self.Data[r]]
                else:
                    for i in range(len(m)):
                        m[i] += self.Data[r][i]
            m = [y / len(refs) for y in m]
            self.means[key] = m
            return m
        else:
            return None

    def getLabelGroups(self, nodes):
        """
        Get the list of labels, associated with each node
        :param nodes: nodes of interest
        :return: the list of label lists
        """
        groups = []
        for n in nodes:
            if n.leaves == 1:
                plabels = []
                for i in range(len(n.refs)):
                    plabels.append(self.Labels[n.refs[i]])
                groups.append(plabels)
        return groups

    def getString(self, v, s, thresholds):
        assert len(s) - 1 == len(
            thresholds
        ), "characters in string s must agree with thresholds + 1"
        ret = ""
        for y in v:
            for i in range(len(thresholds)):
                if y < thresholds[i]:
                    ret += s[i]
                    break
            if y >= thresholds[-1]:
                ret += s[-1]
        return ret

    def draw(
        self,
        view=None,
        maxlabel=15,
        mode="label",
        thresholds=None,
        nodesRGB=None,
        nodesHighlight=None,
    ):
        N = self.getRoot(view).getNodes()
        groups_wlabels = self.getLabelGroups(N)
        groups_windices = self.getRefs(N)
        labs = ["" for _ in groups_wlabels]
        syms = " .o0"
        if thresholds or "vec" in mode:
            for i in range(len(groups_windices)):
                group = groups_windices[i]
                meanv = self.getMeanRefs(group)
                if thresholds:
                    labs[i] += self.getString(meanv, syms, thresholds)
                else:
                    for y in meanv:
                        labs[i] += "%4.2f " % y
        if "lab" in mode:
            for i in range(len(groups_wlabels)):
                group = groups_wlabels[i]
                label = "" if len(labs[i]) == 0 else " "
                label += "" if len(group) == 1 else str(len(group)) + "|"
                for jj in range(len(group)):
                    j = len(group) - 1 - jj
                    label += group[j] + ("" if j == 0 else ":")
                labs[i] += label
        for i in range(len(labs)):
            if len(labs[i]) > maxlabel:
                labs[i] = labs[i][0 : maxlabel - 2] + ".."
        L = self.getLinkage(N)
        Z = np.array(L)
        plt.rcParams["font.family"] = "Andale Mono"
        if not nodesRGB and not nodesHighlight:
            hierarchy.dendrogram(Z, labels=labs, orientation="left")
        elif nodesRGB:
            rgb = [[] for _ in range(3)]
            for i in range(len(nodesRGB)):
                rgb[i] = nodesRGB[i]
            link_cols = self.getRGB(N, rgb)
            hierarchy.dendrogram(
                Z,
                labels=labs,
                orientation="left",
                link_color_func=lambda row: link_cols[row],
            )
        else:  # highlight
            highlight = [(node, -math.log(p)) for (node, p) in nodesHighlight]
            link_cols = self.getHighlight(N, highlight)
            hierarchy.dendrogram(
                Z,
                labels=labs,
                orientation="left",
                link_color_func=lambda row: link_cols[row],
            )

    def indexLabel(self, label):
        if self.Labels == None:
            self.Labels = []
        for i in range(len(self.Labels)):
            if self.Labels[i] == label:
                return i
        self.Labels.append(label)
        return len(self.Labels) - 1


"""

"""


def eucdist(v1, v2):
    diff = 0
    for i in range(len(v1)):
        diff += (v1[i] - v2[i]) ** 2
    return math.sqrt(diff)


def cosdist(v1, v2):
    if len(v1) != len(v2):
        return None
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(len(v1)):
        sum0 += v1[i] * v2[i]
        sum1 += v1[i] * v1[i]
        sum2 += v2[i] * v2[i]
    return 1 - (sum0 / (math.sqrt(sum1 * sum2)))


def normalise(v):
    y = sum(v)
    return [vi / y for vi in v]


def euc(data):
    d = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            d[i, j] = eucdist(data[i], data[j])
    return d


def norm(data):
    ndata = [[] for _ in data]
    for i in range(len(data)):
        ndata[i] = normalise(data[i])
    return ndata


import csv

if __name__ == "__main__0":
    dgram = Dendrogram()
    dgram.loadData("/Users/mikael/simhome/monocyte/test100x10.tsv", headers=True)
    d = euc(dgram.Data)
    dgram.createBaseTree(d, dgram.Labels)
    # dgram.createView('focus slamf7', dgram.getNodesByMembers(['p3@SLAMF7', 'p2@NFKB2'], view = 'original'))
    dgram.createView("focus sorl1", dgram.getRootByMembers(["p2@SORL1", "p1@FAU"])[-1])
    # dgram.createView('focus ddx6', dgram.getNodesByMembers(['p3@DDX6', 'p2@CARS']))
    dgram.createView("focus sorl1 dist", mindist=0.04)
    plt.figure(num=None, figsize=(5, 15), dpi=200, facecolor="w", edgecolor="k")
    # dgram.draw('focus ddx6', mode = 'label')
    dgram.draw("focus sorl1 dist", mode="vec")
    plt.show()

if __name__ == "__main__1":
    dgram = Dendrogram()
    dgram.loadData("/Users/mikael/simhome/CAGE_iPSC/de_promoter.tsv", headers=False)
    d = euc(norm(dgram.Data))
    dgram.createBaseTree(d, dgram.Labels)
    dgram.saveBaseTree("/Users/mikael/simhome/CAGE_iPSC/de_promoter.dgram")
    # plt.figure(num=None, figsize=(5, 15), dpi=200, facecolor='w', edgecolor='k')
    # dgram.draw()
    # plt.show()

if __name__ == "__main__2":
    dgram = Dendrogram()
    dgram.loadData("/Users/mikael/simhome/monocyte/2016_03_22de.tsv", headers=True)
    dgram.loadBaseTree("/Users/mikael/simhome/monocyte/2016_03_22de.dgram")
    dgram.normData()
    dgram.createView("mindist=0.02", mindist=0.02)
    dgram.createView(
        "focus nfkb2", dgram.getRootByMembers(["p1@NFKB2", "p2@NFKB2"]), mindist=0.05
    )
    enriched = dgram.getNodesByEnrichment(
        ["p2@NFKB2", "p3@NFKB2", "p1@PDE4B", "p2@PDE4B", "p4@PDE4B", "p6@PDE4B"]
    )
    print(enriched)
    dgram.createView(
        "enriched_truncated",
        mindist=enriched[0].dist - 0.05,
        terminate_nodes=[enriched[0]],
    )
    dgram.setView("focus nfkb2")
    plt.figure(num=None, figsize=(8, 15), dpi=300, facecolor="w", edgecolor="k")
    dgram.draw(
        "focus nfkb2",
        maxlabel=25,
        mode="label",
        thresholds=[0.05, 0.08, 0.12],
        nodesRGB=[enriched],
    )
    plt.show()

if __name__ == "__main__":
    data = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 1.9],
        [2.1, 2.1, 2.1],
        [3.1, 3.2, 3.3],
        [3.2, 3.4, 3.6],
        [3.5, 3.4, 3.3],
        [0.9, 0.9, 1.0],
    ]
    labels = ["S1", "S2a", "S2b", "S3a", "S3b", "S3c", "S0"]
    dgram = Dendrogram()
    dgram.createBaseTree(euc(data), labels)
    dgram.draw()
    plt.show()

if __name__ == "__main__4":
    data = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 1.9],
        [2.1, 2.1, 2.1],
        [3.1, 3.2, 3.3],
        [3.2, 3.4, 3.6],
        [3.5, 3.4, 3.3],
        [0.9, 0.9, 1.0],
    ]
    labels = ["S1", "S2a", "S2b", "S3a", "S3b", "S3c", "S0"]
    dgram = Dendrogram()
    dgram.loadBaseTree("/Users/mikael/simhome/monocyte/small.dgram")
    dgram.setData(data, labels)
    myroot = dgram.getRootByMembers(["S0", "S2a"])
    print(myroot)
    dgram.createViewFromRoot("nroot", myroot)
    dgram.draw()
    plt.show()
