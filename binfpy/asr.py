from binfpy.sym import *
import itertools
import sympy
import math

"""
Classes for basic ASR based on an underlying Bayesian network. Does not perform inference of tree topology. Currently 
only supports small number of sequences (nodes) as ancestral character inference is not heuristic.

Please be aware that this code is mostly bare bones and errors are not handled comprehensively.
 """


class SubstModel:
    """Time-dependent substitution model using CT Markov chains."""

    def __init__(self, alpha, F, IRM):
        self.alpha = alpha  # Allowable states for characters at nodes, order corresponding to parameters in F and IRM
        self.F = F  # Equilibrium frequencies of characters
        self.IRM = IRM  # Instantaneous rate matrix

        # Set up Q matrix for this model
        Q = fillLower(F, IRM)
        Q = makeValid(Q)
        Q = normalise(Q)
        self.Q = Q

    def __eq__(self, other):
        return self.IRM == other.IRM

    def __hash__(self):
        return hash(tuple(self.IRM))

    def getTransitionMatrix(self, t):
        """Compute the transition probability matrix for a branch length, t
        NOTE: Multiplication of matrices is matrix multiplication (not element-wise multiplication)
        """

        # Diagonalise Q matrix in the form Q=UDU^-1
        U, D = self.Q.diagonalize()  # D contains eigenvalues of Q

        # E(t) = exp(Dt)
        E = sympy.zeros(D.rows)
        for i in range(D.rows):
            E[i, i] = math.exp(D[i, i] * t)

        # P(t) = U*E(t)*U^-1
        P = U * E * U**-1

        return P

    def priorProb(self, x):
        """Return the prior probability of a given character. Throws error if x is not in the model's alphabet."""
        try:
            idx = self.alpha.index(x)
        except ValueError:
            raise RuntimeError(
                f"Character {x} is not in the substitution model's alphabet."
            )

        return self.F[idx]

    def condProb(self, parent, child, t):
        """Return the probability that a parent state transitions to a child state in time t."""
        try:
            parent_idx = self.alpha.index(parent)
            child_idx = self.alpha.index(child)
        except ValueError:
            raise RuntimeError(
                f"Either {parent} or {child} is not in the substitution model's alphabet."
            )

        # Check for cached TM for given model and t. Models are assumed equal if IRMs are identical
        global tmCache  # Stores any transition probability matrix computed for any model
        try:
            if self in tmCache.keys():
                if not t in tmCache[self].keys():
                    tmCache[self][t] = self.getTransitionMatrix(t)
            else:
                tmCache[self] = {}
                tmCache[self][t] = self.getTransitionMatrix(t)
        except NameError:
            tmCache = {}
            tmCache[self] = {}
            tmCache[self][t] = self.getTransitionMatrix(t)

        return tmCache[self][t][parent_idx, child_idx]


class PhyloBNet:
    """Discrete Bayesian network. Implementation here is specifically used for representation of discrete characters
    (such as nt/AA sequence) in phylogenetic trees. Most methods currently support only networks of bifurcating nodes.
    """

    def __init__(self, root):
        self.root = root
        self.alpha = root.alpha
        self.nodes = [root]
        self.model = root.model

    def addNode(self, node):
        """Adds a node to the network's list."""
        self.nodes.append(node)
        node.network = self

    def addNodes(self, nodes):
        """Add a list of nodes to the network."""
        for node in nodes:
            self.addNode(node)

    def getNode(self, label):
        """Return a node with the given label. Returns None if node is not found."""
        for node in self.nodes:
            if node.label == label:
                return node
        return None

    def getNodes(self):
        """Return a list of all nodes under the root."""
        return self.nodes

    def getChildrenOf(self, node):
        """Return the immediate children of a given node. Input can be the node instance or label of parent
        Returns None if query node is not found. Returns empty list if node is a leaf.
        """
        if isinstance(node, str):  # If label is provided, find corresponding node
            found = False
            for node1 in self.nodes:
                if node == node1.label:
                    node = node1
                    found = True
            if not found:
                return None
        else:
            if not isinstance(node, PhyloBNode):
                raise RuntimeError("Query node must be a label or PhyloBNode object.")

        children = []
        for node1 in self.nodes:
            if node1.isRoot():
                continue
            elif node1.parent.label == node.label:
                children.append(node1)

        return children

    def getLL(self, annots):
        """Calculate and return the log-likelihood for a combination of annotations to unknown nodes, given known
        annotations and the specified substitution model. annots should be supplied as a dictionary {node:annot}.
        Unknown nodes for which annotations are not provided will be marginalised."""

        # Ensure tempAnnots for all nodes in network are initially cleared
        for node in self.getNodes():
            node.clearAnnot(temp=True)

        unknown = [
            node for node in self.getNodes() if node.getAnnot() is None
        ]  # List of nodes w/out known annotation
        for node in annots:  # Temporarily annotate unknown nodes in network
            if node not in unknown:
                raise RuntimeError(
                    "Node: "
                    + node.label
                    + " is either already annotated or is not in network."
                )
            node.annotate(annots[node], temp=True)

        # Marginalisation to be implemented in future versions
        # Check for nodes in network w/out either annot or tempAnnot (these need to be marginalised)
        # sumOver = [node for node in self.getNodes() if not (node.getAnnot(temp=True) or node.getAnnot(temp=False))]

        # Make likelihood calculation for network by determining all relevant conditional probabilities
        LL = 0
        for node in self.getNodes():
            if node.annot is None:
                annot = node.tempAnnot
            else:
                annot = node.annot
            if node.isRoot():
                LL += math.log(self.model.priorProb(annot))
            else:
                parentAnnot = node.parent.tempAnnot
                t = node.distance
                LL += math.log(self.model.condProb(parentAnnot, annot, t))

        return LL

    def getMLJoint(self, reduced=True):
        """Return the maximum likelihood set of joint annotations to all unknown nodes. Implementation is currently
        by brute force (checks all potential combinations - no heuristics). If reduced, a minimal alphabet is used
        comprising only of characters observed on at least one known node.
        Returns a dictionary of node labels mapped to inferred annotations, and the overall LL of the ML network.
        Note: this method does NOT subsequently annotate unknown nodes with the inferred maximally likely characters.
        """
        if reduced:
            symbols = set(
                [node.annot for node in self.getNodes() if node.annot is not None]
            )
        else:
            symbols = self.alpha

        # Determine the set of joint annotations to unknown nodes for testing
        unknown = [node for node in self.getNodes() if node.annot is None]
        combos = itertools.product(symbols, repeat=len(unknown))

        # Brute force max-likelihood annotation to unknown nodes
        maxLL = -math.inf  # Store best LL
        bestCombo = None  # Store combination of annotation to unknown nodes with current best LL
        for combo in combos:  # For each possible combination of annotations, get the LL
            LL = self.getLL({unknown[i]: combo[i] for i in range(len(unknown))})
            if LL > maxLL:
                maxLL = LL
                bestCombo = combo

        # Match node label to inferred character at said node
        combo_labels = {unknown[i].label: bestCombo[i] for i in range(len(unknown))}

        return combo_labels, maxLL


class PhyloBNode:
    """Discrete-valued node in a Bayesian network. Can be used to represent a phylogenetic internal or leaf node.
    Implementation supports at most one parent node.
    """

    def __init__(
        self,
        model,
        parent=None,
        distance=None,
        network=None,
        alpha=Protein_Alphabet,
        annot=None,
        label=None,
    ):
        self.model = model
        self.parent = parent
        self.distance = (
            distance  # 'Distance' from parent - akin to time in phylogenetic tree
        )
        self.network = network  # Updated when added to a Bayesian network
        self.alpha = alpha  # Allowable alphabet for annotations (e.g. sequence) to node
        self.annot = annot  # Known annotation (e.g. sequence) at this node
        self.label = label  # Name of node
        self.tempAnnot = None  # For use at unknown nodes during likelihood calculations

        if self.annot and self.annot not in self.alpha:
            raise RuntimeError("Illegal annotation to node: " + label)

    def isRoot(self):
        """Does this appear to be a root node?"""
        if not self.parent:
            return True
        else:
            return False

    def getDescendants(self, transitive=False):
        """Return descendants of this node. If transitive, recursively finds ALL nodes under this in network. Else
        just returns children of this node. Returns empty list if this node has no children.
        """
        descendants = [child for child in self.network.getChildren(self.label)]
        if transitive:
            for child in self.chidren:
                descendants.extend(child.getDescendants(transitive=True))
            return descendants
        else:
            return descendants

    def getAnnot(self, temp=False):
        if temp:
            return self.tempAnnot
        else:
            return self.annot

    def annotate(self, annot, temp=False):
        """Update annotation of this node. If temp, annotation is temporary, such as for use in likelihood calculation
        for unknown nodes in a network."""
        if annot not in self.alpha:
            raise RuntimeError(
                "Illegal annotation to node: " + self.label + ": " + annot
            )
        if temp:
            self.tempAnnot = annot
        else:
            self.annot = annot

    def clearAnnot(self, temp=False):
        if temp:
            self.tempAnnot = None
        else:
            self.annot = None


########################################################################################################################
# Operations for matrices in time-reversible substitution models. All matrices/vectors are sympy.Matrix objects
########################################################################################################################


def fillLower(F, IRM):
    """If the provided IRM is upper-triangular, need to complete with entries in lower triangle such that IRM*F is a
    zero vector."""

    if not len(F) == IRM.rows:
        raise RuntimeError(
            "Number of rows in IRM must be equal to the length of the equilibrium vector."
        )

    Q = IRM
    # Populate matrix below diagonal - assume time-reversibility so Q[j,i]=(F[i]/F[j])*Q[i,j]
    for i in range(Q.rows):
        for j in range(i + 1, Q.cols):
            Q[j, i] = (F[i] / F[j]) * Q[i, j]

    return Q


def makeValid(Q):
    """Ensure row sums of Q are 0 by equating entries in the diagonal to the negative sum of all other elements in
    the corresponding row."""

    for i in range(Q.rows):
        Q[i, i] = 0  # Ensure diagonal entry is initially 0
        rowSum = 0
        for j in range(Q.cols):
            rowSum += Q[i, j]
        Q[i, i] = -rowSum

    return Q


def normalise(Q):
    """Normalise Q such that it represents rates of transition per one unit time."""

    diag_sum = 0  # Sum of diagonal should be 1
    for i in range(Q.rows):
        diag_sum += Q[i, i]

    for i in range(Q.rows):
        for j in range(Q.cols):
            Q[i, j] = Q[i, j] / -diag_sum

    return Q


########################################################################################################################
# Data for pre-defined substitution models (only JTT is currently implemented)
# Note: Ordering of F and IRM must correspond to alphabet order for a given model
########################################################################################################################

# Model alphabets
ALPHA = {
    "JTT": [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
}

# Equilibrium frequency vectors
F = {
    "JTT": sympy.Matrix(
        [
            [
                0.076862,
                0.051057,
                0.042546,
                0.051269,
                0.020279,
                0.041061,
                0.061820,
                0.074714,
                0.022983,
                0.052569,
                0.091111,
                0.059498,
                0.023414,
                0.040530,
                0.050532,
                0.068225,
                0.058518,
                0.014336,
                0.032303,
                0.066374,
            ]
        ]
    )
}

# Instantaneous rate matrices - Only JTT available currently
# Operations on IRMs here assume the matrix is upper-triangular.
IRM = {
    "JTT": sympy.Matrix(
        [
            # A         R         N         D         C   	   Q         E         G         H         I  	     L         K         M         F         P         S         T         W         Y         V
            [
                0.000000,
                0.531678,
                0.557967,
                0.827445,
                0.574478,
                0.556725,
                1.066681,
                1.740159,
                0.219970,
                0.361684,
                0.310007,
                0.369437,
                0.469395,
                0.138293,
                1.959599,
                3.887095,
                4.582565,
                0.084329,
                0.139492,
                2.924161,
            ],
            [
                0.000000,
                0.000000,
                0.451095,
                0.154899,
                1.019843,
                3.021995,
                0.318483,
                1.359652,
                3.210671,
                0.239195,
                0.372261,
                6.529255,
                0.431045,
                0.065314,
                0.710489,
                1.001551,
                0.650282,
                1.257961,
                0.235601,
                0.171995,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                5.549530,
                0.313311,
                0.768834,
                0.578115,
                0.773313,
                4.025778,
                0.491003,
                0.137289,
                2.529517,
                0.330720,
                0.073481,
                0.121804,
                5.057964,
                2.351311,
                0.027700,
                0.700693,
                0.164525,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.105625,
                0.521646,
                7.766557,
                1.272434,
                1.032342,
                0.115968,
                0.061486,
                0.282466,
                0.190001,
                0.032522,
                0.127164,
                0.589268,
                0.425159,
                0.057466,
                0.453952,
                0.315261,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.091304,
                0.053907,
                0.546389,
                0.724998,
                0.150559,
                0.164593,
                0.049009,
                0.409202,
                0.678335,
                0.123653,
                2.155331,
                0.469823,
                1.104181,
                2.114852,
                0.621323,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                3.417706,
                0.231294,
                5.684080,
                0.078270,
                0.709004,
                2.966732,
                0.456901,
                0.045683,
                1.608126,
                0.548807,
                0.523825,
                0.172206,
                0.254745,
                0.179771,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                1.115632,
                0.243768,
                0.111773,
                0.097485,
                1.731684,
                0.175084,
                0.043829,
                0.191994,
                0.312449,
                0.331584,
                0.114381,
                0.063452,
                0.465271,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.201696,
                0.053769,
                0.069492,
                0.269840,
                0.130379,
                0.050212,
                0.208081,
                1.874296,
                0.316862,
                0.544180,
                0.052500,
                0.470140,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.181788,
                0.540571,
                0.525096,
                0.329660,
                0.453428,
                1.141961,
                0.743458,
                0.477355,
                0.128193,
                5.848400,
                0.121827,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                2.335139,
                0.202562,
                4.831666,
                0.777090,
                0.098580,
                0.405119,
                2.553806,
                0.134510,
                0.303445,
                9.533943,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.146481,
                3.856906,
                2.500294,
                1.060504,
                0.592511,
                0.272514,
                0.530324,
                0.241094,
                1.761439,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.624581,
                0.024521,
                0.216345,
                0.474478,
                0.965641,
                0.089134,
                0.087904,
                0.124066,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.436181,
                0.164215,
                0.285564,
                2.114728,
                0.201334,
                0.189870,
                3.038533,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.148483,
                0.943971,
                0.138904,
                0.537922,
                5.484236,
                0.593478,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                2.788406,
                1.176961,
                0.069965,
                0.113850,
                0.211561,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                4.777647,
                0.310927,
                0.628608,
                0.408532,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.080556,
                0.201094,
                1.143980,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.747889,
                0.239697,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.165473,
            ],
            [
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
                0.000000,
            ],
        ]
    )
}

# Instantiate pre-defined models
MODELS = {"JTT": SubstModel(ALPHA["JTT"], F["JTT"], IRM["JTT"])}
