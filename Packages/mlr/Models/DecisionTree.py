from typing import List

from tqdm import trange, tqdm
import torch


class Node:

    def __init__(self, left=None, right=None, attribute: int=None, value: float=None, label: float=None) -> None:
        """ Instantiate member of Node class

        Args:
            left: left child
            right: right child
            attribute: column of dataset to split on
            value: value of attribute to split on
            label: majority class label at this node
        """
        self.right, self.left = right, left
        self.attribute = attribute
        self.value = value
        self.label = label


class DecisionTreeClassifier:

    def __init__(self, maxDepth: int=None, leafSize: int=1) -> None:
        """ Instantiate member of DecisionTreeClassifier class


        Args:
            maxDepth: maximum depth to allow when building tree
            leafSize: minimum number of members required to split a node 
        """
        self.maxDepth = maxDepth
        self.leafSize = leafSize
        self.root = None


    def fit(self, x: torch.Tensor, y: torch.Tensor, classes: List[float]) -> None:
        """ Fit decision tree to dataset

        Args:
            x: training data
            y: labels corresponding to x
            classes: list of unique possible labels
        """
        self.root = buildTree(x, y, classes, leafSize=self.leafSize, maxDepth=self.maxDepth)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Predict label given input

        Args:
            x: input data

        Returns:
            tensor of predicted labels
        """

        y = []
        for ridx in range(x.shape[0]):
            
            curnode = self.root 
            while not (curnode.left is None and curnode.right is None):
                if x[ridx, curnode.attribute] <= curnode.value: curnode = curnode.left
                else: curnode = curnode.right

            y.append(curnode.label)            
        return torch.Tensor(y)[:, None]



def calcGiniIndex(counts: dict, size: int, classes: List[float]) -> float:
    """ Calculate gini index value given 

    Args:
        counts: dictionary of counts for each class
        size: total size of dataset being examined
        classes: list of unique classes

    Returns:
        gini index value
    """

    probs = { c: (counts[c] / size) for c in classes }
    gini = sum([(counts[c]/size) * (1 - (counts[c]/size)) for c in classes])
    return gini


def splitValue(x: torch.Tensor, y: torch.Tensor, attribute: int, classes: List[float]) -> (float, float):
    """ Determine best split value given an attribute

    Args
        x: input dataset
        y: input dataset labels
        attribute: column to find best split value on
        classes: list of possible unique labels

    Returns:
        the best split value and the gini index of the split at that value
    """

    value, purity = None, float("inf")

    # Initialize counts for splits c1 and c2 to calculate rolling gini index
    c1 = { c: 0 for c in classes }
    c2 = { c: torch.sum((y==c).float()).item() for c in classes }
    
    for vidx in range(x.shape[0]-1):
        c1[y[vidx].item()] += 1
        c2[y[vidx].item()] -= 1

        if x[vidx, attribute] != x[vidx+1, attribute]:
            g1 = (vidx + 1) * calcGiniIndex(c1, vidx + 1, classes)
            g2 = (x.shape[0] - (vidx + 1)) * calcGiniIndex(c2, x.shape[0] - (vidx + 1), classes)
            gini = g1 + g2

            if gini < purity:
                value = ((x[vidx, attribute] + x[vidx+1, attribute]) / 2).item()
                purity = gini

    return value, purity


def splitAttributeValue(x: torch.Tensor, y: torch.Tensor, classes: List[float]) -> (int, float, float):
    """ Given a dataset, return best (attribute, value) pair to split on

    Args:
        x: input dataset
        y: input dataset labels
        classes: list of unique possible labels

    Returns:
        attribute to split on, attribute value to split on, and gini index at
        that split
    """

    attribute, value, purity = None, None, float("inf")
    for aidx in range(x.shape[1]):

        xargs = torch.argsort(x[:, aidx])
        x, y = x[xargs], y[xargs]
        val, gini = splitValue(x, y, aidx, classes)

        if gini < purity:
            attribute, value, purity = aidx, val, gini

    return attribute, value, purity


def buildTree(x: torch.Tensor, y: torch.Tensor, classes: List[float], depth: int=0, maxDepth: int=None, leafSize: int=1) -> Node:
    """ Build decision tree given a dataset

    Args:
        x: input dataset
        y: input dataset labels
        classes: list of possible unique labels
        depth: depth of current node
        maxDepth: maximum allowed depth of tree
        leafSize: minimum required members of node for split

    Returns:
        root node of tree
    """

    if (x.shape[0] <= leafSize) or (maxDepth is not None and depth==maxDepth):
        counts = [(c, torch.sum((y==c).float()).item()) for c in classes]
        label = sorted(counts, key=lambda x: x[1])[-1][0]
        return Node(label=label)

    else:
        
        attribute, value, purity = splitAttributeValue(x, y, classes)

        if value is None:
            counts = [(c, torch.sum((y==c).float()).item()) for c in classes]
            label = sorted(counts, key=lambda x: x[1])[-1][1]
            return Node(label=label)

        else:
            largs = (x[:, attribute] <= value)
            rargs = (x[:, attribute] > value)
            left = buildTree(x[largs], y[largs], classes, depth+1, maxDepth, leafSize)
            right = buildTree(x[rargs], y[rargs], classes, depth+1, maxDepth, leafSize)
            return Node(left, right, attribute, value)


