from typing import List

from mlr.Preprocessing.Preprocessing import createOneHotColumn
from mlr.Models.Activation import Softmax
from mlr.Models.DecisionTree import * 

from tqdm import trange, tqdm
import torch


class RandomForestClassifier:


    def __init__(self, numTrees: int=5, maxDepth: int=None, leafSize: int=1, bootstrapRatio: float=0.3) -> None:
        """ Instantiate random forest classifier

        Args:
            numTrees: number of trees to build
            maxDepth: the maximum allowed depth of each tree
            leafSize: the minimum number of data points required to split a node 
            bootstrapRatio: ratio of training data to use for each bootstrap
        """
        self.forest = []
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.leafSize = leafSize
        self.bootstrapRatio = bootstrapRatio


    def fit(self, x: torch.Tensor, y: torch.Tensor, classes: List[float]) -> None:
        """ Fit random forest model to dataset

        Args:
            x: input data
            y: input labels
            classes: list of unique possible labels
        """

        for i in trange(self.numTrees):
            
            # Create bootstrap sample
            rargs = torch.randperm(x.shape[0]) 
            x, y = x[rargs], y[rargs] 
            bidx = int(x.shape[0] * self.bootstrapRatio)

            tree = DecisionTreeClassifier(maxDepth=self.maxDepth, leafSize=self.leafSize)
            tree.fit(x[:bidx], y[:bidx], classes)
            self.forest.append(tree)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Predict outcomes given input

        Args:
            x: input data

        Returns:
            tensor of class labels            
        """

        return torch.mode(
            torch.cat(
                [tree.predict(x) for tree in self.forest], dim=1
            ), dim=1
        ).values[:, None]


class GradientBoostedTreeClassifier:


    def __init__(self, maxDepth: int=3, leafSize: int=1):
        """ Initializes GradientBoostedTreeClassifier class

        Args:
            maxDepth: maximum depth of internal trees
            leafSize: minimum size of node for each tree
        """
        self.maxDepth = maxDepth
        self.leafSize = leafSize
        self.layers = []


    def fit(self, x: torch.Tensor, y: torch.Tensor, alpha: float=0.01) -> None:
        """ Fit classifier to input data

        Args:
            x: input data
            y: input labels
            alpha: alpha parameter for weight update
        """

        y = torch.Tensor(createOneHotColumn(y.numpy())[0])
        self.numClasses = y.shape[1]
        ypred = torch.full(y.shape, 1 / y.shape[1])

        layers, fx = [], y - ypred
        for i in trange(10):

            trees = [ DecisionTreeRegressor(maxDepth=self.maxDepth, leafSize=self.leafSize) for i in range(y.shape[1]) ]
            for i in range(len(trees)): 
                trees[i].fit(x, fx[:, i])

            layers.append(trees)

            probs = self.probs(x, trees)
            fx -= alpha * (probs - y)

        self.layers = layers


    def probs(self, x: torch.Tensor, trees: List[DecisionTreeRegressor]) -> torch.Tensor:
        """ Determine probability of belonging to each class

        Args:
            x: input data
            trees: list of decision trees, one for each class

        Returns:
            probability for each input
        """

        probs = [trees[i].predict(x)[:, None] for i in range(len(trees))]
        return Softmax(torch.cat(probs, dim=1))


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Predict labels

        Args:
            x: input data

        Returns:
            labels for each member of input
        """

        ypred = torch.zeros((x.shape[0], self.numClasses))
        for trees in self.layers:
            probs = self.probs(x, trees)
            ypred += probs

        return torch.argmax(ypred, dim=1)[:, None]


