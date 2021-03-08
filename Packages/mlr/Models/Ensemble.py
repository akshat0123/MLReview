from typing import List

from mlr.Models.DecisionTree import DecisionTreeClassifier

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


