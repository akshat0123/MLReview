from abc import ABC, abstractmethod
from typing import List
import copy

from mlr.NN.Metric import Accuracy
from mlr.NN.Layer import Layer
from mlr.NN.Optimizer import *
from mlr.NN.Loss import *
from tqdm import trange
import torch


# Applicable loss functions
LOSSES = {
    'binary_cross_entropy':      BinaryCrossEntropy,
    'categorical_cross_entropy': CategoricalCrossEntropy,
    'mean_squared_error':        MeanSquaredError
}


class Network(ABC):
    """ Abstract base class for neural network models 
    """
        
    @abstractmethod
    def __init__(self):
        """ Initialize model
        """
        pass

    
    @abstractmethod
    def forward(self):
        """ Run forward pass through network
        """
        pass


    @abstractmethod
    def backward(self):
        """ Run backpropagation through network
        """
        pass
    

    @abstractmethod
    def fit(self):
        """ Fit network to data
        """
        pass


    @abstractmethod
    def predict(self):
        """ Return predictions given input
        """
        pass


def Model(layers: List[Layer], loss: str, optimizer: Optimizer) -> Network:
    """ Return initialized neural network model provided loss type

    Args:
        layers: list of initialized neural network Layer objects
        loss: string describing loss type => should be a key of LOSSES
        optimizer: Optimizer for model

    Returns:
        Initialized neural nework model object
    """

    if loss == 'binary_cross_entropy':
        return BinaryClassifier(layers, loss='binary_cross_entropy', optimizer=optimizer)

    elif loss == 'categorical_cross_entropy':
        return DefaultClassifier(layers, loss='categorical_cross_entropy', optimizer=optimizer)

    elif loss == 'mean_squared_error':
        return DefaultRegressor(layers, loss='mean_squared_error', optimizer=optimizer)


class DefaultClassifier(Network):
    """ Default classifier class (one-hot encoded multinomial output)
    """

    def __init__(self, layers: List[Layer], loss: str, optimizer: str) -> None:
        """ Initialize model

        Args:
            layers: list of initialized neural network Layer objects
            loss: string describing loss type => should be a key of LOSSES
            optimizer: Optimizer for model
        """
        for layer in layers: layer.setOptimizer(copy.copy(optimizer))
        self.layers = layers
        self.loss = loss

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Run forward pass through network

        Args:
            x: input data

        Returns:
            output of final layer in network
        """

        ypred, totalpenalty = x, 0
        for layer in self.layers:
            ypred, penalty = layer.forward(ypred)                        
            totalpenalty += penalty

        return ypred, totalpenalty


    def backward(self, dl: torch.Tensor, alpha: float, lambdaa: float) -> None:
        """ Run backpropagation through network

        Args:
            dl: cumlulative gradient at loss function
            alpha: learning rate
        """

        for layer in self.layers[::-1]:
            dl = layer.backward(dl, alpha, lambdaa)

    
    def fit(self, x: torch.Tensor, y: torch.Tensor, batch: int, alpha: float, epochs: int, lambdaa: float=1.0) -> None:
        """ Fit network to data
        
        Args:
            x: input data
            y: input labels
            batch: batch size for training
            alpha: learning rate
            epochs: number of iterations over entire dataset to train
            lambdaa: regularization rate
        """

        epochs = trange(epochs)
        for epoch in epochs:

            l, start, end = [], 0, batch
            for b in range((x.shape[0]//2) + 1):

                xbatch, ybatch = x[start:end], y[start:end]
                if xbatch.shape[0] > 0:


                    ypred, penalty = self.forward(xbatch)     # Forward pass
                    bl, dl = LOSSES[self.loss](ybatch, ypred) # Calculate loss
                    bl += lambdaa * penalty                   # Add regularization term
                    dl = self.backward(dl, alpha, lambdaa)    # Backpropagation
                    l.append(bl.item())

                start += batch; end += batch

            ypred = self.predict(x)
            acc = Accuracy(y, ypred)
            epochs.set_description('Loss: %.8f | Acc: %.8f' % (sum(l) / len(l), acc))


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Return predictions given input

        Args:
            x: input data

        Returns:
            predictions               
        """
         
        return self.forward(x)[0]


class BinaryClassifier(DefaultClassifier):
    """ Classifier class for binomial output only
    """

    def predict(self, x: torch.Tensor):
        """ Return predictions given input

        Args:
            x: input data

        Returns:
            predictions               
        """

        ypred = self.forward(x)[0]
        if self.loss == 'binary_cross_entropy':
            ypred[ypred >= 0.5] = 1
            ypred[ypred < 0.5] = 0

        return ypred


class DefaultRegressor(DefaultClassifier):
    """ Default regressor class
    """

    def fit(self, x: torch.Tensor, y: torch.Tensor, batch: int, alpha: float, epochs: int, lambdaa: float=1.0):
        """ Fit network to data
        
        Args:
            x: input data
            y: input labels
            batch: batch size for training
            alpha: learning rate
            epochs: number of iterations over entire dataset to train
        """

        epochs = trange(epochs)
        for epoch in epochs:

            l, start, end = [], 0, batch
            for b in range((x.shape[0]//2) + 1):

                xbatch, ybatch = x[start:end], y[start:end]
                if xbatch.shape[0] > 0:

                    ypred, penalty = self.forward(xbatch)     # Forward pass
                    bl, dl = LOSSES[self.loss](ybatch, ypred) # Calculate loss
                    bl += lambdaa * penalty                   # Add regularization term
                    dl = self.backward(dl, alpha, lambdaa)    # Backpropagation
                    l.append(bl.item())

                start += batch 
                end += batch

            ypred = self.predict(x)
            mse, _ = MeanSquaredError(y, ypred)
            epochs.set_description('Loss: %.8f' % mse)
