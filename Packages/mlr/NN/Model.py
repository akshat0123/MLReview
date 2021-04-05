from abc import ABC, abstractmethod
from typing import List

from mlr.NN.Metric import Accuracy
from mlr.NN.Layer import Layer
from mlr.NN.Loss import *
from tqdm import trange
import torch


losses = {
    'binary_cross_entropy': BinaryCrossEntropy,
    'categorical_cross_entropy': CategoricalCrossEntropy,
    'mean_squared_error': MeanSquaredError
}


class Network(ABC):
        
    @abstractmethod
    def __init__(self):
        pass

    
    @abstractmethod
    def forward(self):
        pass


    @abstractmethod
    def backward(self):
        pass
    

    @abstractmethod
    def fit(self):
        pass


    @abstractmethod
    def predict(self):
        pass


def Model(layers: List[Layer], loss=str) -> Network :

    if loss == 'binary_cross_entropy':
        return BinaryClassifier(layers, loss='binary_cross_entropy')

    elif loss == 'categorical_cross_entropy':
        return DefaultClassifier(layers, loss='categorical_cross_entropy')

    elif loss == 'mean_squared_error':
        return DefaultRegressor(layers, loss='mean_squared_error')


class DefaultClassifier(Network):

    def __init__(self, layers: List[Layer], loss=str) -> None:
        self.layers = layers
        self.loss = loss

    
    def forward(self, x: torch.Tensor):

        ypred = x
        for layer in self.layers:
            ypred = layer.forward(ypred)                        

        return ypred


    def backward(self, dl: torch.Tensor, alpha):

        for layer in self.layers[::-1]:
            dl = layer.backward(dl, alpha)

    
    def fit(self, x: torch.Tensor, y: torch.Tensor, batch: int, alpha: float, epochs: int):

        epochs = trange(epochs)
        for epoch in epochs:

            l, start, end = [], 0, batch
            for b in range((x.shape[0]//2) + 1):

                xbatch, ybatch = x[start:end], y[start:end]
                if xbatch.shape[0] > 0:

                    ypred = self.forward(xbatch)
                    bl, dl = losses[self.loss](ybatch, ypred)
                    dl = self.backward(dl, alpha)
                    l.append(bl.item())

                start += batch 
                end += batch

            ypred = self.predict(x)
            acc = Accuracy(y, ypred)
            epochs.set_description('Loss: %.8f | Acc: %.8f' % (sum(l) / len(l), acc))


    def predict(self, x: torch.Tensor):
        return self.forward(x)


class BinaryClassifier(DefaultClassifier):

    def predict(self, x: torch.Tensor):
        ypred = self.forward(x)
        if self.loss == 'binary_cross_entropy':
            ypred[ypred >= 0.5] = 1
            ypred[ypred < 0.5] = 0

        return ypred


class DefaultRegressor(DefaultClassifier):
        

    def fit(self, x: torch.Tensor, y: torch.Tensor, batch: int, alpha: float, epochs: int):

        epochs = trange(epochs)
        for epoch in epochs:

            l, start, end = [], 0, batch
            for b in range((x.shape[0]//2) + 1):

                xbatch, ybatch = x[start:end], y[start:end]
                if xbatch.shape[0] > 0:

                    ypred = self.forward(xbatch)
                    bl, dl = losses[self.loss](ybatch, ypred)
                    dl = self.backward(dl, alpha)
                    l.append(bl.item())

                start += batch 
                end += batch

            ypred = self.predict(x)
            mse, _ = MeanSquaredError(y, ypred)
            epochs.set_description('Loss: %.8f' % mse)
