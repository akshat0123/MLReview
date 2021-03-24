from typing import Callable, List

from mlr.NN.Layer import Layer
from mlr.NN.Loss import *
from tqdm import trange
import torch


losses = {
    'cross_entropy': CrossEntropy
}


class Model:


    def __init__(self, layers: List[Layer]):
        self.layers = layers


    def forward(self, x: torch.Tensor):
        yhat = x
        for layer in self.layers:
            yhat = layer.forward(yhat)

        return yhat


    def backward(self, dl_da: torch.Tensor, alpha: float):

        dl = self.layers[-1].backward(dl_da, alpha)
        for layer in self.layers[1:][::-1]:
            dl = layer.backward(dl, alpha)


    def fit(self, x: torch.Tensor, y: torch.Tensor, alpha: float, epochs: int, batch: int, loss: str):
        
        epochs = trange(epochs, desc='Loss')
        for epoch in epochs:

            start, end = 0, batch
            for b in range((x.shape[0]//batch)+1):

                if x[start:end].shape[0] > 0:
                    yhat = self.forward(x[start:end])
                    l, dl_da = losses[loss](y[start:end], yhat)
                    dl = self.backward(dl_da, alpha)
                    start += batch
                    end += batch

            epochs.set_description('Loss: %.4f' % torch.mean(l).item())


    def predict(self, x: torch.Tensor):

        yhat = self.forward(x) 
        yhat[yhat >= 0.5] = 1
        yhat[yhat < 0.5] = 0
        return yhat


