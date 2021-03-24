from abc import ABC, abstractmethod
from typing import Callable

from mlr.NN.Activation import *
import torch


activations = {
    'relu': Relu,
    'sigmoid': Sigmoid
}


class Layer(ABC):

    @abstractmethod
    def __init__(self):
        pass
    

    @abstractmethod
    def forward(self):
        pass


    @abstractmethod
    def backward(self):
        pass


class Dense(Layer): 


    def __init__(self, inputdim: int, units: int, activation: str) -> None:

        self.w = torch.rand((inputdim, units)) / 10
        self.activation = activations[activation]
        self.dz_dw = None
        self.da_dz = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, self.dz_dw = torch.einsum('ij,jk->ik', x, self.w), x
        a, self.da_dz = self.activation(z)
        return a


    def backward(self, dl: torch.Tensor, alpha: float) -> torch.Tensor:

        out = torch.einsum('ij,kl->ki', self.w, self.da_dz * dl)
        dl_dw = torch.mean(torch.einsum('ij,ik->ji', self.dz_dw, self.da_dz * dl), dim=1)[:, None]
        self.w = self.w - (alpha * dl_dw)
        return out


