from abc import ABC, abstractmethod

from mlr.NN.Activation import *
import torch


activations = {
    'linear': Linear,
    'relu': Relu,
    'sigmoid': Sigmoid,
    'softmax': Softmax
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


def Dense(inputdim: int, units: int, activation: str) -> Layer:

    if activation == 'softmax':                
        return SoftmaxDenseLayer(inputdim=inputdim, units=units, activation='softmax')

    else: 
        return DefaultDenseLayer(inputdim=inputdim, units=units, activation=activation)            


class DefaultDenseLayer(Layer): 


    def __init__(self, inputdim: int, units: int, activation: str) -> None:
        self.w = (torch.rand((inputdim, units)) * 2 - 1)
        self.activation = activation
        self.dz_dw = None
        self.dz_dx = None
        self.da_dz = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, self.dz_dw, self.dz_dx = torch.einsum('ij,jk->ik', x, self.w), x, self.w
        a, self.da_dz = activations[self.activation](z)
        return a


    def backward(self, dl: torch.Tensor, alpha: float) -> torch.Tensor:
        dl_dz = self.da_dz * dl
        dl_dw = torch.einsum('ij,ik->jk', self.dz_dw, dl_dz) / dl.shape[0] 
        dl_dx = torch.einsum('ij,kj->ki', self.dz_dx, dl_dz)
        self.w -= alpha * dl_dw
        return dl_dx


class SoftmaxDenseLayer(DefaultDenseLayer):


    def backward(self, dl: torch.Tensor, alpha: float) -> torch.Tensor:
        dl_dz = torch.einsum('ijk,ik->ij', self.da_dz, dl)
        dl_dw = torch.einsum('ij,ik->jk', self.dz_dw, dl_dz) / dl.shape[0] 
        dl_dx = torch.einsum('ij,kj->ki', self.dz_dx, dl_dz)
        self.w -= alpha * dl_dw

        return dl_dx


