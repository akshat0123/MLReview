from abc import ABC, abstractmethod

from mlr.NN.Regularizer import *
from mlr.NN.Initializer import *
from mlr.NN.Activation import *
import torch


# Applicable activation functions
ACTIVATIONS = {
    'linear' : Linear,
    'relu'   : Relu,
    'sigmoid': Sigmoid,
    'softmax': Softmax
}


# Applicable initializers
INITIALIZERS = {
    'glorot': GlorotInitializer,
    'he': HeInitializer,
    'random': RandomInitializer
}

# Applicable regularizers
REGULARIZERS = {
    'l': LRegularizer,
    'l1': L1Regularizer,
    'l2': L2Regularizer
}


class Layer(ABC):
    """ Abstract base class for neural network layers
    """

    @abstractmethod
    def __init__(self):
        """ Initialize layer
        """
        pass
    

    @abstractmethod
    def forward(self):
        """ Run forward pass
        """
        pass


    @abstractmethod
    def backward(self):
        """ Run backward propagation
        """
        pass


def Dense(inputdim: int, units: int, activation: str, initializer: str=None, regularizer: str=None, dropout: float=None) -> Layer:
    """ Returns appropriate initialized layer architecture provided activation

    Args:
        inputdim: number of input units
        units: number of units in layer
        activation: activation function string => should be a key of ACTIVATIONS
        initializer: weight initialization scheme => should be a key of INITIALIZERS
        regularizer: regularization method => should be a key of REGULARIZERS
        dropout: probability that a hidden unit should be dropped out

    Returns:
        Initialized neural network layer
    """

    if activation == 'softmax':                
        return SoftmaxDenseLayer(inputdim=inputdim, units=units,
        activation='softmax', initializer=initializer, regularizer=regularizer, dropout=dropout)

    else: 
        return DefaultDenseLayer(inputdim=inputdim, units=units,
        activation=activation, initializer=initializer, regularizer=regularizer, dropout=dropout)


class DefaultDenseLayer(Layer): 
    """ Default dense layer class
    """


    def __init__(self, inputdim: int, units: int, activation: str, initializer: str=None, regularizer: str=None, dropout: float=None) -> None:
        """ Initialize default dense layer

        Args:
            inputdim: number of input units
            units: number of units in layer
            activation: activation function string => should be a key of ACTIVATIONS
            initializer: weight initialization scheme => should be a key of INITIALIZERS
            regularizer: regularization method => should be a key of REGULARIZERS
            dropout: probability that a hidden unit should be dropped out
        """

        self.w = INITIALIZERS[initializer](inputdim, units) if initializer else INITIALIZERS['random'](inputdim, units)
        self.regularizer = regularizer if regularizer else 'l'
        self.activation = activation
        self.dropout = dropout
        self.dz_dw = None
        self.dz_dx = None
        self.da_dz = None
        self.dr_dw = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Run forward pass through layer, saving local gradients

        Args:
            x: input data

        Returns:
            output of layer given input x
        """

        z, self.dz_dw, self.dz_dx = torch.einsum('ij,jk->ik', x, self.w), x, self.w
        a, self.da_dz = ACTIVATIONS[self.activation](z)

        if self.dropout:
            mask = torch.rand(a.shape)
            mask[mask <= self.dropout] = 0
            mask[mask > self.dropout] = 1
            a = (a * mask) / (1-self.dropout)
            self.da_dz = (self.da_dz * mask) / (1-self.dropout)

        r, self.dr_dw = REGULARIZERS[self.regularizer](self.w)

        return a, r


    def backward(self, dl: torch.Tensor, alpha: float, lambdaa: float=1.0) -> torch.Tensor:
        """ Run backward pass through layer, updating weights and returning
            cumulative gradient from last connected layer (output layer)
            backwards through to this layer

        Args:
            dl: cumulative gradient calculated from layers ahead of this layer
            alpha: learning rate
            lambdaa: regularization rate

        Returns:
            cumulative gradient calculated at this layer
        """

        dl_dz = self.da_dz * dl
        dl_dw = torch.einsum('ij,ik->jk', self.dz_dw, dl_dz) / dl.shape[0] 
        dl_dx = torch.einsum('ij,kj->ki', self.dz_dx, dl_dz)
        self.w -= alpha * (dl_dw + lambdaa * self.dr_dw)
        return dl_dx


class SoftmaxDenseLayer(DefaultDenseLayer):
    """ Dense layer class for multinomial classification using the Softmax
        activation function
    """


    def backward(self, dl: torch.Tensor, alpha: float, lambdaa: float=1.0) -> torch.Tensor:
        """ Run backward pass through layer, updating weights and returning
            cumulative gradient from last connected layer (output layer)
            backwards through to this layer

        Args:
            dl: cumulative gradient calculated from layers ahead of this layer
            alpha: learning rate
            lambdaa: regularization rate

        Returns:
            cumulative gradient calculated at this layer
        """

        dl_dz = torch.einsum('ijk,ik->ij', self.da_dz, dl)
        dl_dw = torch.einsum('ij,ik->jk', self.dz_dw, dl_dz) / dl.shape[0] 
        dl_dx = torch.einsum('ij,kj->ki', self.dz_dx, dl_dz)
        self.w -= alpha * (dl_dw + lambdaa * self.dr_dw)

        return dl_dx


