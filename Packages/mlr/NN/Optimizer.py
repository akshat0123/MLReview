from abc import ABC, abstractmethod

import torch


class Optimizer(ABC):
    """ Abstract base class for optimizers
    """

    @abstractmethod
    def __init__(self):
        """ Initialize optimizer
        """
        pass            


    @abstractmethod
    def update(self):
        """ Update weights
        """
        pass


    @abstractmethod
    def __copy__(self):
        """ Copy class instance
        """
        pass


def SGDOptimizer(momentum: bool=False, epsilon: float=1e-4) -> Optimizer:
    """ Return stochastic gradient descent optimizer

    Args:
        momentum: whether to include momentum or not
        epsilon: epsilon parameter for momentum

    Returns:
        stochastic gradient descent optimizer
    """

    optimizer = SGDMomentumOptimizer(epsilon=epsilon) if momentum else DefaultSGDOptimizer()
    return optimizer


class DefaultSGDOptimizer(Optimizer):
    """ Stochastic Gradient Descent optimizer (without momentum)
    """
    

    def __init__(self) -> None:
        """ Initialize default SGD optimizer
        """
        pass


    def __copy__(self) -> DefaultSGDOptimizer:
        """ Return copy of default SGD optimizer

        Returns: 
            copy of optimizer
        """

        return DefaultSGDOptimizer()


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """
        return w - (alpha * (dw + (lambdaa * dr)))


class SGDMomentumOptimizer(Optimizer):
    """ Stochastic Gradient Descent optimizer (with momentum)
    """
        

    def __init__(self, epsilon: float=1e-4) -> None:
        """ Initialize default SGD optimizer
        """
        self.epsilon = epsilon
        self.v = None


    def __copy__(self) -> SGDMomentumOptimizer:
        """ Return copy of default SGD optimizer

        Returns: 
            copy of optimizer
        """
        return SGDMomentumOptimizer(epsilon=self.epsilon)            


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        if self.v is None: 
            self.v = torch.zeros(w.shape)

        self.v = (self.epsilon * self.v) - (alpha * (dw + (lambdaa * dr)))
        return w + self.v


class AdaGradOptimizer(Optimizer):
    """ AdaGrad optimizer (with momentum)
    """


    def __init__(self) -> None:
        """ Initialize AdaGrad optimizer
        """
        self.delta = 1e-5 
        self.r = None


    def __copy__(self) -> AdaGradOptimizer:
        """ Return copy of default SGD optimizer

        Returns: 
            copy of optimizer
        """
        return AdaGradOptimizer()


    def update(self, w: torch.Tensor, alpha: float, dw: torch.Tensor, dr: torch.Tensor, lambdaa: float=1.0) -> torch.Tensor:
        """ Update weights

        Args:
            w: weight tensor
            alpha: learning rate 
            dw: weight gradient
            dr: regularization gradient
            lambdaa: regularization lambda parameter

        Returns: 
            updated weight tensor
        """

        if self.r is None: 
            self.r = torch.zeros(w.shape)

        self.r = self.r + (dw)**2
        return w - (alpha * ((dw + (lambdaa * dr)) / (self.delta + torch.sqrt(self.r))))
