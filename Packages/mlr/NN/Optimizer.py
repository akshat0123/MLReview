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
    def __copy__(self):
        """ Copy class instance
        """
        pass


    @abstractmethod
    def update(self):
        """ Update weights
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


    def __copy__(self):
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


    def __copy__(self):
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
    """ AdaGrad optimizer
    """


    def __init__(self) -> None:
        """ Initialize AdaGrad optimizer
        """

        self.delta = 1e-5 
        self.r = None


    def __copy__(self):
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


def RMSPropOptimizer(momentum: bool=False, rho: float=0.9, epsilon: float=1e-4) -> Optimizer:
    """ Return RMSProp optimizer

    Args:
        momentum: whether to include momentum or not
        epsilon: epsilon parameter for momentum
        rho: rho parameter for RMSProp

    Returns:
        RMSProp optimizer
    """

    optimizer = RMSPropMomentumOptimizer(rho=rho, epsilon=epsilon) if momentum else DefaultRMSPropOptimizer(rho=rho)
    return optimizer


class DefaultRMSPropOptimizer:
    """ RMSProp optimizer (without momentum)
    """


    def __init__(self, rho: float=0.9) -> None:
        """ Initialize optimizer
        """

        self.delta = 1e-5
        self.rho = rho
        self.r = None


    def __copy__(self):
        """ Return copy of DefaultRMSPropOptimizer
        """

        return DefaultRMSPropOptimizer(self.rho)


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

        self.r = (self.rho * self.r) + ((1 - self.rho) * (dw**2))
        return w - (alpha * ((dw + (lambdaa * dr)) / (torch.sqrt(self.delta + self.r))))


class RMSPropMomentumOptimizer:
    """ RMSProp optimizer (without momentum)
    """


    def __init__(self, rho: float=0.9, epsilon: float=1e-4):
        """ Initialize optimizer
        """

        self.epsilon = epsilon
        self.delta = 1e-05
        self.rho = rho
        self.r = None
        self.v = None


    def __copy__(self):
        """ Return copy of RMSPropMomentumOptimizer
        """

        return RMSPropMomentumOptimizer(self.rho, self.epsilon)


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
            self.v = torch.zeros(w.shape)

        r = (self.rho * self.r) + ((1 - self.rho) * (dw**2))
        v = (self.epsilon * self.v) - (alpha * ((dw + (lambdaa * dr)) / (torch.sqrt(self.delta + self.r))))

        return w + v


class AdamOptimizer(ABC):
    """ Adam optimizer
    """

    def __init__(self, rho1: float=0.9, rho2: float=0.999):
        """ Initialize optimizer
        """

        self.delta = 1e-5
        self.rho1 = rho1
        self.rho2 = rho2
        self.s = None
        self.r = None


    def __copy__(self):
        """ Return copy of AdamOptimizer
        """

        return AdamOptimizer(self.rho1, self.rho2)


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

        if self.s is None:
            self.s = torch.zeros(w.shape)                
            self.r = torch.zeros(w.shape)                

        self.s = (self.rho1 * self.s) + ((1 - self.rho1) * (dw + (lambdaa * dr)))
        self.r = (self.rho2 * self.r) + ((1 - self.rho2) * (dw + (lambdaa * dr))**2)
        shat = self.s / (1 - self.rho1)
        rhat = self.r / (1 - self.rho2)

        return w - (alpha * (shat / (torch.sqrt(rhat) + self.delta)))
