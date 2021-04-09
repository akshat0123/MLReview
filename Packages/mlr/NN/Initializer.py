import torch


def RandomInitializer(inputdim: int, units: int) -> torch.Tensor:
    """ Returns randomly initialized weights in range [-1, 1]

    Args:
        inputdim: number of input units         
        units: number of units in layer

    Returns:
        initialized weight tensor            
    """

    return ((torch.rand((inputdim, units)) * 2) - 1) / 10


def GlorotInitializer(inputdim: int, units: int) -> torch.Tensor:
    """ Returns weights initialized using Glorot initialization 

    Args:
        inputdim: number of input units         
        units: number of units in layer

    Returns:
        initialized weight tensor            
    """

    tail = torch.sqrt(torch.Tensor([1/inputdim]))
    weights = (torch.rand((inputdim, units)) * tail * 2) - tail

    return weights


def HeInitializer(inputdim: int, units: int) -> torch.Tensor:
    """ Returns weights initialized using He initialization 

    Args:
        inputdim: number of input units         
        units: number of units in layer

    Returns:
        initialized weight tensor            
    """

    tail = torch.sqrt(torch.Tensor([2/inputdim]))
    weights = (torch.rand((inputdim, units)) * tail * 2) - tail

    return weights
