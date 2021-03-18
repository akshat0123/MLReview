import torch


def Softmax(x: torch.Tensor) -> torch.Tensor:
    """ Apply softmax function to tensor

    Args:
        x: input tensor

    Returns:
        tensor with softmax function applied to all members
    """

    return torch.exp(x) / torch.sum(torch.exp(x), dim=1)[:, None]


def BinaryStep(x: torch.Tensor) -> torch.Tensor:
    """ Apply binary step function to tensor

    Args:
        x: input tensor

    Returns:
        tensor with binary step function applied to all members
    """        

    x[x >= 0] = 1
    x[x < 0] = 0
    return x
