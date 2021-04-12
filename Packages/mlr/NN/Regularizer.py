import torch


def LRegularizer(w: torch.tensor) -> torch.Tensor:
    """

    Args:

    Returns:
    """

    return 0, torch.Tensor([0])        


def L1Regularizer(w: torch.Tensor) -> torch.Tensor:
    """

    Args:

    Returns:
    """

    penalty = torch.sum(torch.abs(torch.clone(w)))
    grad = torch.clone(w)
    grad[grad >= 0] = 1
    grad[grad < 0] = -1

    return penalty, grad


def L2Regularizer(w: torch.Tensor) -> torch.Tensor:
    """

    Args:

    Returns:
    """

    penalty = torch.sum(torch.square(torch.clone(w)))
    grad = 2 * torch.clone(w)

    return penalty, grad
