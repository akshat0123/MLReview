import torch


def MeanSquaredError(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """ Calculate mean squared error rate

    Args:
        y: true labels
        yhat: predicted labels

    Returns:
        mean squared error        
    """

    return torch.sum((y - yhat)**2) / y.shape[0]


def ErrorRate(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """ Calculate error rate (1 - accuracy)

    Args:
        y: true labels
        yhat: predicted labels

    Returns:
        error rate
    """

    return torch.sum((y != yhat).float()) / y.shape[0]


def OneHotErrorRate(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    """ Calculate error rate for one-hot encoded multiclass problem

    Args:
        y: true labels
        yhat: predicted labels

    Returns:
        error rate
    """

    return ErrorRate(torch.argmax(y, dim=1), torch.argmax(yhat, dim=1))
