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


def Softmax(x: torch.Tensor) -> torch.Tensor:
    """ Apply softmax function to tensor

    Args:
        x: input tensor

    Returns:
        tensor with softmax function applied to all members
    """

    return torch.exp(x) / torch.sum(torch.exp(x), dim=1)[:, None]


