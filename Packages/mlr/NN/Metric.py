import torch


def ErrorRate(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    """ Calculate error rate given true and predicted values

    Args:
        y: true values
        yhat: predicted values

    Returns:
        error rate
    """

    if len(y.shape) > 1 and y.shape[1] > 1: err = OneHotErrorRate(y, yhat)
    else: err = torch.sum((y != yhat).float()) / y.shape[0]
    return err


def OneHotErrorRate(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    """ Calculate error rate given true and predicted values for one-hot encoded
        vectors

    Args:
        y: true values
        yhat: predicted values

    Returns:
        error rate
    """
    
    return ErrorRate(torch.argmax(y, dim=1), torch.argmax(yhat, dim=1))


def Accuracy(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    """ Calculate accuracy given true and predicted values

    Args:
        y: true values
        yhat: predicted values

    Returns:
        accuracy 
    """

    return 1 - ErrorRate(y, yhat)


def MeanSquaredError(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    """ Calculate mean squared error given true and predicted values

    Args:
        y: true values
        yhat: predicted values

    Returns:
        mean squared error 
    """
    
    return torch.mean((y - yhat)**2)
