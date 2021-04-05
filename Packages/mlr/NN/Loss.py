import torch


def BinaryCrossEntropy(y: torch.Tensor, yhat: torch.tensor) -> (torch.Tensor, torch.Tensor):
    """ Calculate binary cross entropy given true values and predictions, and
        returns loss as well as local gradient

    Args:
        y: true values
        yhat: predicted values

    Returns:
        binary cross entorpy loss
        local gradient
    """
    
    output = torch.mean(- ((y * torch.log(yhat)) + ((1-y) * torch.log(1-yhat))))
    grad = ((1 - y) / (1 - yhat)) - (y / yhat)
    return output, grad


def CategoricalCrossEntropy(y: torch.Tensor, yhat: torch.Tensor):
    """ Calculate categorical cross entropy given true values and predictions,
        and returns loss as well as local gradient

    Args:
        y: true values
        yhat: predicted values

    Returns:
        categorical cross entorpy loss
        local gradient
    """

    loss = torch.mean(-1 * torch.sum(y * torch.log(yhat), dim=1))
    grad = -1 * (y / yhat)
    return loss, grad            


def MeanSquaredError(y: torch.Tensor, yhat: torch.Tensor):
    """ Calculate mean squared error given true values and predictions, and
        returns loss as well as local gradient

    Args:
        y: true values
        yhat: predicted values

    Returns:
        mean squared error loss
        local gradient
    """

    loss = torch.mean((y - yhat)**2)
    grad = yhat - y
    return loss, grad


