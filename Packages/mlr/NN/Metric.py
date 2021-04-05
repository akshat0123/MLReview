import torch


def ErrorRate(y: torch.Tensor, yhat: torch.Tensor) -> float:
    if len(y.shape) > 1 and y.shape[1] > 1: err = OneHotErrorRate(y, yhat)
    else: err = torch.sum((y != yhat).float()) / y.shape[0]
    return err


def OneHotErrorRate(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    return ErrorRate(torch.argmax(y, dim=1), torch.argmax(yhat, dim=1))


def Accuracy(y: torch.Tensor, yhat: torch.Tensor):
    return 1 - ErrorRate(y, yhat)


def MeanSquaredError(y: torch.Tensor, yhat: torch.Tensor):
    return torch.mean((y - yhat)**2)
