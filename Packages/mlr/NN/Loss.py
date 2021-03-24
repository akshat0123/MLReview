import torch


def CrossEntropy(y: torch.Tensor, yhat: torch.tensor) -> (torch.Tensor, torch.Tensor):
    
    output = - ((y * torch.log(yhat)) + ((1-y) * torch.log(1-yhat)))
    grad = ((1 - y) / (1 - yhat)) - (y / yhat)
    return output, grad


