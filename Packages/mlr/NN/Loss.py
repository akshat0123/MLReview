import torch


def BinaryCrossEntropy(y: torch.Tensor, yhat: torch.tensor) -> (torch.Tensor, torch.Tensor):
    
    output = - ((y * torch.log(yhat)) + ((1-y) * torch.log(1-yhat)))
    grad = ((1 - y) / (1 - yhat)) - (y / yhat)
    return output, grad


def CategoricalCrossEntropy(y: torch.Tensor, yhat: torch.Tensor):
    loss = -1 * torch.sum(y * torch.log(yhat), dim=1)
    grad = -1 * (y / yhat)
    return loss, grad            

