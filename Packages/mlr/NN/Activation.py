import torch


def Relu(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):

    output, grad = torch.clone(x), torch.clone(x)
    output[output<0] = 0
    grad[grad>0] = 1
    grad[grad<0] = 0
    return output, grad


def Sigmoid(x: torch.Tensor) -> (torch.Tensor, torch.Tensor): 

    output = 1 / (1 + torch.exp(-x))
    grad = (1 - output) * (output)
    return output, grad


