import torch


def Linear(x: torch.Tensor):
    return x, 1


def Relu(x: torch.Tensor):
    z, grad = torch.clone(x), torch.clone(x)                
    grad[grad > 0] = 1
    grad[grad < 0] = 0 
    z[z < 0] = 0 
    return z, grad


def Sigmoid(x: torch.Tensor) -> (torch.Tensor, torch.Tensor): 
    output = (1 / (1 + torch.exp(-x)))
    grad = ((1 - output) * (output))
    return output, grad


def Softmax(x: torch.Tensor):
    output = torch.exp(x) / torch.sum(torch.exp(x), dim=1)[:, None]
    diags = torch.stack([torch.diag(output[i]) for i in range(output.shape[0])])
    grad = diags - torch.einsum('ij,ik->ijk', output, output)
    return output, grad
