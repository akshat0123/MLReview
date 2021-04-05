import torch


def Linear(x: torch.Tensor) -> (torch.Tensor, int):
    """ Placeholder activation function for linear layers

    Args:
        x: input tensor

    Returns:
        activated output tensor
        local gradient 
    """

    return x, 1


def Relu(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """ Relu activation function

    Args:
        x: input tensor

    Returns:
        activated output tensor
        local gradient 
    """

    output, grad = torch.clone(x), torch.clone(x)                
    output[output < 0] = 0 
    grad[grad > 0] = 1
    grad[grad < 0] = 0 
    return output, grad


def Sigmoid(x: torch.Tensor) -> (torch.Tensor, torch.Tensor): 
    """ Sigmoid activation function

    Args:
        x: input tensor

    Returns:
        activated output tensor
        local gradient 
    """

    output = (1 / (1 + torch.exp(-x)))
    grad = ((1 - output) * (output))
    return output, grad


def Softmax(x: torch.Tensor) -> (torch.tensor, torch.Tensor):
    """ Softmax activation function

    Args:
        x: input tensor

    Returns:
        activated output tensor
        local gradient 
    """

    output = torch.exp(x) / torch.sum(torch.exp(x), dim=1)[:, None]
    diags = torch.stack([torch.diag(output[i]) for i in range(output.shape[0])])
    grad = diags - torch.einsum('ij,ik->ijk', output, output)
    return output, grad
