from mlr.Preprocessing.Preprocessing import createOneHotColumn
from mlr.Models.Loss import OneHotErrorRate
from abc import abstractmethod, ABC
from tqdm import trange, tqdm
import torch

from utils import loadData


SAVED = './data.pickle'
DATASET = 'Iris'


def Relu(x: torch.Tensor):
        
    z, grad = torch.clone(x), torch.clone(x)                
    grad[grad > 0] = 1
    grad[grad < 0] = 1e-8
    z[z < 0] = 1e-8 

    return z, grad


def Sigmoid(x: torch.Tensor) -> (torch.Tensor, torch.Tensor): 

    output = (1 / (1 + torch.exp(-x)))
    grad = ((1 - output) * (output))
    return output, grad


def Softmax(x: torch.Tensor):

    output = torch.exp(x) / torch.sum(torch.exp(x))
    diags = torch.stack([torch.diag(output[i]) for i in range(output.shape[0])])
    grad = diags - torch.einsum('ij,ik->ijk', output, output)

    return output, grad


activations = {
    'relu': Relu,
    'sigmoid': Sigmoid,
    'softmax': Softmax
}


class Layer(ABC):

    @abstractmethod
    def __init__(self):
        pass
    

    @abstractmethod
    def forward(self):
        pass


    @abstractmethod
    def backward(self):
        pass


class Dense(Layer): 


    def __init__(self, inputdim: int, units: int, activation: str) -> None:

        self.w = (torch.rand((inputdim, units)) * 2 - 1)
        self.activation = activations[activation]
        self.dz_dw = None
        self.dz_dx = None
        self.da_dz = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, self.dz_dw, self.dz_dx = torch.einsum('ij,jk->ik', x, self.w), x, self.w
        a, self.da_dz = self.activation(z)
        return a


    def backward(self, dl: torch.Tensor, alpha: float) -> torch.Tensor:
        dl_dz = self.da_dz * dl
        dl_dw = torch.einsum('ij,ik->jk', self.dz_dw, dl_dz) / dl.shape[0] 
        dl_dx = torch.einsum('ij,kj->ki', self.dz_dx, dl_dz)
        self.w -= alpha * dl_dw
        return dl_dx


class SoftmaxLayer(Layer): 


    def __init__(self, inputdim: int, units: int) -> None:

        self.w = (torch.rand((inputdim, units)) * 2 - 1)
        self.activation = Softmax 
        self.dz_dw = None
        self.dz_dx = None
        self.da_dz = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, self.dz_dw, self.dz_dx = torch.einsum('ij,jk->ik', x, self.w), x, self.w
        a, self.da_dz = self.activation(z)
        return a


    def backward(self, dl: torch.Tensor, alpha: float) -> torch.Tensor:
        dl_dz = torch.einsum('ijk,ik->ij', self.da_dz, dl)
        dl_dw = torch.einsum('ij,ik->jk', self.dz_dw, dl_dz) / dl.shape[0]
        dl_dx = torch.einsum('ij,kj->ki', self.dz_dx, dl_dz)
        self.w -= alpha * dl_dw

        return dl_dx


def CategoricalCrossEntropy(y: torch.Tensor, yhat: torch.Tensor):

    loss = -1 * torch.sum(y * (torch.log(1e-8 + yhat)), dim=1)
    grad = -1 * (y / yhat)
    return loss, grad            


def main():

    # Load data
    x, y, columns = loadData(DATASET, SAVED)

    # Randomly permute data
    rargs = torch.randperm(x.shape[0])
    x, y = x[rargs], y[rargs]
    x = torch.cat([x, torch.ones((x.shape[0], 1))], dim=1)
    y = torch.Tensor(createOneHotColumn(y.numpy())[0])

    # Train/Test split 80/20
    trnidx = int(x.shape[0] * .8)
    xtrain, ytrain = x[:trnidx], y[:trnidx]
    xtest, ytest = x[trnidx:], y[trnidx:]

    batch = 32 
    alpha = 0.0002
    epochs = 1000

    layers = [
        Dense(inputdim=xtrain.shape[1], units=8, activation='relu'),
        SoftmaxLayer(inputdim=8, units=ytrain.shape[1])
    ]

    epochs = trange(epochs)
    for epoch in epochs:

        rargs = torch.randperm(int(xtrain.shape[0] * 0.8))
        xbatch, ybatch = xtrain[rargs], ytrain[rargs]

        l, start, end = [], 0, batch
        for b in range((xbatch.shape[0]//2) + 1):

            if xbatch[start:end].shape[0] > 0:

                # Forward pass
                ypred = xbatch[start:end] 
                for layer in layers:
                    ypred = layer.forward(ypred)                        

                # Loss
                bl, dl = CategoricalCrossEntropy(ybatch[start:end], ypred)
                l.append(torch.mean(bl).item())

                # Backward pass
                for layer in layers[::-1]:
                    dl = layer.backward(dl, alpha)

            start += batch
            end += batch

        ypred = xtrain
        for layer in layers:
            ypred = layer.forward(ypred)                
        acc = 1 - OneHotErrorRate(ytrain, ypred)
        epochs.set_description('Loss: %.8f | Acc: %.8f' % (sum(l) / len(l), acc))

    ypred = xtest
    for layer in layers:
        ypred = layer.forward(ypred)                
    acc = 1 - OneHotErrorRate(ytest, ypred)
    print('Test Acc: %.4f' % acc)


if __name__ == '__main__':
    main()
