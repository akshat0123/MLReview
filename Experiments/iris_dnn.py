from mlr.Preprocessing.Preprocessing import createOneHotColumn
from mlr.NN.Loss import CategoricalCrossEntropy 
from mlr.Models.Loss import OneHotErrorRate
from mlr.NN.Layer import Dense
from tqdm import trange, tqdm
import torch

from utils import loadData


SAVED = './data.pickle'
DATASET = 'Iris'


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

    batch = 8 
    alpha = 0.1
    epochs = 1000

    layers = [
        Dense(inputdim=xtrain.shape[1], units=8, activation='relu'),
        Dense(inputdim=8, units=ytrain.shape[1], activation='softmax')
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
