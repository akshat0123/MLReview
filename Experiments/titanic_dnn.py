from typing import Callable, List

from mlr.NN.Loss import BinaryCrossEntropy 
from mlr.Models.Loss import ErrorRate 
from mlr.NN.Layer import Dense
from tqdm import trange, tqdm
import torch

from utils import loadData


SAVED = './data.pickle'
DATASET = 'Titanic'


def main():

    # Load data
    x, y, columns = loadData(DATASET, SAVED)

    # Train/Test split 80/20
    trnidx = int(x.shape[0] * .8)
    xtrain, ytrain = x[:trnidx], y[:trnidx]
    xtest, ytest = x[trnidx:], y[trnidx:]

    batch = 128 
    alpha = 0.01
    epochs = 1000

    # Train
    layers = [
        Dense(xtrain.shape[1], 16, activation='relu'), 
        Dense(16, 1, activation='sigmoid')
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
                bl, dl = BinaryCrossEntropy(ybatch[start:end], ypred)
                l.append(torch.mean(bl).item())

                # Backward pass
                for layer in layers[::-1]:
                    dl = layer.backward(dl, alpha)

            start += batch
            end += batch

        ypred = xtrain
        for layer in layers:
            ypred = layer.forward(ypred)                
        ypred[ypred >= 0.5] = 1
        ypred[ypred < 0.5] = 0

        acc = 1 - ErrorRate(ytrain, ypred)
        epochs.set_description('Loss: %.8f | Acc: %.8f' % (sum(l) / len(l), acc))

    # Test
    ypred = xtest
    for layer in layers:
        ypred = layer.forward(ypred)                
    ypred[ypred >= 0.5] = 1
    ypred[ypred < 0.5] = 0
    error = ErrorRate(ytest, ypred)
    print('Test Acc: %.8f' % acc)

if __name__ == '__main__':
    main()
