from typing import Callable, List

from mlr.NN.Layer import Dense
from mlr.NN.Loss import *
from mlr.NN.Model import Model
from mlr.Models.Loss import ErrorRate
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

    # Train
    dnn = Model([Dense(xtrain.shape[1], 256, activation='relu'), Dense(256, 1, activation='sigmoid')])
    alpha, batch, epochs = 0.1, 128, 1000
    dnn.fit(xtrain, ytrain, alpha, epochs, batch, loss='cross_entropy')

    # Test
    yhat = dnn.predict(xtest)
    error = ErrorRate(ytest, yhat)
    print('Test Err: %.4f' % error)


if __name__ == '__main__':
    main()
