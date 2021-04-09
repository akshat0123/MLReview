from mlr.Preprocessing.Preprocessing import createOneHotColumn
from mlr.NN.Metric import Accuracy 
from mlr.NN.Layer import Dense
from mlr.NN.Model import Model
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

    # Train
    alpha, batch, epochs, lambdaa = 1.0, 32, 100, 1e-4
    dnn = Model([
        Dense(inputdim=xtrain.shape[1], units=8, activation='relu', initializer='he', regularizer='l2'),
        Dense(inputdim=8, units=ytrain.shape[1], activation='softmax', initializer='glorot', regularizer='l2')
    ], loss='categorical_cross_entropy')

    # Test
    dnn.fit(x=xtrain, y=ytrain, batch=batch, alpha=alpha, epochs=epochs, lambdaa=lambdaa)
    ypred = dnn.predict(xtest)
    print('Test Acc: %.4f' % Accuracy(ytest, ypred))


if __name__ == '__main__':
    main()
