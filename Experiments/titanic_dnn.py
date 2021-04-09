from mlr.NN.Metric import Accuracy 
from mlr.NN.Layer import Dense
from mlr.NN.Model import Model
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
    alpha, batch, epochs = 0.01, 128, 1000
    dnn = Model([
        Dense(inputdim=xtrain.shape[1], units=16, activation='relu', initializer='he'),
        Dense(inputdim=16, units=1, activation='sigmoid', initializer='glorot')
    ], loss='binary_cross_entropy')

    # Test
    dnn.fit(x=xtrain, y=ytrain, batch=batch, alpha=alpha, epochs=epochs)
    ypred = dnn.predict(xtest)
    print('Test Acc: %.4f' % Accuracy(ytest, ypred))


if __name__ == '__main__':
    main()
