from mlr.NN.Metric import MeanSquaredError
from mlr.NN.Layer import Dense
from mlr.NN.Model import Model
import torch

from utils import loadData


SAVED = './data.pickle'
DATASET = 'Grades'


def main():

    # Load data
    x, y, columns = loadData(DATASET, SAVED)

    # Train/Test split 80/20
    trnidx = int(x.shape[0] * .8)
    xtrain, ytrain = x[:trnidx], y[:trnidx][:, None]
    xtest, ytest = x[trnidx:], y[trnidx:][:, None]

    # Train
    alpha, batch, epochs, lambdaa = 1e-4, 4, 100, 1e-2
    dnn = Model([
        Dense(inputdim=xtrain.shape[1], units=32, activation='relu', initializer='glorot', regularizer='l2'),
        Dense(inputdim=32, units=1, activation='linear', regularizer='l2')
    ], loss='mean_squared_error')

    # Test
    dnn.fit(x=xtrain, y=ytrain, batch=batch, alpha=alpha, epochs=epochs, lambdaa=lambdaa)
    ypred = dnn.predict(xtest)
    print('Test MSE: %.4f' % MeanSquaredError(ytest, ypred))


if __name__ == '__main__':
    main()
