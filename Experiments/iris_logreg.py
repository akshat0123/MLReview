from mlr.Models.LinearModel import SoftmaxRegressionClassifier, OneHotErrorRate
from mlr.Preprocessing.Preprocessing import createOneHotColumn
import torch

from utils import loadData


SAVED = './data.pickle'
DATASET = 'Iris'


def main():

    # Load data
    x, y, columns = loadData(DATASET, SAVED)
    y = torch.Tensor(createOneHotColumn(y.numpy())[0])

    # Randomly permute data
    rargs = torch.randperm(x.shape[0])
    x, y = x[rargs], y[rargs]

    # Train/Test split 80/20
    trnidx = int(x.shape[0] * .8)
    xtrain, ytrain = x[:trnidx], y[:trnidx]
    xtest, ytest = x[trnidx:], y[trnidx:]

    # Train
    alpha, batch, epochs = 1e-1, 8, 1000
    clf = SoftmaxRegressionClassifier()
    clf.fit(xtrain, ytrain, alpha=alpha, epochs=epochs, batch=batch)

    # Test
    ypred = clf.predict(xtest)
    accuracy = 1 - OneHotErrorRate(ytest, ypred)
    print('Accuracy: %.4f' % accuracy)


if __name__ == '__main__':
    main()
