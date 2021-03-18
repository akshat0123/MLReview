from mlr.Models.LinearModel import Perceptron
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
    alpha, epochs = 0.0001, 100
    clf = Perceptron()
    clf.fit(xtrain, ytrain, alpha, epochs)

    # Test
    ypred = clf.predict(xtest)    
    error = ErrorRate(ytest, ypred)
    print('Test Error: %.4f' % error)
            

if __name__ == '__main__':
    main()
