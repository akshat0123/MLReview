from mlr.Models.Ensemble import GradientBoostedTreeClassifier 
from mlr.Models.Loss import ErrorRate
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

    # Train/Test split 80/20
    trnidx = int(x.shape[0] * .8)
    xtrain, ytrain = x[:trnidx], y[:trnidx]
    xtest, ytest = x[trnidx:], y[trnidx:]

    # Train
    alpha = 0.001
    clf = GradientBoostedTreeClassifier()
    clf.fit(xtrain, ytrain, alpha)

    # Test
    ypred = clf.predict(xtest)
    accuracy = 1 - ErrorRate(ytest, ypred)
    print('Accuracy: %.4f' % accuracy)


if __name__ == '__main__':
    main()
