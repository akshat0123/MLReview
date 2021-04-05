from mlr.Models.DecisionTree import DecisionTreeRegressor
from mlr.Models.Loss import MeanSquaredError
import torch

from utils import loadData


SAVED = './data.pickle'
DATASET = 'Grades'


def main():

    # Load data
    x, y, columns = loadData(DATASET, SAVED)

    # Train/Test split 80/20
    trnidx = int(x.shape[0] * .8)
    xtrain, ytrain = x[:trnidx], y[:trnidx]
    xtest, ytest = x[trnidx:], y[trnidx:]

    # Train
    clf = DecisionTreeRegressor(maxDepth=2)
    clf.fit(xtrain, ytrain)

    # Test
    ypred = clf.predict(xtest)    
    error = MeanSquaredError(ytest, ypred)
    print('Test MSE: %.4f' % error)


if __name__ == '__main__':
    main()
