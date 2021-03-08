from mlr.Models.DecisionTree import DecisionTreeClassifier
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
    classes = [c.item() for c in torch.unique(ytrain)]

    # Train
    tree = DecisionTreeClassifier(maxDepth=2)
    tree.fit(xtrain, ytrain, classes)

    # Test
    ypred = tree.predict(xtest)
    acc = torch.sum((ytest==ypred).float()) / ytest.shape[0]
    print('Test Accuracy: %.4f' % acc)


if __name__ == '__main__':
    main()
