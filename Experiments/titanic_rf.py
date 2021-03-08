from mlr.Models.Ensemble import RandomForestClassifier
import torch

from utils import loadData


PATH = '../Datasets/Titanic/train.csv'
SAVED = './data.pickle'
DATASET = 'Titanic'


def main():

    # Load data
    x, y, columns = loadData(DATASET, SAVED)

    # Train/Test split 80/20
    trnidx = int(x.shape[0] * .8)
    xtrain, ytrain = x[:trnidx], y[:trnidx]
    xtest, ytest = x[trnidx:], y[trnidx:]
    classes = [c.item() for c in torch.unique(ytrain)]

    # Train
    forest = RandomForestClassifier(numTrees=10, maxDepth=5, leafSize=1, bootstrapRatio=0.1)
    forest.fit(xtrain, ytrain, classes)

    # Test
    ypred = forest.predict(xtest)
    acc = torch.sum((ytest==ypred).float()) / ytest.shape[0]
    print('Accuracy: %.4f' % acc)


if __name__ == '__main__':
    main()
