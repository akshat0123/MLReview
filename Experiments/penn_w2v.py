import os

from mlr.Models.Word2Vec import getUnigramProbs, Word2Vec
import numpy as np

from utils import loadData


EMBEDDINGS = 'w2v.csv'
SAVED = './data.pickle'
DATASET = 'PennTreebank' 


def main():

    # Load data
    xtrain, xtest, _ = loadData(DATASET, SAVED)
    xtrain = xtrain

    if os.path.isfile(EMBEDDINGS):

        # Load embeddings
        embeddings = np.loadtxt('w2v.csv', dtype=str, delimiter=',', comments=None) 
        vocab = embeddings[:, 0]
        vdict = { vocab[idx]: idx for idx in range(vocab.shape[0])}
        embeddings = embeddings[:, 1:].astype(np.float)

    else:

        # Creat embeddings
        vocab, probs = getUnigramProbs(xtrain)
        w2v = Word2Vec(vocab, probs, 50)
        w2v.train(xtrain[:100], 1, 4, 4, 1e-2)

        # Save embeddings
        embeddings = (w2v.w + w2v.c).astype(str)
        vocab = np.asarray(vocab)[:, None]
        embeddings = np.concatenate((vocab, embeddings), axis=1)
        np.savetxt('w2v.csv', embeddings, fmt='%s', delimiter=',')


if __name__ == '__main__':
    main()
