from typing import List
import random

from tqdm import trange, tqdm
import numpy as np
import torch

from utils import loadData


SAVED = './data.pickle'
DATASET = 'PennTreebank' 


def getUnigramProbs(dataset: List[str], alpha: float=0.75) -> (List[str], List[float]):
    """ Calculate unigram probabilities for all words in dataset. Probabilities
        are weighted to increase the probability of the rarest words

    Args:
        dataset: list of strings
        alpha: weighting hypterparameter

    Returns:
        list of vocabulary terms sorted by highest to lowest probability
        list of probabilities corresponding to the list of vocabulary terms
    """

    # Collect unigram counts
    probs = {}
    for doc in tqdm(dataset, desc='Calculating Unigram Probabilities'):
        words = doc.strip().split(' ')

        for word in words: 
            probs[word] = probs[word] + 1 if word in probs else 1

    # Transform counts into probabilities
    total = 0
    for word in probs: total += probs[word]**alpha
    for word in probs: probs[word] = (probs[word]**alpha) / total

    # Sort vocabulary by probabilities
    probs = sorted(probs.items(), key=lambda x: x[1])[::-1]
    vocab, probs = [pair[0] for pair in probs], [pair[1] for pair in probs]

    return vocab, probs


class WeightedSampler:
    """ Implementation of Walker's Alias method for weighted random sampling
    """


    def __init__(self, vocab: List[str], probs: List[float]):
        """ Initialize weighted sampler class

        Args:
            vocab: list of vocabulary words
            probs: list of unigram probabilities for each word
        """

        self.buckets = self.getRandomSamplingBuckets(vocab, probs, 1/len(vocab))
        self.tprob = 1 / len(vocab)


    def initializeBuckets(self, buckets: List[dict], tprob: float) -> (List[dict], List[dict], List[dict]):
        """ Initialize list of buckets into one of overfull, underfull and full
            categories. Buckets are dictionaries containing two vocabulary words
            'w1' and 'w2', as well as a splitting probability 's', and a sum of
            probabilities 't'

        Args:
            buckets: list of buckets
            tprob: the total probability that each bucket must represent

        Returns:
            list of buckets that are full
            list of buckets that are underfull
            list of buckets that are overfull
        """

        full, underfull, overfull = [], [], []

        while buckets:
            cur = buckets.pop()
            if cur['t'] == tprob:
                full.append(cur)

            elif cur['t'] < tprob:
                underfull.append(cur)

            else:
                overfull.append(cur)

        return full, underfull, overfull


    def getRandomSamplingBuckets(self, vocab: List[str], probs: List[float], tprob: float) -> List[dict]:
        """ Reorganize buckets until each bucket contains two vocabulary words
            and all buckets have an equivalent total probability. Buckets are
            dictionaries containing two vocabulary words 'w1' and 'w2', as well
            as a splitting probability 's', and a sum of probabilities 't'

        Args:
            vocab: list of vocabulary words
            probs: list of unigram probabilities for each word
            tprob: the total probability that each bucket must represent

        Returns:
            list of buckets s.t. all buckets have two words and the same total probability
        """

        buckets = [ {'w1': vocab[idx], 'w2': None, 's': probs[idx], 't': probs[idx] } for idx in range(len(vocab)) ]
        full, underfull, overfull = self.initializeBuckets(buckets, tprob)

        while underfull:
            under = underfull.pop()
            over = overfull.pop()

            under['w2'] = over['w1']
            under['t'] = tprob 
            full.append(under)

            over['t'] = over['t'] - (tprob - under['s'])
            over['s'] = over['s'] - (tprob - under['s'])

            if over['t'] == tprob: 
                full.append(over)

            elif over['t'] < tprob: 
                underfull.append(over)                

            else: 
                overfull.append(over)

        full += overfull
        return full


    def generateWord(self) -> str:
        """ Generate randomly sampled word 

        Returns:
            randomly sampled word as string  
        """

        bucket = random.choice(self.buckets)
        splitval = random.uniform(0, self.tprob)

        word = None
        if splitval >= bucket['s']:
            word = bucket['w2']                

        else:
            word = bucket['w1']                

        return word


class Windows:
    """
    """


    def __init__(self, words, wsize):
        """
        """
        self.words = words
        self.start = -((wsize+1)//2)
        self.end = ((wsize+1)//2)+1
        self.cidx = 0


    def __iter__(self):
        """
        """
        return self


    def __next__(self):
        """
        """

        if self.cidx < len(self.words):

            if self.end > len(self.words):
                seq = self.words[self.start:]

            elif self.start < 0:
                seq = self.words[:self.end]

            else:
                seq = self.words[self.start:self.end]

            self.start += 1
            self.cidx += 1
            self.end += 1
            return seq

        else:
            raise StopIteration


def weightUpdate(w, c, wi, pi, cni, alpha):
    """
    """

    cpw = np.einsum('ij,kj->i', c[[pi], :], w[wi, :])
    cnw = np.einsum('ij,kj->i', c[cni, :], w[wi, :])

    scpw = (1 / (1 + np.exp(-cpw))) - 1
    scnw = (1 / (1 + np.exp(-cnw)))[:, None]

    dwa = c[[pi], :] * scpw
    dwb = np.sum(c[cni, :] * scnw, axis=0)[None, :]

    dw = dwa + dwb
    dcp = w[wi, :] * scpw
    dcn = w[wi, :] * scnw

    w[wi, :] = w[wi, :] - (alpha * dw)
    c[[pi], :] = c[[pi], :] - (alpha * dcp)
    c[cni, :] = c[cni, :] - (alpha * dcn)


def csim(x, y):
    """
    """

    num = np.dot(x, y.T)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return num / den


class Word2Vec:


    def __init__(self, vocab, probs, d):
        """
        """

        self.w = (np.random.rand(len(vocab), d) - 0.5) / 100
        self.c = (np.random.rand(len(vocab), d) - 0.5) / 100
        self.sampler = WeightedSampler(vocab, probs)
        self.vocab = vocab
        self.vdict = { vocab[idx]: idx for idx in range(len(vocab)) }

    
    def parse(self, document):
        """
        """
        return [self.vdict[word] for word in document.strip().split(' ')]


    def train(self, corpus, epochs, wsize, k, alpha):
        """
        """

        for i in range(epochs):
            for document in tqdm(corpus):
                words = self.parse(document)

                cidx = 0
                for seq in Windows(words, wsize):
                    cpi = seq[:cidx] + seq[cidx+1:]
                    wi = [words[cidx]]

                    for pi in cpi:
                        cni = [self.vdict[self.sampler.generateWord()] for i in range(k)]
                        self.weightUpdate(wi, pi, cni, alpha)

                    cidx += 1


    def weightUpdate(self, wi, pi, cni, alpha):
        """
        """

        cpw = np.einsum('ij,kj->i', self.c[[pi], :], self.w[wi, :])
        cnw = np.einsum('ij,kj->i', self.c[cni, :], self.w[wi, :])

        scpw = (1 / (1 + np.exp(-cpw))) - 1
        scnw = (1 / (1 + np.exp(-cnw)))[:, None]

        dwa = self.c[[pi], :] * scpw
        dwb = np.sum(self.c[cni, :] * scnw, axis=0)[None, :]

        dw = dwa + dwb
        dcp = self.w[wi, :] * scpw
        dcn = self.w[wi, :] * scnw

        self.w[wi, :] = self.w[wi, :] - (alpha * dw)
        self.c[[pi], :] = self.c[[pi], :] - (alpha * dcp)
        self.c[cni, :] = self.c[cni, :] - (alpha * dcn)



def main():

    # # Load data
    # xtrain, xtest, _ = loadData(DATASET, SAVED)
    # xtrain = xtrain

    # vocab, probs = getUnigramProbs(xtrain)
    # w2v = Word2Vec(vocab, probs, 50)
    # w2v.train(xtrain[:100], 1, 4, 4, 1e-2)

    # Load data
    xtrain, xtest, _ = loadData(DATASET, SAVED)
    xtrain = xtrain

    vocab, probs = getUnigramProbs(xtrain)
    vdict = { vocab[idx]: idx for idx in range(len(vocab)) } 
    sampler = WeightedSampler(vocab, probs)        

    alpha = 1e-2
    epochs = 1
    wsize = 4
    d = 50
    k = 4
    w = (np.random.rand(len(vocab), d) - 0.5) / 100
    c = (np.random.rand(len(vocab), d) - 0.5) / 100

    for i in range(epochs):
        for doc in tqdm(xtrain):
            words = [vdict[word] for word in doc.strip().split(' ')]

            start = -((wsize+1)//2) 
            end = ((wsize+1)//2)+1 
            cidx = 0

            while cidx < len(words):

                if end > len(words): 
                    seq = words[start:]

                elif start < 0: 
                    seq = words[:end]

                else: 
                    seq = words[start:end]

                cpi = seq[:cidx] + seq[cidx+1:]
                wi = [words[cidx]]

                for pi in cpi:
                    cni = [vdict[sampler.generateWord()] for i in range(k)]
                    weightUpdate(w, c, wi, pi, cni, alpha)

                start += 1 
                cidx += 1
                end += 1

    # # Save embeddings
    # embeddings = (w + c).astype(str)
    # vocab = np.asarray(vocab)[:, None]
    # embeddings = np.concatenate((vocab, embeddings), axis=1)
    # np.savetxt('w2v.csv', embeddings, fmt='%s', delimiter=',')

    # # Load Embeddings
    # embeddings = np.loadtxt('w2v.csv', dtype=str, delimiter=',', comments=None) 
    # vocab = embeddings[:, 0]
    # vdict = { vocab[idx]: idx for idx in range(vocab.shape[0])}
    # embeddings = embeddings[:, 1:].astype(np.float)


if __name__ == '__main__':
    main()
