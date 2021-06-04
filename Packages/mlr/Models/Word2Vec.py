from typing import List
import random

from tqdm import trange, tqdm
import numpy as np


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


def csim(x: np.ndarray, y: np.ndarray) -> float:
    """ Calculate cosine similarity between two vectors
    """

    num = np.dot(x, y.T)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return num / den


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
    """ Iterator for sequence of windows on list
    """


    def __init__(self, items: List, wsize: int) -> None:
        """ Instantiate Windows class

        Args:
            items: list to create windows form
            wsize: total number of items to account for on either side of item
                Given the list ['this', 'is', 'a', 'test'] with a wsize of 1 the
                Windows class will return the following windows in a for loop:
                    ['this', 'is']
                    ['this', 'is', 'a']
                    ['is', 'a', 'test']
                    ['a', 'test']
        """
        self.start = -((wsize+1)//2)
        self.end = ((wsize+1)//2)+1
        self.items = items 
        self.cidx = 0


    def __iter__(self):
        """ Returns iterator
        """
        return self


    def __next__(self) -> List:
        """ Return next available window of list

        Returns:
            window of items as a list
        """

        if self.cidx < len(self.items):

            if self.end > len(self.items):
                seq = self.items[self.start:]

            elif self.start < 0:
                seq = self.items[:self.end]

            else:
                seq = self.items[self.start:self.end]

            self.start += 1
            self.cidx += 1
            self.end += 1
            return seq

        else:
            raise StopIteration


class Word2Vec:
    """ Implementation of Word2Vec word embeddings using skip-gram negative
        sampling
    """


    def __init__(self, vocab: List[str], probs: List[float], d: int):
        """ Instantiate word2vec class

        Args:
            vocab: list of vocabulary words
            probs: unigram probabilities of each vocabulary word
            d: size of embeddings
        """

        self.w = (np.random.rand(len(vocab), d) - 0.5) / 100
        self.c = (np.random.rand(len(vocab), d) - 0.5) / 100
        self.sampler = WeightedSampler(vocab, probs)
        self.vocab = vocab
        self.vdict = { vocab[idx]: idx for idx in range(len(vocab)) }

    
    def parse(self, document: str) -> List[str]:
        """ Split document into tokens on white spaces

        Args:
            document: string to parse

        Returns:
            list of parsed tokens 
        """
        return [self.vdict[word] for word in document.strip().split(' ')]


    def train(self, corpus: List[str], epochs: int, wsize: int, k: int, alpha: float) -> None:
        """ Train word2vec model

        Args:
            corpus: list of document strings to train on
            epochs: number of passes to take over training corpus
            wsize: size of window to examine for skip-grams
            k: number of negative samples to consider for each positive sample
            alpha: learning rate
        """

        for i in range(epochs):
            for document in tqdm(corpus, desc='Training Word2Vec Model'):
                words = self.parse(document)

                cidx = 0
                for seq in Windows(words, wsize):
                    cpi = seq[:cidx] + seq[cidx+1:]
                    wi = [words[cidx]]

                    for pi in cpi:
                        cni = [self.vdict[self.sampler.generateWord()] for i in range(k)]
                        self.weightUpdate(wi, pi, cni, alpha)

                    cidx += 1


    def weightUpdate(self, wi: List[int], pi: List[int], cni: List[int], alpha: float) -> None:
        """ Update word2vec weights

        Args:
            wi: list of index for word being trained on
            pi: list of index for positive examples
            cni: list of indices for negative examples
            alpha: learning rate
        """

        # Dot products of positive examples and word and negative examples and word
        cpw = np.einsum('ij,kj->i', self.c[[pi], :], self.w[wi, :])
        cnw = np.einsum('ij,kj->i', self.c[cni, :], self.w[wi, :])

        # Sigmoid function run on dot products
        scpw = (1 / (1 + np.exp(-cpw))) - 1
        scnw = (1 / (1 + np.exp(-cnw)))[:, None]

        # Gradient for matrix w
        dwa = self.c[[pi], :] * scpw
        dwb = np.sum(self.c[cni, :] * scnw, axis=0)[None, :]
        dw = dwa + dwb

        # Gradients for matrix c
        dcp = self.w[wi, :] * scpw
        dcn = self.w[wi, :] * scnw

        # Weight updates using calculated gradients
        self.w[wi, :] = self.w[wi, :] - (alpha * dw)
        self.c[[pi], :] = self.c[[pi], :] - (alpha * dcp)
        self.c[cni, :] = self.c[cni, :] - (alpha * dcn)
