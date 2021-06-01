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
        list of vocabulary terms sorted by probability
        list of probabilities corresponding to the list of vocabulary terms
    """

    # Collect unigram counts
    probs = {}
    for doc in tqdm(dataset):
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


def calcCumulativeProbs(probs: List[float]) -> List[float]:
    """ Takes in list of probabilities and returns list of cumulative
        probabilities (relative to provided order of probabilities)

    Args:
        probs: list of probabilities

    Returns:
        list of cumulative probabilities
    """

    cprobs = [probs[0]]
    for i in range(1, len(probs)):
        cprobs.append(probs[i] + cprobs[-1])

    return cprobs


def binarySearch(nums: List[float], target: float) -> int:
    """ Return index where float should be located in list

    Args:
        nums: list of floats
        target: float to search for

    Returns:
        index where target float should exist in list
    """

    if len(nums) == 1:
        return 0

    else:
       hlfidx = len(nums) // 2 
       if target < nums[hlfidx]: 
           return binarySearch(nums[:hlfidx], target)

       else: 
           return hlfidx + binarySearch(nums[hlfidx:], target)


def weightUpdate(w, c, wi, pi, cni, alpha):

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

    num = np.dot(x, y.T)
    den = np.linalg.norm(x) * np.linalg.norm(y) 
    return num / den 


def main():

    # Load data
    xtrain, xtest, _ = loadData(DATASET, SAVED)
    xtrain = xtrain

    vocab, probs = getUnigramProbs(xtrain)
    cprobs = calcCumulativeProbs(probs)
    vdict = { vocab[idx]: idx for idx in range(len(vocab)) } 

    v, d, wsize, k, alpha, epochs = len(vocab), 50, 4, 4, 1e-2, 1
    w, c = (np.random.rand(v, d) - 0.5) / 100, (np.random.rand(v, d) - 0.5) / 100

    for i in range(epochs):
        for doc in tqdm(xtrain):
            words = [vdict[word] for word in doc.strip().split(' ')]

            start = -((wsize+1)//2) 
            end = ((wsize+1)//2)+1 
            cidx = 0

            while cidx < len(words):

                if end > len(words): seq = words[start:]
                elif start < 0: seq = words[:end]
                else: seq = words[start:end]

                cpi = seq[:cidx] + seq[cidx+1:]
                wi = [words[cidx]]

                for pi in cpi:
                    cni = [binarySearch(cprobs, random.random()) for i in range(k)]
                    weightUpdate(w, c, wi, pi, cni, alpha)

                cidx += 1; start += 1; end += 1


if __name__ == '__main__':
    main()
