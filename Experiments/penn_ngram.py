from typing import List

from mlr.Models.LanguageModel import NGramLanguageModel
from tqdm import trange, tqdm

from utils import loadData


SAVED = './data.pickle'
DATASET = 'PennTreebank' 


def main():

    # Load data
    xtrain, xtest, _ = loadData(DATASET, SAVED)

    lm = NGramLanguageModel(n=3)

    for line in tqdm(xtrain): 
        lm.add(line)

    lm.calcSmoothingCounts()    
    perplexity = lm.perplexity(xtest)

    print(perplexity)
    

if __name__ == '__main__':
    main()
