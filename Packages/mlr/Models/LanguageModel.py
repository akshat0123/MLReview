from typing import List
import re

from tqdm import tqdm
import numpy as np


class NGramLanguageModel:


    def __init__(self, n: int, pad: str="<PAD>") -> None:
        """ Instantiate NGram language model

        Args:
            n: size of NGrams
            pad: string to use as padding
        """
        self.vocab = set([])
        self.ngrams = {}
        self.totals = {}
        self.pad = pad
        self.n = n


    def parse(self, line: str) -> List[str]:
        """ Parse string and turn it into list of tokens, removing all
            non-alphanumeric characters and splitting by space

        Args:
            line: string to be parsed

        Returns:
            list of parsed tokens
        """
        line = re.sub("[^a-z\s]", "", line.lower().strip())
        tokens = [t for t in line.split(' ') if t != '']
        return tokens


    def add(self, line: str) -> None:
        """ Add line to language model

        Args:
            line: string to be added to language model
        """
        seq = [self.pad for i in range(self.n)]
        tokens = self.parse(line)

        # Split list of tokens into NGrams and add to model
        for i in range(len(tokens)):
            seq = seq[1:] + [tokens[i]]                
            tseq = tuple(seq[:-1])

            if tseq in self.ngrams:

                if tokens[i] in self.ngrams[tseq]: 
                    self.ngrams[tseq][tokens[i]] += 1

                else: 
                    self.ngrams[tseq][tokens[i]] = 1

            else: 
                self.ngrams[tseq] = { tokens[i]: 1 }

            self.vocab.add(tokens[i])


    def calcSmoothingCounts(self) -> None:
        """ After all lines are added, this method can be called to generate
            counts required for Good-Turing smoothing
        """

        self.totalCounts = 0
        self.counts = {} 

        for ngram in tqdm(self.ngrams):
            for token in self.ngrams[ngram]:
                c = self.ngrams[ngram][token]
                self.counts[c] = self.counts[c] + 1 if c in self.counts else 1
                self.totalCounts += 1


    def prob(self, sequence: List[str]) -> float:
        """ Calculate probability of sequence being produced by language model,
            smoothed using Good-Turing smoothing

        Args: 
            sequence: sequence as a list of tokens

        Returns:
            probability of language model producing sequence                
        """

        tseq = tuple(sequence[:-1])

        c = 0
        if tseq in self.ngrams and \
           sequence[-1] in self.ngrams[tseq]:
            c += self.ngrams[tseq][sequence[-1]]           

        ncn = self.counts[c+1] if c+1 in self.counts else 0
        ncd = self.counts[c] if c in self.counts else 0
        n = ncn / ncd if ncd != 0 else 0

        cstar = (c + 1) * n
        return cstar / self.totalCounts


    def perplexity(self, dataset: List[str]) -> float:
        """ Calculate preplexity of dataset with regard to model

        Args:
            dataset: list of string sequences in testing dataset

        Returns:
            perplexity score                
        """

        perp = 0; N = 0;
        for line in dataset:

            seq = [self.pad for i in range(self.n)]
            tokens = self.parse(line)
            N += len(tokens)

            for i in range(len(tokens)):
                seq = seq[1:] + [tokens[i]]                
                prob = self.prob(seq)
                perp += np.log(prob) if prob != 0 else 0

        perp = np.exp((-1/N) * perp)
        return perp


