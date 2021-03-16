from mlr.Preprocessing.Preprocessing import createOneHotColumn
from mlr.Models.Activation import *
from mlr.Models.Loss import *
from tqdm import trange, tqdm
import torch


class LogisticRegressionClassifier:

    def __init__(self) -> None:
        """ Instantiate logistic regression classifier
        """

        self.w = None
        self.calcError = ErrorRate


    def fit(self, x, y, alpha=1e-4, epochs=1000, batch=32) -> None:
        """ Fit logistic regression classifier to dataset

        Args:
            x: input data
            y: input labels
            alpha: alpha parameter for weight update
            epochs: number of epochs to train
            batch: size of batches for training
        """

        self.w = torch.rand((1, x.shape[1]))

        epochs = trange(epochs, desc='Error')
        for epoch in epochs:

            start, end = 0, batch
            for b in range((x.shape[0]//batch)+1):
                hx = self.probs(x[start:end])
                dw = self.calcGradient(x[start:end], y[start:end], hx)
                self.w = self.w - (alpha * dw)
                start += batch
                end += batch

            hx = self.predict(x)
            error = self.calcError(y, hx)
            epochs.set_description('Err: %.4f' % error)


    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """ Determine probability of label being 1

        Args:
            x: input data

        Returns:
            probability for each member of input
        """

        hx = 1 / (1 + torch.exp(-torch.einsum('ij,kj->i', x, self.w)))[:, None]
        return hx


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Predict labels

        Args:
            x: input data

        Returns:
            labels for each member of input
        """

        hx = self.probs(x)
        hx = (hx >= 0.5).float()
        return hx


    def calcGradient(self, x: torch.Tensor, y: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        """ Calculate weight gradient

        Args:
            x: input data
            y: input labels
            hx: predicted probabilities

        Returns:
            tensor of gradient values the same size as weights
        """

        return torch.sum(x * (hx - y), dim=0) / x.shape[0]


class SoftmaxRegressionClassifier:


    def __init__(self) -> None:
        """ Instantiate softmax regression classifier
        """

        self.w = None
        self.calcError = OneHotErrorRate


    def fit(self, x, y, alpha=1e-4, epochs=1000, batch=32):
        """ Fit logistic regression classifier to dataset

        Args:
            x: input data
            y: input labels
            alpha: alpha parameter for weight update
            epochs: number of epochs to train
            batch: size of batches for training
        """

        y = torch.Tensor(createOneHotColumn(y.numpy())[0])
        self.w = torch.rand((y.shape[1], x.shape[1]))

        epochs = trange(epochs, desc='Accuracy')
        for epoch in epochs:

            rargs = torch.randperm(x.shape[0])
            x, y = x[rargs], y[rargs]

            start, end = 0, batch
            for b in range((x.shape[0]//batch)+1):
                if start < x.shape[0]:
                    sz = self.probs(x[start:end]) 
                    dw = self.calcGradient(x[start:end], y[start:end], sz)
                    self.w = self.w - alpha * dw

                start += batch
                end += batch

            sz = self.probs(x)
            accuracy = 1 - self.calcError(y, sz)
            epochs.set_description('Accuracy: %.4f' % accuracy)


    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """ Predict probabilities of belonging to each class

        Args:
            x: input data

        Returns:
            probabilities for each member of input
        """

        return Softmax(torch.einsum('ij,kj->ik', x, self.w))


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Predict labels

        Args:
            x: input data

        Returns:
            labels for each member of input
        """

        hx = self.probs(x)
        return torch.argmax(hx, dim=1)[:, None]


    def calcGradient(self, x: torch.Tensor, y: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """ Calculate weight gradient

        Args:
            x: input data
            y: input labels
            probs: predicted probabilities

        Returns:
            tensor of gradient values the same size as weights
        """

        return torch.einsum('ij,ik->jk', probs - y , x) / x.shape[0]


class LinearRegressor:


    def __init__(self) -> None:
        """ Instantiate linear regressor 
        """

        self.w = None
        self.calcError = MeanSquaredError


    def fit(self, x: torch.Tensor, y: torch.Tensor, alpha: float=0.00001, epochs: int=1000, batch: int=32) -> None:
        """ Fit logistic regression classifier to dataset

        Args:
            x: input data
            y: input labels
            alpha: alpha parameter for weight update
            epochs: number of epochs to train
            batch: size of batches for training
        """

        self.w = torch.zeros((1, x.shape[1]))

        epochs = trange(epochs, desc='Error')
        for epoch in epochs:

            start, end = 0, batch
            for b in range((x.shape[0]//batch)+1):
                hx = self.predict(x[start:end])
                dw = self.calcGradient(x[start:end], y[start:end], hx)
                self.w = self.w - (alpha * dw)
                start += batch
                end += batch

            hx = self.predict(x)
            error = self.calcError(y, hx)
            epochs.set_description('MSE: %.4f' % error)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ Predict output values 

        Args:
            x: input data

        Returns:
            regression output for each member of input
        """


        return torch.einsum('ij,kj->i', x, self.w)


    def calcGradient(self, x: torch.Tensor, y: torch.tensor, hx: torch.Tensor) -> torch.Tensor:
        """ Calculate weight gradient

        Args:
            x: input data
            y: input labels
            hx: predicted output

        Returns:
            tensor of gradient values the same size as weights
        """

        return torch.einsum('ij,i->j', -x, (y - hx)) / x.shape[0]


