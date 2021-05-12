from typing import List
import pickle, csv, os

from tqdm import trange, tqdm
import numpy as np
import torch

from mlr.Preprocessing.Preprocessing import *


def loadTitanicData(path: str) -> (torch.Tensor, torch.Tensor, np.ndarray):
    """ Load Titanic dataset

    Args:
        path: path to dataset

    Returns:
        dataset encoded to floats 
        labels
        column names
    """

    reader = csv.reader(open(path, 'r'))
    data = [line for line in reader]
    columns, data = np.asarray(data[0]), np.asarray(data[1:])
    
    # Print to determine how to encode columns
    colreport = getColReport(data, columns)

    # Set column encoding metadata by hand using column report
    colmeta = {
        'PassengerId': { 'type': 'skip', 'null': None },
        'Survived': { 'type': 'float', 'null': None },
        'Pclass': { 'type': 'hot', 'null': None },
        'Name': { 'type': 'skip', 'null': None },
        'Sex': { 'type': 'hot', 'null': None },
        'Age': { 'type': 'float', 'null': '0.0' },
        'SibSp': { 'type': 'hot', 'null': None },
        'Parch': { 'type': 'hot', 'null': None },
        'Ticket': { 'type': 'skip', 'null': None },
        'Fare': { 'type': 'float', 'null': None },
        'Cabin': { 'type': 'hot', 'null': '_' },
        'Embarked': { 'type': 'hot', 'null': '_' }
    }

    # Encode dataset
    data, columns = encodeDataset(data, columns, colmeta)

    # Separate x and y
    xcols = np.argwhere(columns!='Survived').flatten()
    ycol = np.argwhere(columns=='Survived').flatten()[0]
    x, y = data[:, xcols], data[:, ycol]
    columns = columns[xcols]

    # Min-Max Scale dataset
    minMaxScale(x)

    x = torch.Tensor(x)
    y = torch.Tensor(y)[:, None]
    return x, y, columns


def loadGradesData(path: str) -> (torch.Tensor, torch.Tensor, np.ndarray):
    """ Load Grades dataset

    Args:
        path: path to dataset

    Returns:
        dataset encoded to floats 
        labels
        column names
    """

    reader = csv.reader(open(path, 'r'), delimiter=';')
    data = [line for line in reader]
    columns, data = np.asarray(data[0]), np.asarray(data[1:])
    
    # Print to determine how to encode columns
    colreport = getColReport(data, columns)

    # Set column encoding metadata by hand using column report
    colmeta = {
        'school': { 'type': 'hot', 'null': None },
        'sex': { 'type': 'cardinal', 'null': None, 'vals': { 'F': 0, 'M': 1 }},
        'age': { 'type': 'float', 'null': None },
        'address': { 'type': 'cardinal', 'null': None, 'vals': { 'R': 0, 'U': 1 }},
        'famsize': { 'type': 'cardinal', 'null': None, 'vals': { 'LE3': 0, 'GT3': 1 }},
        'Pstatus': { 'type': 'cardinal', 'null': None, 'vals': { 'A': 0, 'T': 1 }},
        'Medu': { 'type': 'float', 'null': None },
        'Fedu': { 'type': 'float', 'null': None },
        'Mjob': { 'type': 'hot', 'null': None },
        'Fjob': { 'type': 'hot', 'null': None },
        'reason': { 'type': 'hot', 'null': None },
        'guardian': { 'type': 'hot', 'null': None },
        'traveltime': { 'type': 'float', 'null': None },
        'studytime': { 'type': 'float', 'null': None },
        'failures': { 'type': 'float', 'null': None },
        'schoolsup': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 }},
        'famsup': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 }},
        'paid': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 }},
        'activities': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 }},
        'nursery': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 } },
        'higher': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 }  },
        'internet': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 }  },
        'romantic': { 'type': 'cardinal', 'null': None, 'vals': { 'no': 0, 'yes': 1 }  },
        'famrel': { 'type': 'float', 'null': None },
        'freetime': { 'type': 'float', 'null': None },
        'goout': { 'type': 'float', 'null': None },
        'Dalc': { 'type': 'float', 'null': None },
        'Walc': { 'type': 'float', 'null': None },
        'health': { 'type': 'float', 'null': None },
        'absences': { 'type': 'float', 'null': None },
        'G1': { 'type': 'skip', 'null': None },
        'G2': { 'type': 'skip', 'null': None },
        'G3': { 'type': 'float', 'null': None }
    }

    # Encode dataset
    data, columns = encodeDataset(data, columns, colmeta)

    # Separate x and y
    xcols = np.argwhere(columns!='G3').flatten()
    ycol = np.argwhere(columns=='G3').flatten()[0]
    x, y = data[:, xcols], data[:, ycol]
    columns = columns[xcols]

    # Min-Max Scale dataset
    minMaxScale(x)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    return x, y, columns


def loadIrisData(path: str) -> (torch.Tensor, torch.Tensor, np.ndarray):
    """ Load Iris dataset

    Args:
        path: path to dataset

    Returns:
        dataset encoded to floats 
        labels
        column names
    """

    reader = csv.reader(open(path, 'r'), delimiter=',')
    data = [line for line in reader]
    columns, data = np.asarray(data[0]), np.asarray(data[1:])
    
    # Print to determine how to encode columns
    colreport = getColReport(data, columns)

    # Set column encoding metadata by hand using column report
    colmeta = {
        'sepalLength': { 'type': 'float', 'null': None },
        'sepalWidth': { 'type': 'float', 'null': None },
        'petalLength': { 'type': 'float', 'null': None },
        'petalWidth': { 'type': 'float', 'null': None },
        'class': { 'type': 'cardinal', 'null': None, 'vals': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2} },
    }

    # Encode dataset
    data, columns = encodeDataset(data, columns, colmeta)

    # Separate x and y
    xcols = np.argwhere(columns!='class').flatten()
    ycol = np.argwhere(columns=='class').flatten()
    x, y = data[:, xcols], data[:, ycol]
    columns = columns[xcols]

    # Min-Max Scale dataset
    minMaxScale(x)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    return x, y, columns


def loadPennData(path: str) -> (List[str], List[str], None):
    """
    """

    trainpath = "%s/%s" % (path, "ptb.train.txt")
    testpath = "%s/%s" % (path, "ptb.test.txt")

    train = [line for line in open(trainpath , 'r').readlines()]
    test = [line for line in open(testpath , 'r').readlines()]

    return train, test, None 


DATASETS = {
    'Iris': {
        'path': '../Datasets/Iris/iris.data',
        'loader': loadIrisData,
        'save': True
    },
    'Titanic': {
        'path': '../Datasets/Titanic/train.csv',
        'loader': loadTitanicData,
        'save': True
    },
    'Grades': {
        'path': '../Datasets/Grades/student-mat.csv',
        'loader': loadGradesData,
        'save': True
    },
    'PennTreebank': {
        'path': '../Datasets/PennTreebank',
        'loader': loadPennData,
        'save': True 
    }
}


def loadData(dataset: str, picklepath: str):
    """ Load one of four datasets from DATASETS

    Args:
        dataset: key name of dataset (from DATASETS dict above)
        picklepath: path to save pickled dataset to for faster loading

    Returns:
        dataset encoded to floats 
        labels
        column names
    """

    if os.path.isfile(picklepath):
        data = pickle.load(open(picklepath, 'rb'))

        if dataset in data:
            data = data[dataset]
            x, y, columns = data['x'], data['y'], data['columns']

        else:
            x, y, columns = DATASETS[dataset]['loader'](DATASETS[dataset]['path'])
            pickle.dump({ dataset: {'x': x, 'y': y, 'columns': columns}}, open(picklepath, 'wb'))

    else:
        x, y, columns = DATASETS[dataset]['loader'](DATASETS[dataset]['path'])
        pickle.dump({ dataset: {'x': x, 'y': y, 'columns': columns}}, open(picklepath, 'wb'))

    return x, y, columns        


