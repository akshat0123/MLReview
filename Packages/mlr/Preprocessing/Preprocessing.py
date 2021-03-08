import numpy as np


def getColReport(data: np.ndarray, columns: np.ndarray, nullval: str='') -> str:
    """ Return basic information about each column in dataset, including number
        of unique values, number of blanks, as well as a sample of the data in each
        column

    Args:
        data: input data
        columns: list of column names 
        nullval: value to accept as Null for this dataset

    Returns:
        a string value with a line of information for each column
    """
    report = ''

    for i in range(len(columns)):
        colname, numvals = str(columns[i]), str(len(np.unique(data[:, i])))
        sample = ','.join(data[0:10, i])
        blanks = len(np.argwhere(data[:, i]==nullval))
        report += '{:<20} {:<20} {:<20} {:<20}\n'.format(colname, numvals, blanks, sample)

    return report        


def replaceBlanks(data: np.ndarray, columns: np.ndarray, colmeta: dict, nullval: str='') -> None:
    """ Replace all the blanks in the dataset 

    Args:
        data: input data
        columns: list of column names
        colmeta: dictionary of metadata information about each column, with the
                 following structure:

                    colmeta = { 
                        [COLUMN NAME]:  {
                            'type': [ONE OF 'skip', 'hot', 'float', OR 'cardinal'],
                            'null': [NULL VALUE],
                            'vals': [
                                DICTIONARY MAPPING CARDINAL VALUES TO INTEGERS ONLY
                                NECESSARY IF 'type' IS SET TO 'cardinal'
                            ]
                        }, ...
                    }

        nullval: value to accept as Null for dataset
    """

    for col in colmeta:
        colidx = np.argwhere(columns==col)[0]
        if colmeta[col]['null'] == 'colref':
            ncolidx = np.argwhere(columns==colmeta[col]['colref'])[0]
            nrows = np.argwhere(data[:, colidx]==nullval)
            data[nrows, colidx] = data[nrows, ncolidx]

        elif colmeta[col]['null'] is not None:
            nval = colmeta[col]['null']
            nrows = np.argwhere(data[:, colidx]==nullval)
            data[nrows, colidx] = nval


def removeSkips(data: np.ndarray, columns: np.ndarray, colmeta: dict) -> None:
    """ Remove all columns designated to be skipped

    Args:
        data: input data
        columns: list of column names
        colmeta: dictionary of metadata information about each column, with the
                 following structure:

                    colmeta = { 
                        [COLUMN NAME]:  {
                            'type': [ONE OF 'skip', 'hot', 'float', OR 'cardinal'],
                            'null': [NULL VALUE],
                            'vals': [
                                DICTIONARY MAPPING CARDINAL VALUES TO INTEGERS ONLY
                                NECESSARY IF 'type' IS SET TO 'cardinal'
                            ]
                        }, ...
                    }
    """

    for col in colmeta:
        if colmeta[col]['type'] == 'skip':
            colidxs = np.argwhere(columns!=col).flatten()
            columns = columns[colidxs]
            data = data[:, colidxs]


def createOneHotColumn(data: np.ndarray, colidx: int=0) -> (np.ndarray, np.ndarray):
    """ Turn single categorical column to one-hot columns

    Args:
        data: input data
        colidx: column to transform

    Returns:
        one-hot vectors of specified column
        list of column names for new one-hot vectors 
    """

    ucols = np.unique(data[:, colidx])
    uidxs = { ucols[i]: i for i in range(len(ucols)) }
    urows = np.zeros((data.shape[0], ucols.shape[0]))

    for ridx in range(data.shape[0]):
        urows[ridx, uidxs[data[ridx, colidx]]] = 1.0

    return urows, ucols        


def encodeDataset(data: np.ndarray, columns: np.ndarray, colmeta: dict, nullval: str='') -> (np.ndarray, np.ndarray):
    """ Encodes dataset into a numpy array of type float

    Args:
        data: input data
        columns: list of column names
        colmeta: dictionary of metadata information about each column, with the
                 following structure:

                    colmeta = { 
                        [COLUMN NAME]:  {
                            'type': [ONE OF 'skip', 'hot', 'float', OR 'cardinal'],
                            'null': [NULL VALUE],
                            'vals': [
                                DICTIONARY MAPPING CARDINAL VALUES TO INTEGERS ONLY
                                NECESSARY IF 'type' IS SET TO 'cardinal'
                            ]
                        }, ...
                    }

        nullval: value to accept as Null for dataset
    """

    # Replace blanks as necessary
    replaceBlanks(data, columns, colmeta, nullval)

    # Remove columns to be skipped
    removeSkips(data, columns, colmeta)

    # Transform dataset
    ndata, ncolumns = [], []
    for col in columns:
        try:
            colidx = np.argwhere(columns==col)[0, 0]

            # Transform string columns to float
            if colmeta[col]['type'] == 'float':
                ndata.append(data[:, colidx].astype(np.float64).reshape((-1, 1)))
                ncolumns.append(col)

            # Create one hot encoding columns
            elif colmeta[col]['type'] == 'hot':
                urows, ucols = createOneHotColumn(data, colidx)
                ndata.append(urows)
                ncolumns += ['%s_%s' % (col, ucols[i]) for i in range(len(ucols))]

            # Create cardinal encoding columns
            elif colmeta[col]['type'] == 'cardinal':
                ndata.append(np.asarray([colmeta[col]['vals'][data[i, colidx]] for i in range(data.shape[0])]).reshape((-1, 1)))
                ncolumns.append(col)

        except Exception as e:
            print('Error with column %s' % col)

    return np.concatenate(ndata, axis=1), np.asarray(ncolumns)


def minMaxScale(x: np.ndarray) -> None:
    """ Min-max scale numpy array column-wise in-place

    Args:
        x: data to be scaled
    """

    for i in range(x.shape[1]):
        xmax, xmin = np.max(x[:, i]), np.min(x[:, i]) 
        if xmax - xmin != 0:
            x[:, i] = (x[:, i] - xmin) / (xmax - xmin)
