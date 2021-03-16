# ML Review
This repository contains code and notes on various different machine learning algorithms. All code is written from scratch, with the exception of the use of [numpy][1] and [pytorch][2] libraries to make use of their provided datatypes (np.ndarray, and torch.Tensor respectively).

### Packages
The packages directory contains the code for all algorithms and preprocessing modules. Models are stored in the `mlr/Models` directory, and preprocessing utilities are located in the `mlr/Preprocessing` directory. This directory also includes the [conda][3] environment required to run any of the experiments in the `Experiments` directory. More information on how to set up this environment is described in the [Environment Setup and Running Experiments](#environment-setup-and-running-experiments) section of this file.

### Experiments
This directory contains experiments on datasets using the machine learning packages created in the `Packages` directory of this repository.

### Docs
This directory contains [Jupyter][8] notebooks and [Latex][4] documents containing notes for all the algorithms created in the `Packages` directory of this repository.

### Datasets
This directory contains all the datasets tested in the `Experiments` section of this repository. The following datasets are provided:
- `Grades`: The [Student Performance Dataset][5], used for regression models
- `Iris`: The [Iris][6] dataset, used for multiclass classification models
- `Titanic`: The [Titanic - Machine Learning from Disaster][7] dataset, used for binary classification models

### Environment Setup and Running Experiments
In order to set up the [conda][3] environment used to run experiments in the `Experiments` directory, the conda environment must be created and activated, and the `mlr` package must be installed. This can be done from the root of this project as follows:
```
cd Packages;
conda env create -f environment.yml;
conda activate MLR;
pip install .
```
After this, experiments in the `Experiments` directory can be run from within the newly created `MLR` conda environment.

[1]: https://numpy.org/
[2]: https://pytorch.org/
[3]: https://docs.conda.io/en/latest/
[4]: https://www.latex-project.org/
[5]: https://archive.ics.uci.edu/ml/datasets/student+performance
[6]: https://archive.ics.uci.edu/ml/datasets/iris
[7]: https://www.kaggle.com/c/titanic
[8]: https://jupyter.org/
