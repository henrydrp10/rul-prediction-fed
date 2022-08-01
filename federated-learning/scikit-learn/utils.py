import numpy as np
from sklearn.ensemble import RandomForestRegressor

# from typing import Tuple, Union, List
# XY = Tuple[np.ndarray, np.ndarray]
# Dataset = Tuple[XY, XY]
# LogRegParams = Union[XY, Tuple[np.ndarray]]
# XYList = List[XY]


def get_model_parameters(model):
    return model.get_params()


def set_model_params(model, parameters):
    return model.set_params(parameters)


# def set_initial_params(model: RandomForestRegressor):
#     """Sets initial parameters as zeros Required since model params are
#     uninitialized until model.fit is called.

#     But server asks for initial parameters from clients at launch. Refer
#     to sklearn.linear_model.LogisticRegression documentation for more
#     information.
#     """
#     n_classes = 10  # MNIST has 10 classes
#     n_features = 784  # Number of features in dataset
#     model.classes_ = np.array([i for i in range(10)])

#     model.coef_ = np.zeros((n_classes, n_features))
#     if model.fit_intercept:
#         model.intercept_ = np.zeros((n_classes,))


# def load_mnist() -> Dataset:
#     """Loads the MNIST dataset using OpenML.

#     OpenML dataset link: https://www.openml.org/d/554
#     """
#     mnist_openml = openml.datasets.get_dataset(554)
#     Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
#     X = Xy[:, :-1]  # the last column contains labels
#     y = Xy[:, -1]
#     # First 60000 samples consist of the train set
#     x_train, y_train = X[:60000], y[:60000]
#     x_test, y_test = X[60000:], y[60000:]
#     return (x_train, y_train), (x_test, y_test)


# def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
#     """Shuffle X and y."""
#     rng = np.random.default_rng()
#     idx = rng.permutation(len(X))
#     return X[idx], y[idx]


# def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
#     """Split X and y into a number of partitions."""
#     return list(
#         zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
#    )