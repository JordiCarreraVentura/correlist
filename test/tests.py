import pandas as pd
import random

from copy import deepcopy as cp
from tqdm import tqdm


def make__test_data(n_classes=10, n_docs=100, n_features=1, feature_weight=(0.5, 1.0)):

    def add_class(n, n_docs, n_features, feature_weight, X, D):
        wmin, wmax = feature_weight
        feature_range__start = n * n_features
        feature_idxs = D[feature_range__start:feature_range__start + n_features]
        for _ in range(n_docs):
            vec = [random.uniform(-1, wmin) for _ in D]
            for idx in feature_idxs:
                vec[idx] = random.uniform(wmin, wmax)
            X.append(vec)

    X = []
    D = [idx for idx in range(n_classes * n_features)]
    n = 0
    while n < n_classes:
        add_class(n, n_docs, n_features, feature_weight, X, D)
        n += 1
    return X



def make__correlation_matrix(X):
    df = pd.DataFrame(X)
    from src.CorrelationMatrix import CorrelationMatrix
    correl = CorrelationMatrix()
    Y = correl(df)
    for row in Y:
        print(row)



def test__matrix_shape():
    import numpy as np

    n_features = 1
    n_classes = 5
    n_docs = 10

    _X = make__test_data(n_classes=n_classes, n_docs=n_docs, n_features=n_features)
    X = np.array(_X)
    assert X.shape == (n_docs * n_classes, n_classes * n_features)

    for row in X:
        positive_dims = [dim for dim in row if dim > 0.5]
        assert len(positive_dims) == 1



def test__redundant_features():
    import numpy as np

    n_classes = 2
    n_docs = 5
    n_features = 5

    _X = make__test_data(n_classes=n_classes, n_docs=n_docs, n_features=n_features)
    X = np.array(_X)
    assert X.shape == (n_docs * n_classes, n_classes * n_features)

    for row in X:
        positive_dims = [dim for dim in row if dim > 0.5]
        assert len(positive_dims) == n_features



def test__feature_weights():

    min__feature_weight = 0.9
    n_features = 65

    X = make__test_data(
        n_classes=4,
        n_docs=100,
        n_features=n_features,
        feature_weight=(min__feature_weight, 1.0)
    )

    for row in X:
        positive_dims = [dim for dim in row if dim > min__feature_weight]
        assert len(positive_dims) == n_features

    X = make__test_data(
        n_classes=4,
        n_docs=100,
        n_features=n_features,
        feature_weight=(0.5, 1.0)
    )

    for row in X:
        positive_dims = [dim for dim in row if dim > min__feature_weight]
        assert len(positive_dims) < n_features



def test__correlation_matrix():
    X = make__test_data(
        n_classes=2,
        n_docs=10,
        n_features=5,
        feature_weight=(0.5, 1.0)
    )
    correl_X = make__correlation_matrix(X)


def tests():

    tests = [
        test__matrix_shape,
        test__redundant_features,
        test__feature_weights,
        test__correlation_matrix,
    ]
    for test in tqdm(tests):
        print(test.__name__)
        test()


# def tests():
#     return []