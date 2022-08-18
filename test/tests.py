import pandas as pd
import random

from copy import deepcopy as cp
from tqdm import tqdm

from src.util import (
    cosine,
    df_to_list
)



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

    df = pd.DataFrame(X)

    assert df.shape == (20, 10)

    from src.CorrelationMatrix import CorrelationMatrix
    correl = CorrelationMatrix(verbose=1)
    correl_X = correl(df, n=4)

    print(df)
    print(correl_X)
    assert correl_X.shape == (20, 4)

    a, b, c, d = correl_X.columns
    a1 = correl_X[a]
    a2 = correl_X[b]
    b1 = correl_X[c]
    b2 = correl_X[d]

    assert cosine(a1, a2) < 0.4
    assert cosine(b1, b2) < 0.4
    assert cosine(a1, b1) > 0.7
    assert cosine(a1, b2) > 0.7
    assert cosine(a2, b1) > 0.7
    assert cosine(a2, b2) > 0.7
    assert cosine(a1, b2) / cosine(a1, a2) > 3





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