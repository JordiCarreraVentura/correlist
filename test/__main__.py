from __init__ import make__test_data

from tqdm import tqdm


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

    X = make__test_data(
        n_classes=4,
        n_docs=100,
        n_features=5,
        feature_weight=(min__feature_weight, 1.0)
    )

    for row in X:
        positive_dims = [dim for dim in row if dim > min__feature_weight]
        assert len(positive_dims) == 5

    X = make__test_data(
        n_classes=4,
        n_docs=100,
        n_features=5,
        feature_weight=(0.5, 1.0)
    )

    for row in X:
        positive_dims = [dim for dim in row if dim > min__feature_weight]
        assert len(positive_dims) < 5



if __name__ == '__main__':

    tests = [
        test__matrix_shape,
        test__redundant_features,
        test__feature_weights,
    ]
    for test in tqdm(tests):
        print(test.__name__)
        test()
