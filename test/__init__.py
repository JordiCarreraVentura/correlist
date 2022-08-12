import random

from copy import deepcopy as cp


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
