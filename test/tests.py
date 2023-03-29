import pandas as pd
import random

from copy import deepcopy as cp
from tqdm import tqdm

from src.util import (
    cosine,
    df_to_list,
    normalize_name
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
        feature_weight=(0.0, 1.0)
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



def test__warcraft_data():

    def preprocess_test(df):

        def split_damage(cell):
            if '-' not in cell:
                return int(cell)
            a, b = cell.split('-')
            a = int(a)
            b = int(b)
            return int(sum([a, b]) / 2)

        columns_num = [
            col for col in df.columns
            if col not in ['Race', 'Unit', 'Other']
        ]

        columns_alpha = [
            col for col in df.columns
            if col in ['Race', 'Unit', 'Other']
        ]

        # non-numerical:    Race	Unit    Other
        df_alpha = df.copy()

        df_alpha.drop(columns_num, axis=1, inplace=True)

        # numerical:        HP	Mana	Gold	Wood	Oil	Armor	Pop.	Sight	Speed	Damage	P. Damage	Range	Build Time
        df_num = df.copy()
        df_num.drop(columns_alpha, axis=1, inplace=True)

        # Separate numerical and non-numerical columns.

        # Rescaling not necessary if using a matrix of correlations.

        # Remove invariant columns
        for col in df_num.columns:
            if len(set(df_num[col])) == 1:
                df_num.drop(col, axis=1, inplace=True)

        # Fill empty cells
        df_num['Mana'].fillna(0, inplace=True)
        df_num['Oil'].fillna(0, inplace=True)
        df_num['Damage'] = df_num['Damage'].apply(split_damage)
        df_num['Damage'] = df_num['Damage'].astype(int)

        # Normalize column names to PEP
        column_renaming = {
            column: normalize_name(column)
            for column in df_num.columns
        }
        df_num.rename(columns=column_renaming, inplace=True)

#         for col in df_num.columns:
#             print(col, set(df_num[col]), '\n' * 3)

        return df_num, df_alpha


    df = pd.read_csv('data/test_data_1.csv', delimiter=';')

    df_num, df_alpha = preprocess_test(df)

    from src.CorrelationMatrix import CorrelationMatrix
    correl = CorrelationMatrix(verbose=0)
    correl_X = correl(df_num, n=2)

    expected = set(['_wood', '_speed'])
    assert set(correl_X.columns).intersection(expected) == expected




def test__warlords_data():

    def preprocess_test(df):

        columns_num = [
            col for col in df.columns
            if col not in ['Unit', 'M.Bonus', 'C.Bonus', 'Power']
        ]

        columns_alpha = [
            col for col in df.columns
            if col in ['Unit', 'M.Bonus', 'C.Bonus', 'Power']
        ]

        # non-numerical:    Race	Unit    Other
        df_alpha = df.copy()
        df_alpha.drop(columns_num, axis=1, inplace=True)

        # numerical:        HP	Mana	Gold	Wood	Oil	Armor	Pop.	Sight	Speed	Damage	P. Damage	Range	Build Time
        df_num = df.copy()
        df_num.drop(columns_alpha, axis=1, inplace=True)

        # Separate numerical and non-numerical columns.

        # Rescaling not necessary if using a matrix of correlations.

        # Remove invariant columns
        for col in df_num.columns:
            if len(set(df_num[col])) == 1:
                df_num.drop(col, axis=1, inplace=True)

        # Fill empty cells

        # Normalize column names to PEP
        column_renaming = {
            column: normalize_name(column)
            for column in df_num.columns
        }
        df_num.rename(columns=column_renaming, inplace=True)

#         for col in df_num.columns:
#             print(col, set(df_num[col]), '\n' * 3)

        return df_num, df_alpha


    df = pd.read_csv('data/test_data_2.csv', delimiter=';')

    df_num, df_alpha = preprocess_test(df)


    from src.CorrelationMatrix import CorrelationMatrix
    correl = CorrelationMatrix(verbose=0)
    correl_X = correl(df_num, n=2)

    expected = set(['_setup', '_mdl'])
    assert set(correl_X.columns).intersection(expected) == expected



def test_preprocessor():
    from src.Pipeline import Preprocessor
    preproc = Preprocessor()

    df = pd.DataFrame({
        'num_1': [0, 1, 2, 3],
        'num_2': [0.0, 1.0, 2.5, 3.5],
        'alpha_1': ['A', 'A', 'B', 'B']
    })

    df_num, df_alpha, column_mapping = preproc(df)

    expected = set(['_num_1', '_num_2'])
    excluded = set(['alpha_1'])

    assert column_mapping == {
        '_num_1': 'num_1',
        '_num_2': 'num_2',
    }
    assert set(df_num.columns).intersection(expected) == expected
    assert set(df_alpha.columns).intersection(excluded) == excluded




def test_postprocessor():
    from src.Pipeline import Postprocessor
    posproc = Postprocessor()

    df_num = pd.DataFrame({
        '_num_1': [0, 1, 2, 3],
        '_num_2': [0.0, 1.0, 2.5, 3.5]
    })

    df_alpha = pd.DataFrame({
        'alpha_1': ['A', 'A', 'B', 'B']
    })

    column_mapping = {
        '_num_1': 'num_1',
        '_num_2': 'num_2',
    }

    df = posproc(df_alpha, df_num, column_mapping, key='num_1 num_2 alpha_1'.split())

    assert df.values.tolist() == pd.DataFrame({
        'num_1': [0, 1, 2, 3],
        'num_2': [0.0, 1.0, 2.5, 3.5],
        'alpha_1': ['A', 'A', 'B', 'B']
    }).values.tolist()


def test_pipeline():
    from src.Pipeline import pipeline
    df = pd.read_csv('data/test_data_2.csv', delimiter=';')
    df_ = pipeline(df, n=2, verbose=1)

    expected = set(['Unit', 'Setup', 'M.Bonus', 'C.Bonus', 'Power', 'Mdl'])

    assert df.shape == (134, 13)
    assert df_.shape == (134, 6)
    assert set(df_.columns).intersection(expected) == expected




def tests():

    tests = [
        test__matrix_shape,
        test__redundant_features,
        test__feature_weights,
        test__correlation_matrix,
        test__warcraft_data,
        test__warlords_data,
        test_preprocessor,
        test_postprocessor,
        test_pipeline,
    ]
    for test in tqdm(tests):
        print(test.__name__)
        test()


# def tests():
#     return []