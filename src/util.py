import csv
import re
from copy import deepcopy as cp

from scipy.spatial.distance import cosine


def autoconfigure():
    import os, sys
    from pathlib import Path
    new = str(Path(os.path.realpath(__file__)).parent.parent)
    if new not in sys.path:
        sys.path.append(new)

    for obj in os.listdir(new):
        path = Path(new) / obj
        full_path = str(path)
        if not path.is_dir() \
        or obj.startswith('.') \
        or obj.startswith('__'):
            continue
        if full_path not in sys.path:
            sys.path.append(full_path)

    #print(sys.path)


def df_to_list(df):
    """
    >>> import pandas as pd
    >>> row1 = [1, 1, 1]
    >>> row2 = [2, 2, 2]
    >>> row3 = [3, 3, 3]
    >>> rows = [row1, row2, row3]
    >>> df = pd.DataFrame(rows)
    >>> assert df.shape == (3, 3)
    >>> assert df[0][0] == 1
    >>> assert df[1][0] == 1
    >>> assert df[1][1] == 2
    >>> assert df[2][1] == 2
    >>> assert df[0][2] == 3
    >>> assert df[2][2] == 3
    >>> assert df_to_list(df) == rows
    """
    cols = list(df.columns)
    X = []
    for row in df.itertuples(index=False):
        X.append([row[cols.index(col)] for col in cols])
    return X


def normalize_name(column_name):
    """
    >>> assert normalize_name('alpha') == '_alpha'
    >>> assert normalize_name('.alpha') == '_alpha'
    >>> assert normalize_name('alpha..') == '_alpha'
    >>> assert normalize_name('1alpha') == '_1alpha'
    >>> assert normalize_name('alpha_beta!_gamma^_123') == '_alpha_beta_gamma_123'
    >>> assert normalize_name('ThisIs...ATest') == '_thisis_atest'
    """
    NON_ALPHA = re.compile('[^a-z0-9]+', re.IGNORECASE)
    non_alphas = NON_ALPHA.findall(column_name)
    name = column_name.lower().strip(''.join(non_alphas))
    name = NON_ALPHA.sub('_', name)
    return f'_{name}'


def write_output(df, out):
    if out:
        df.to_csv(out, index=False)
    else:
        for row in csv.writer(df.iterrows(index=False), quoting=csv.QUOTEALL):
            out.write(row)
    return True


if __name__ == '__main__':
    import doctest
    doctest.testmod()