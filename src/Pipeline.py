import numpy as np
import pandas as pd
import re

from collections import Counter

from src.CorrelationMatrix import CorrelationMatrix

from src.util import normalize_name


class Preprocessor:

    def __init__(self):
        return

    def normalize_name(column_name):
        NON_ALPHA = re.compile('[^a-z0-9]+', re.IGNORECASE)
        non_alphas = NON_ALPHA.findall(column_name)
        name = column_name.lower().strip(''.join(non_alphas))
        name = NON_ALPHA.sub('_', name)
        return f'_{name}'

    def is_alpha(self, dtypes):
        mass = float(sum(dtypes.values()))
        if dtypes['str'] / mass >= 0.85:
            return True
        return False

    def is_num(self, dtypes):
        mass = float(sum(dtypes.values()))
        if dtypes['float'] / mass >= 0.85 \
        or dtypes['int'] / mass >= 0.85:
            return True
        return False

    def impute_defaultval(self, most_freq__dtype):
        if most_freq__dtype == float:
            return 0.0
        elif most_freq__dtype == int:
            return 0
        else:
            return 'no_value'

    def __call__(self, df, drop=[], keep=[]):
        """
        :param df:  The input dataset containing the matrix that must undergo
                    dimensionality reduction.
        :type df:   pandas.DataFrame

        :rtype:     tuple[pandas.DataFrame, pandas.DataFrame, dict]
        :return:    A triple <XN, XA, m> where:
                    - `XN` is a copy of the subset of input `DataFrame` `df` containing
                      continuous dimensions (either `float` or `int`).
                    - `XA` is a copy of the subset containing categorial dimensions
                      (`str`).
                    - A `dict[str, str]` mapping where the keys are the character-
                      normalized versions of the column names in the input `DataFrame`,
                      and the values are the corresponding original column names, so that
                      the dimensionality-reduced continuous data can be merged with the
                      categorial data.

        """

        # Remove columns manually dropped
        if drop:
            df.drop(drop, axis=1, inplace=True)

        # Remove invariant columns
        for col in df.columns:
            if len(set(df[col])) == 1:
                df.drop(col, axis=1, inplace=True)

        columns_num, columns_alpha, columns_drop = [], [], []
        for col in df.columns:

            df.select_dtypes(object).fillna('None', inplace=True)
            df.select_dtypes(float).fillna(0.0, inplace=True)
            df.select_dtypes(int).fillna(0, inplace=True)

            if col in keep:
                columns_alpha.append(col)
                continue

            dtypes = Counter([val.__class__.__name__ for val in df[col]])
            most_freq__dtype = dtypes.most_common(1)[0] if dtypes.items() else None

            if self.is_alpha(dtypes):
                columns_alpha.append(col)
            elif self.is_num(dtypes):
                columns_num.append(col)
            elif most_freq__dtype in [float, int]:
                most_freq__dtype_str = str(most_freq__dtype)
                print(f'Info: imputing values for type {most_freq__dtype_str} in column "{col}".')
                df[col] = df[col].fillna(self.impute_defaultval(most_freq__dtype))
            elif most_freq__dtype:
                columns_alpha.append(col)
            else:
                dtypes_str = str(dtypes_str)
                print(f'Warning: column "{col}" contains data types that cannot be handled and will be dropped: {dtypes_str}.')
                columns_drop.append(col)


        # Separate numerical and non-numerical columns.
        df_alpha = df.copy()
        df_alpha.drop(columns_num, axis=1, inplace=True)

        df_num = df.copy()
        df_num.drop(columns_alpha, axis=1, inplace=True)


        # Drop unrecognized columns
        df_num.drop(columns_drop, axis=1, inplace=True)


        # Rescaling not necessary if using a matrix of correlations.
#         from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
#         for col in df_num.columns:
#             sclr = StandardScaler()
#             df_num[col] = sclr.fit_transform(np.array(df_num[col]).reshape(-1, 1))

#         # Fill empty cells
#         for col in columns_num:
#             df_num[col].fillna(0, inplace=True)

        # Normalize column names to PEP
        column_mapping = {
            column: normalize_name(column)
            for column in df_num.columns
        }
        df_num.rename(columns=column_mapping, inplace=True)

#         for col in df_num.columns:
#             print(col, set(df_num[col]), '\n' * 3)

        return df_num, df_alpha, {val: key for key, val in column_mapping.items()}



class Postprocessor:

    def __init__(self):
        return

    def __call__(self, df_alpha, df_num, column_mapping, key=[]):
        """
        :param df_num:  ...........
        :type df_num:   pandas.DataFrame

        :param df_alpha:  ...........
        :type df_alpha:   pandas.DataFrame

        :param column_mapping:  ...........
        :type column_mapping:   dict

        :rtype:     pandas.DataFrame
        :return:    ...........

        """
        if not key:
            merged = df_alpha.copy()
            for col in df_num.columns:
                merged[column_mapping[col]] = df_num[col]
        else:
            original_keys = key
            current_keys = []
            for col in original_keys:
                mapped = False
                for key, val in column_mapping.items():
                    if val == col:
                        current_keys.append(key)
                        mapped = True
                        break
                if not mapped:
                    current_keys.append(col)

            columns = dict([])
            A = df_alpha.columns
            N = df_num.columns
            for col, _col in zip(original_keys, current_keys):
                if col == _col:
                    columns[col] = df_alpha[col]
                elif col != _col and _col in N:
                    columns[col] = df_num[_col]
            merged = pd.DataFrame(columns)

        return merged



def pipeline(df, n=2, verbose=0, drop=[], keep=[]):
    preproc = Preprocessor()
    posproc = Postprocessor()
    correl = CorrelationMatrix(verbose=verbose)

    df_num, df_alpha, column_mapping = preproc(df, drop=drop, keep=keep)

    df_red = correl(df_num, n=n)

    df_pos = posproc(df_alpha, df_red, column_mapping, key=list(df.columns))

    return df_pos