import numpy as np

from scipy.spatial.distance import cosine

from statistics import pvariance as variance

from util import (
    cp,
    df_to_list
)


class Column:

    def __init__(self, obj, uid=None):
        if isinstance(obj, list):
            self.cells = cp(obj)
        elif isinstance(obj, Column):
            self.cells = cp(obj.cells)
        else:
            raise TypeError(obj)
        self.base = cp(self.cells)
        self.variance = variance(self.base)
        self.uid = id(self) if uid == None else uid

    def __getitem__(self, idx):
        return self.base[idx]

    def __str__(self):
#         return '<Column<uid={} variance={} revariance={}>>'.format(
#             str(self.uid), self.variance, self.revariance()
#         )
        return '<Column uid={}>'.format(
            str(self.uid)
        )

    def __len__(self):
        return len(self.cells)

    def revariance(self):
        return variance(self.cells)

    def cosine(self, column):
        return cosine(np.array(self.cells), np.array(column.cells))

    def __iter__(self):
        for cell in self.cells:
            yield cell

    def normalize(self):
        for idx, cell in enumerate(self.cells):
            self.cells[idx] /= 2

    def __iadd__(self, column):
        for idx, cell in enumerate(column):
            self.cells[idx] += cell
        return self

    def product(self, column):
        for idx, cell in enumerate(column):
            self.cells[idx] = 1 - (self.cells[idx] * cell)
        return self

    def __isub__(self, column):
        for idx, cell in enumerate(column):
            self.cells[idx] -= cell
        return self




class CorrelationMatrix:

    def __init__(self):
        return

    def __matrix__to__column_list(self, correl):
        Y = df_to_list(correl)
        columns = []
        for idx in range(len(Y[0])):
            column = [row[idx] for row in Y]
            columns.append(Column(column, uid=len(columns)))
        return columns


    def __call__(self, df, n=5, radius=1.0):
        df = df.copy()
        correl = df.corr()
        columns = self.__matrix__to__column_list(correl)
        columns.sort(reverse=True, key=lambda x: x.variance)

        print('columns:', [str(c) for c in columns])

        reduced = []
        while columns and len(reduced) < n:
            L = len(columns)
            sim_argmaxs = dict([(i, None) for i in range(L)])
            sim_maxs = dict([(i, 0.0) for i in range(L)])
            for jdx, column in enumerate(columns):
                for idx, prev in enumerate(reduced):
                    sim = column.cosine(prev)
                    if jdx not in sim_argmaxs \
                    or sim > sim_maxs[jdx]:
                        sim_argmaxs[jdx] = idx
                        sim_maxs[jdx] = sim

            _columns = []
            if reduced:
                for jdx, column in enumerate(columns):
                    _column = cp(columns[jdx])
                    if jdx not in sim_argmaxs:
                        _columns.append(_column)
                        continue
#                     print(_column.cells)
                    _column -= Column(reduced[sim_argmaxs[jdx]].base)
                    _column.normalize()
#                     _column.product(reduced[sim_argmaxs[jdx]])
#                     print(reduced[sim_argmaxs[jdx]].base)
#                     print('--')
#                     print(_column.cells)
#                     input()
                    _columns.append(_column)
                columns = _columns

            columns.sort(reverse=True, key=lambda x: x.revariance())
            column = columns.pop(0)
            print('columns SELECT:', column)
            reduced.append(column)

            print('columns IN:', [str(c) for c in columns])
            print('reduced:', [str(c) for c in reduced])
            print()

        df.drop(
            [
                col  for col in df.columns
                if col not in [_col.uid for _col in reduced]
            ],
            axis=1,
            inplace=True
        )
        return df