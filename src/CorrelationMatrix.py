import numpy as np

from scipy.spatial.distance import cosine

from statistics import (
    median,
    pvariance as variance
)

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
#         return '<Column uid={}>'.format(
#             str(self.uid)
#         )
        return '<Column<uid={} revariance={}>>'.format(
            str(self.uid), self.revariance()
        )


    def __len__(self):
        return len(self.cells)

    def revariance(self):
        return variance(self.cells)

    def minimum(self):
        mdn = median(self.cells)
        #return -sum(abs(cell - mdn) for cell in self.cells)
        return -sum(abs(cell) for cell in self.cells)

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
        columns.sort(reverse=False, key=lambda x: x.minimum())
        #columns.sort(reverse=False, key=lambda x: x.revariance())

        #print('columns:', '\n'.join([str(c) for c in columns]))

        reduced = []
        while columns and len(reduced) < n:
            L = len(columns)
            sim_argmaxs = dict([(i, None) for i in range(L)])
            sim_maxs = dict([(i, None) for i in range(L)])
            for jdx, column in enumerate(columns):
                for idx, prev in enumerate(reduced):
                    sim = 1 - column.cosine(prev)
                    if sim_maxs[jdx] == None \
                    or sim > sim_maxs[jdx]:
                        #print('sim={} prev={}'.format(sim, sim_maxs[jdx]))
                        sim_argmaxs[jdx] = idx
                        sim_maxs[jdx] = sim

            _columns = []
            if reduced:
                for jdx, column in enumerate(columns):
                    _column = cp(columns[jdx])
                    if jdx not in sim_argmaxs:
                        _columns.append(_column)
                        continue
                    _column -= Column(reduced[sim_argmaxs[jdx]].base)
                    _column.normalize()
                    _columns.append(_column)
                columns = _columns

            #columns.sort(reverse=True, key=lambda x: x.revariance())
            columns.sort(reverse=False, key=lambda x: x.minimum())

            print('reduced ({}):'.format(len(reduced)), '\n'.join([str(c) for c in reduced]))
            print('columns IN:', '\n'.join([str(c) for c in columns]))
            column = columns.pop(0)

            print('columns SELECT:', column)
            reduced.append(column)

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