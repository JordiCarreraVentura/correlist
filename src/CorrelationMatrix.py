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
        self.variance = 0.0
        self.uid = id(self) if uid == None else uid

    def fit_variance(self):
        self.variance = variance(self.cells)

    def __str__(self):
        return '<Column<uid={} variance={}>>'.format(
            str(self.uid), self.variance
        )

    def cosine(self, column):
        return cosine(np.array(self.cells), np.array(column.cells))

    def __iter__(self):
        for cell in self.cells:
            yield cell

    def normalize(self, vec_sum):
        for idx, cell in enumerate(vec_sum):
            self.cells[idx] /= cell

    def __iadd__(self, column):
        for idx, cell in enumerate(column):
            self.cells[idx] += cell
        return self

    def __isub__(self, column):
        for idx, cell in enumerate(column):
            self.cells[idx] -= cell
        return self

    def penalize(self, penalty_vec):
        for idx, cell in enumerate(penalty_vec):
            val = self.cells[idx]
            self.cells[idx] = val * (1 - cell)




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


    def __call__(self, pd, n=5, radius=1.0):
        correl = pd.corr()
        columns = self.__matrix__to__column_list(correl)
        for column in columns:
            column.fit_variance()
        columns.sort(reverse=True, key=lambda x: x.variance)

        print('columns:', [str(c) for c in columns])

        reduced = []
        while columns and len(reduced) < n:

            overlapping = set([])
            sims = dict([])
            for jdx, column in enumerate(columns):
                _max = 0.0
                for idx, prev in enumerate(reduced):
                    sim = column.cosine(prev)
                    sims[jdx] = idx
                    if sim >= radius and sim > _max:
                        _max = sim
                if _max > 0.0:
                    overlapping.add(idx)

            _columns = []
            for idx, column in enumerate(columns):
                if idx in overlapping:
                    continue
                _column = cp(columns[idx])
                if reduced:
                    _column -= reduced[sims[idx]]
                _column.fit_variance()
                _columns.append(_column)

            columns = _columns
            columns.sort(reverse=True, key=lambda x: x.variance)
            column = columns.pop(0)
            print('columns SELECT:', column)
            reduced.append(column)

            print('columns IN:', [str(c) for c in columns])
            print('reduced:', [str(c) for c in reduced])
            print()

        return reduced