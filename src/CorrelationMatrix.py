import numpy as np

from statistics import (
    median,
    pvariance as variance
)

from src.util import (
    cosine,
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
        return -sum(abs(np.array(self.cells)))

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

    def __init__(self, verbose=0):
        self.verbose = verbose

    def __matrix__to__column_list(self, correl):
        Y = df_to_list(correl)
        columns = []
        for idx in range(len(Y[0])):
            column = [row[idx] for row in Y]
            columns.append(Column(column, uid=len(columns)))
        return columns


    def __call__(self, df, n=5):
        """

        :type n: int
        :param n: The number of columns to keep from the total number of input columns.

        :rtype: pandas.DataFrame
        :return: The input dataset without the columns removed by this method, namely,
                 those with the highest correlation with any of the other columns.

        Description
        -----------
        Given a DataFrame with `m` variables (dimensions, columns), this method retains the `n` columns (such that m: `n < m`) whose correlation with the other columns has the highest variance.

        The intuition is that columns that correlate with most other columns (and hence, are likely to be redundant) will show a lower variance, as most correlation coefficients will tend to be higher on average, and the variance across similarly high values can be expected to be low.

        Conversely, columns that alternate strong positive correlations with strong negative correlations and just about the same amount of no correlation events, will have the highest variance (as long as there are not a lot of the latter, since the global average over which to measure the correlation would otherwise be closer to those values and, as a result, most of the values in the distribution would again have low variance with respect to an average dominated by them).
        """
        df = df.copy()
        correl = df.corr()

        if self.verbose:
            print(list(enumerate(df.columns)))

        columns = self.__matrix__to__column_list(correl)
        columns.sort(reverse=False, key=lambda x: x.minimum())
        #columns.sort(reverse=False, key=lambda x: x.revariance())

        reduced = []
        while columns and len(reduced) < n:

            sim_argmaxs = self.__greedy__sim_search(columns, reduced)

            columns = self.__select_and_reweight(columns, reduced, sim_argmaxs)

            if self.verbose:
                print('\ncolumns kept so far ({}):\n{}'.format(len(reduced), '\n'.join([str(c) for c in reduced])))
                print('\ninput columns:\n{}'.format('\n'.join([str(c) for c in columns])))

            column = columns.pop(0)

            if self.verbose:
                print('\nselected column:', column)
                print('\n============\n')
            reduced.append(column)


        if self.verbose:
            print('\ncolumns kept at the end ({}):\n{}'.format(len(reduced), '\n'.join([str(c) for c in reduced])))

        self.__apply_reduction(df, reduced)

        return df


    def __select_and_reweight(self, columns, reduced, sim_argmaxs):
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

        columns.sort(reverse=True, key=lambda x: x.revariance())
        #columns.sort(reverse=False, key=lambda x: x.minimum())

        return columns


    def __greedy__sim_search(self, columns, reduced):
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
        return sim_argmaxs


    def __apply_reduction(self, df, reduced):
        df.drop(
            [
                df.columns[col_idx] for col_idx in range(len(df.columns))
                if col_idx not in [_col.uid for _col in reduced]
            ],
            axis=1,
            inplace=True
        )