from util import df_to_list


class CorrelationMatrix:

    def __init__(self):
        return

    def __call__(self, pd):
        correl = pd.corr()
        Y = df_to_list(correl)
        return Y