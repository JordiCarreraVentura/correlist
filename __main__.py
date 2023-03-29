import pandas as pd

from cli import CLI
from src.Pipeline import pipeline
from src.util import autoconfigure, write_output
from test.tests import tests

autoconfigure()


if __name__ == '__main__':

    cli = CLI()
    cli()

    if cli.test:
        tests()

    df = pd.read_csv(cli.path_csv, delimiter=cli.delimiter)
    df_ = pipeline(df, n=cli.n_dims, verbose=cli.verbose, drop=cli.drop, keep=cli.keep)

    write_output(df_, cli.out)