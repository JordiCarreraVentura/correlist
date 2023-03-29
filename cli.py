import argparse
import sys


HELP_DELIMITER = 'Record-separting character. Defaults to ";".'

HELP_DROP = 'A list of column names. Any columns in this list will be dropped forcefully and will not be included in the output DataFrame. Defaults to an empty list.'

HELP_KEEP = 'A list of column names. Any columns in this list will be kept forcefully and returned in the output DataFrame. Defaults to an empty list.'

HELP__N_DIMS = 'Maximum number of (numerical) dimensions from the CSV to keep, after sorting them by explanatory power. Required argument.'

HELP_OUT = 'Location where the output will be written, either a string denoting a path in the local filesystem, or nothing, in which case all output will be written to `stdout`.'

HELP__PATH_CSV = 'Local path to the CSV file to be analyzed. Required argument.'

HELP_TEST = 'Runs the tests. Defaults to `False.`'

HELP_VERBOSE = 'Set to `True` to enable verbosity. Defaults to `False`.'


class CLI:
    def __init__(self):
        return

    def __call__(self):
        prsr = argparse.ArgumentParser()
        prsr.add_argument('path_csv', type=str, help=HELP__PATH_CSV)
        prsr.add_argument('n_dims', type=int, help=HELP__N_DIMS)
        prsr.add_argument('-d', '--delimiter', type=str, help=HELP_DELIMITER, default=";")
        prsr.add_argument('--keep', nargs='*', type=str, default=[], help=HELP_DROP)
        prsr.add_argument('--drop', nargs='*', type=str, default=[], help=HELP_KEEP)
        prsr.add_argument('-o', '--out', type=str, default=sys.stdout, help=HELP_OUT)
        prsr.add_argument('--test', action="store_true", default=False, help=HELP_TEST)
        prsr.add_argument(
            '--verbose', action="store_true", default=False, help=HELP_VERBOSE
        )
        self.__dict__.update(prsr.parse_args().__dict__)
