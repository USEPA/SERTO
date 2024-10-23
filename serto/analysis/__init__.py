"""
The serto.analysis module provides the functionality for various analysis tasks.
"""

# python imports
import argparse
import pandas as pd

# third party imports

# local imports


def get_description():
    return 'Run the CHAMA optimization'


def configure_subparser(sub_parser: argparse.ArgumentParser):
    """
    Configure the subparser for the chamaoptimizer command.
    :param sub_parser:
    :return:
    """
    parser = sub_parser.add_parser(
        'wind',
        help='Run wind speed and direction analysis'
    )

    sub_parsers = sub_parser.add_subparsers(
        title='wind',
        description='Wind speed and direction analysis commands to execute',
        help='Additional help'
    )

    # Sample
    sample_parser = sub_parsers.add_parser(
        'sample',
        help='Sample wind speed and direction data'
    )

    sample_parser.add_argument(
        '-f',
        '--csv-file',
        help='Wind speed and direction file',
        required=True,
        action='store'
    )

    sample_parser.add_argument(
        '-d',
        '--dir',
        help='Wind direction column name',
        required=True,
        action='store',
        default='drct',
    )

    sample_parser.add_argument(
        '-s',
        '--speed',
        help='Wind speed column name',
        required=True,
        action='store',
        default='sknt',
    )

    sample_parser.add_argument(
        '--dbins',
        help='Num wind direction bins',
        required=True,
        action='store',
        default='sknt',
    )

    # Plot wind rose

    # Plot Quiver


def sample(data: pd.DataFrame, dir: str, speed: str, dbins: int, *args, **kwargs):
    """
    Sample the wind speed and direction data
    :param data:
    :param dir:
    :param speed:
    :param dbins:
    :param args:
    :param kwargs:
    :return:
    """
    return GaussianPlume(data, dir, speed, dbins)

def process_args(data: pd.DataFrame, dir: str, speed: str,  *args, **kwargs):
    """
    Process the arguments for the chamaoptimizer command.
    :param data:
    :param dir:
    :param speed:
    :param args:
    :param kwargs:
    :return:
    """
    return args
