"""
The serto.analysis module provides the functionality for various analysis tasks.
"""

# python imports
import argparse
import pandas as pd


# third party imports

# local imports
from .plumes import configure_subparsers as configure_plume_parsers


def get_description():
    return 'Run the CHAMA optimization'


def configure_subparser(sub_parsers: argparse.ArgumentParser):
    """
    Configure the subparser for the chamaoptimizer command.
    :param sub_parser:
    :return:
    """
    analysis_parser = sub_parsers.add_parser('analysis', help='Perform specific analysis')
    analysis_subparsers = analysis_parser.add_subparsers(
        title='analysis',
        description='Analysis tasks to execute',
        help='Additional help',
        dest='analysis_command'
    )

    configure_plume_parsers(analysis_subparsers)



    # Wind sample
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


def main(args: argparse.Namespace):
    """
    """
    if args.analysis.wind.sample:
        sample(parser_args.csv_file, parser_args.dir, parser_args.speed, parser_args.dbins)
