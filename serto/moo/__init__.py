"""
The serto.moo module provides the functionality for multi-objective evolutionary algorithms.
"""
# python imports
import argparse


# third party imports

# local imports


def get_description():
    return 'Run the CHAMA optimization'


def configure_subparser(subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the subparser for the chamaoptimizer command.
    :param subparser:
    :return:
    """

    return subparser


def process_args(args: argparse.Namespace):
    """
    Process the arguments for the chamaoptimizer command.
    :param args:
    :return:
    """
    pass
