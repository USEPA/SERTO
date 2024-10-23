"""
The serto.chamaopt module provides the functionality for CHAMA optimization.
"""

# python imports
import argparse

# third party imports

# local imports


def get_description():
    return 'Run the CHAMA optimization'


def configure_subparser(sub_parser: argparse.ArgumentParser):
    """
    Configure the subparser for the chamaoptimizer command.
    :param sub_parser:
    :param sub_parser: The
    :return:
    """

    parser = sub_parser.add_parser(
        'chama',
        help='Run CHAMA optimization functions'
    )

    parser.add_argument(
        '-s',
        '--step',
        choices=['gen_sensors', 'sensormodel', 'sensormodel'],
        required=True,
        help='The CHAMA optimization step to run'
             'gen_sensors: Generate sensor locations'
    )


def process_args(args: argparse.Namespace):
    """
    Process the arguments for the chamaoptimizer command.
    :param args:
    :return:
    """

    pass
