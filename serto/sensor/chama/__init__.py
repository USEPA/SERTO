"""
This module contains the subparser configuration for the chamaoptimizer command.
"""

# python imports
import argparse
import pandas as pd


# third party imports

# local imports


def configure_subparser(sub_parsers: argparse.ArgumentParser):
    """
    Configure the subparser for the chamaoptimizer command.
    :param sub_parser:
    :param sub_parser: The
    :return:
    """

    sensor_placement_parser = sub_parser.add_parser(
        'sensor',
        help='Sensor placement optimization'
    )

    chama_sensor_placement_subparsers = sensor_placement_parser.add_subparsers(
        title='Chama Sensor Placement',
        description='Sensor placement optimization commands to execute',
        help='Additional help'
    )